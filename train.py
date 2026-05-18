import math
import os
import re
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from config import TrainConfig
from model import PitchUNet, count_params
from losses import TotalLoss
from data import PitchDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
train_config = TrainConfig()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        # Ramp decay so early EMA isn't dominated by the random init.
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])


def lr_lambda(step):
    if step < train_config.warmup_steps:
        return step / max(1, train_config.warmup_steps)
    # cosine to 10% of base
    progress = (step - train_config.warmup_steps) / max(
        1, train_config.max_steps - train_config.warmup_steps
    )
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    model = PitchUNet().to(device)
    eval_model = PitchUNet().to(device)
    print(f"Model: {count_params(model) / 1e6:.2f} M params")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = TotalLoss().to(device)
    ema = EMA(model, decay=train_config.ema_decay)

    train_loader = PitchDataset.make_loader("train")
    val_loader = PitchDataset.make_loader("val")

    writer = SummaryWriter(log_dir=train_config.tb_dir)

    step = 0
    ckpt = load_latest(Path(train_config.ckpt_dir))
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        ema.shadow = ckpt["ema"]["shadow"]
        ema.step = ckpt["ema"]["step"]
        step = ckpt["step"]
        print(f"resumed from step {step}")

    model.train()
    pbar = tqdm(total=train_config.max_steps, initial=step)
    while step < train_config.max_steps:
        for batch in train_loader:
            mel_in = batch["mel_in"].to(device, non_blocking=True)
            mel_tgt = batch["mel_tgt"].to(device, non_blocking=True)
            f0 = batch["f0"].to(device, non_blocking=True)
            shift = batch["shift"].to(device, non_blocking=True)

            # Conditioning dropout: zero the shift and flag those samples via
            # cond_mask so the model can distinguish "dropped" from "shift=0".
            keep = (
                torch.rand(shift.shape[0], device=shift.device)
                >= train_config.cond_dropout
            ).to(shift.dtype)
            shift = shift * keep

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                pred = model(
                    mel_in,
                    f0,
                    shift,
                    keep,
                    skip_dropout_p=train_config.skip_dropout,
                )
                loss, parts = loss_fn(pred, mel_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )
            opt.step()
            sched.step()
            ema.update(model)

            step += 1
            pbar.update(1)

            if step % train_config.log_every == 0:
                lr = sched.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{parts['l1']:.4f}",
                    mr=f"{parts['mr']:.4f}",
                    lr=f"{lr:.2e}",
                )
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/l1", parts["l1"], step)
                writer.add_scalar("train/mr", parts["mr"], step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            if step % train_config.save_every == 0:
                save_ckpt(model, ema, opt, sched, step)

            if step % train_config.val_every == 0:
                validate(model, ema, eval_model, val_loader, step, writer)

            if step >= train_config.max_steps:
                break

    save_ckpt(model, ema, opt, sched, step, final=True)
    pbar.close()
    writer.close()


@torch.no_grad()
def validate(model, ema, eval_model, loader, step, writer):
    # Validate with EMA weights — reuse a single eval_model instead of
    # deep-copying `model` each call.
    ema.copy_to(eval_model)
    eval_model.eval()

    total, n = 0.0, 0
    for batch in loader:
        mel_in = batch["mel_in"].to(device)
        mel_tgt = batch["mel_tgt"].to(device)
        f0 = batch["f0"].to(device)
        shift = batch["shift"].to(device)
        cond_mask = torch.ones_like(shift)
        pred = eval_model(mel_in, f0, shift, cond_mask)
        total += F.l1_loss(pred, mel_tgt).item() * mel_in.size(0)
        n += mel_in.size(0)
    if n == 0:
        print(f"\n[val step {step}] empty val set — skipping")
    else:
        val_l1 = total / n
        print(f"\n[val step {step}] L1={val_l1:.4f}")
        writer.add_scalar("val/l1", val_l1, step)


_STEP_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")


def _step_ckpts(ckpt_dir: Path):
    out = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = _STEP_CKPT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def save_ckpt(model, ema, opt, sched, step, final=False):
    name = "final.pt" if final else f"step_{step}.pt"
    ckpt_dir = Path(train_config.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "ema": {"shadow": ema.shadow, "step": ema.step},
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)

    if not final:
        # Prune oldest step_*.pt so at most `keep_last` remain.
        ckpts = _step_ckpts(ckpt_dir)
        for _, old in ckpts[: -train_config.keep_last]:
            old.unlink(missing_ok=True)


def load_latest(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    if (ckpt_dir / "final.pt").exists():
        print(
            f"{ckpt_dir / 'final.pt'} exists — training already complete. "
            "Delete it to start a new run.",
            file=sys.stderr,
        )
        sys.exit(0)
    ckpts = _step_ckpts(ckpt_dir)
    if not ckpts:
        return None
    _, path = ckpts[-1]
    print(f"loading {path}")
    return torch.load(path, map_location=device, weights_only=False)


if __name__ == "__main__":
    main()
