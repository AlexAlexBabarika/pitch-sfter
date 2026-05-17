import os, math, time, random, copy
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import TrainConfig
from model import PitchUNet, count_params
from losses import TotalLoss, MultiResMelLoss
from data import mel_pitch_shift, PitchDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
train_config = TrainConfig()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

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

    step = 0
    model.train()
    pbar = tqdm(total=train_config.max_steps)
    while step < train_config.max_steps:
        for batch in train_loader:
            mel_in = batch["mel_in"].to(device, non_blocking=True)
            mel_tgt = batch["mel_tgt"].to(device, non_blocking=True)
            f0 = batch["f0"].to(device, non_blocking=True)
            shift = batch["shift"].to(device, non_blocking=True)

            # Conditioning dropout: 10% of the time, zero out the shift signal
            # so the model can't depend on it to detect identity case.
            if random.random() < train_config.cond_dropout:
                shift = torch.zeros_like(shift)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                pred = model(
                    mel_in, f0, shift, skip_dropout_p=train_config.skip_dropout
                )
                loss, parts = loss_fn(pred, mel_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)

            step += 1
            pbar.update(1)

            if step % train_config.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{parts['l1']:.4f}",
                    mr=f"{parts['mr']:.4f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                )

            if step % train_config.val_every == 0:
                validate(model, ema, val_loader, step)

            if step >= train_config.max_steps:
                break

    save_ckpt(model, ema, step, final=True)
    pbar.close()


@torch.no_grad()
def validate(model, ema, loader, step):
    # Validate with EMA weights
    eval_model = copy.deepcopy(model)
    ema.copy_to(eval_model)
    eval_model.eval()

    total, n = 0.0, 0
    for batch in loader:
        mel_in = batch["mel_in"].to(device)
        mel_tgt = batch["mel_tgt"].to(device)
        f0 = batch["f0"].to(device)
        shift = batch["shift"].to(device)
        pred = eval_model(mel_in, f0, shift)
        total += F.l1_loss(pred, mel_tgt).item() * mel_in.size(0)
        n += mel_in.size(0)
    print(f"\n[val step {step}] L1={total / n:.4f}")
    save_ckpt(model, ema, None, step)


def save_ckpt(model, ema, step, final=False):
    name = "final.pt" if final else f"step_{step}.pt"
    path = Path(train_config.ckpt_dir) / name
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "step": step,
            "lr": train_config.lr,
        },
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
