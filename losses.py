import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResMelLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for s in self.scales:
            if s == 1:
                p, t = pred, tgt
            else:
                p = F.avg_pool2d(pred, kernel_size=(s, 1))
                t = F.avg_pool2d(tgt, kernel_size=(s, 1))
            loss = loss + F.l1_loss(p, t)
        # pyrefly: ignore [bad-return]
        return loss / len(self.scales)


class TotalLoss(nn.Module):
    def __init__(self, w_l1: float = 1.0, w_mr: float = 0.5):
        super().__init__()
        self.mr = MultiResMelLoss()
        self.w_l1 = w_l1
        self.w_mr = w_mr

    def forward(self, pred, tgt):
        l1 = F.l1_loss(pred, tgt)
        mr = self.mr(pred, tgt)
        return self.w_l1 * l1 + self.w_mr * mr, {"l1": l1.item(), "mr": mr.item()}
