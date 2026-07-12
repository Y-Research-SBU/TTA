import numpy as np
import torch
from torch import nn
from .sinkhornknopp import SemiCurrSinkhornKnopp_stable


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def set_rho(rho_strategy, current, total, rho_upper, rho_base):
    if rho_strategy == "sigmoid":
        rho = sigmoid_rampup(current, total) * rho_upper + rho_base
    elif rho_strategy == "linear":
        rho = current / total * rho_upper + rho_base
    else:
        raise NotImplementedError
    if rho > 1.0:
        rho = min(rho, 1.0)
    return rho


class OT_Attn(nn.Module):
    def __init__(self, impl="hot") -> None:
        super().__init__()
        if impl in ["batch", "batchot"]:
            impl = "batchot"
        self.impl = impl
        print("ot impl: ", impl)
    
    def normalize_feature(self, x, mode="l2"):
        # Subtract-min only when mode=='none', otherwise l2 optional
        if mode == "none":
            return x - x.min(-1)[0].unsqueeze(-1)
        x = x - x.min(-1)[0].unsqueeze(-1)
        if mode == "l2":
            denom = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
            x = x / denom
        return x

    def OT(self, weight1, weight2, iterations, iterations_per_epoch):
        """
        Parmas:
            weight1 : (B, N, D)
            weight2 : (B, M, D)

        Returns:
            flow: (N, M)
            dist: (1,)
        """
        if self.impl in ["hot", "batchot"]:
            rho_value = set_rho(rho_strategy="sigmoid", current=iterations, total=10 * iterations_per_epoch, rho_upper=1, rho_base=0.1)
            sk = SemiCurrSinkhornKnopp_stable(num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-10, numItermax=1000, rho=rho_value, semi_use=True, prior=None)

            cost = torch.cdist(weight1, weight2)
            denom = cost.max().clamp_min(1e-12)
            cost = cost / denom

            flow, _ = sk.cost_forward(cost, final=False, count=True)

            flow = flow / rho_value

            cost = cost.type(torch.FloatTensor).to(weight1.device)
            flow = flow.type(torch.FloatTensor).to(weight1.device)
            dist = cost * flow
            dist = torch.sum(dist)
            return flow, dist

        else:
            raise NotImplementedError

    def forward(self, x, y, iterations=0, iterations_per_epoch=20, feat_norm_mode="l2"):
        """
        x: (B, N, D)
        y: (B, M, D)
        """
        x = self.normalize_feature(x, mode=feat_norm_mode)
        y = self.normalize_feature(y, mode=feat_norm_mode)

        pi, dist = self.OT(x, y, iterations, iterations_per_epoch)
        return pi, dist


