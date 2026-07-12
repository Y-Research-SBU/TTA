import numpy as np
import torch
from torch import nn
from .sk_batch import SemiCurrSinkhornKnopp_stable, MMOT
from torch.nn import functional as F


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def set_rho(rho_strategy, current, total, rho_upper, rho_base):
    if rho_strategy == "sigmoid":
        rho = sigmoid_rampup(current, total)* rho_upper + rho_base
    elif rho_strategy == "linear":
        rho = current / total * rho_upper + rho_base
    else:
        raise NotImplementedError
    if rho > 1.0:
        rho = min(rho, 1.0)
    return rho

class OT_PseudoLabel(nn.Module):
    def __init__(self,
                 impl="hot",
                 # semantic/similarity options
                 semantic_ops=None,                # e.g., ["rmDiag", "knn"]
                 topk: int = 0,                   # used when knn/topk enabled
                 offset_similarity: float = 0.0,  # affine offset
                 scale_similarity: float = 1.0,   # affine scale
                 # OT schedules
                 rho_strategy: str = "sigmoid",   # "sigmoid" | "linear"
                 rho_base: float = 0.1,
                 rho_upper: float = 1.0,
                 gamma_schedule: str = None,      # None | "linear"
                 gamma_base: float = 1.0,
                 gamma_upper: float = 1.0,
                 # SK/MM hyperparams
                 sk_epsilon: float = 0.1,
                 sk_iter: int = 3,
                 sk_iter_limit: int = 1000,
                 semi_use: bool = True,
                 prior = None,
                 mm_factor: float = 0.5,
                 mm_iter_limit: int = 100,
                 ema: float = 1.0) -> None:
        super().__init__()
        self.impl = impl
        print("ot impl: ", impl)

        # semantic config
        self.semantic_ops = [op.lower() for op in (semantic_ops or [])]
        self.topk = int(topk) if topk is not None else 0
        self.offset_similarity = float(offset_similarity)
        self.scale_similarity = float(scale_similarity)

        # schedule config
        self.rho_strategy = str(rho_strategy)
        self.rho_base = float(rho_base)
        self.rho_upper = float(rho_upper)
        self.gamma_schedule = None if gamma_schedule is None else str(gamma_schedule)
        self.gamma_base = float(gamma_base)
        self.gamma_upper = float(gamma_upper)

        # ot solver config
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iter = int(sk_iter)
        self.sk_iter_limit = int(sk_iter_limit)
        self.semi_use = bool(semi_use)
        self.prior = prior
        self.mm_factor = float(mm_factor)
        self.mm_iter_limit = int(mm_iter_limit)
        self.ema = float(ema)

    @staticmethod
    @torch.no_grad()
    def _get_feature_similarity(feature):
        # feature=feature.detach()
        feature_norm = F.normalize(feature, dim=-1, p=2)
        similarity = feature_norm @ feature_norm.transpose(-2, -1)
        return similarity

    @torch.no_grad()
    def _apply_semantic_ops(self, similarity: torch.Tensor) -> torch.Tensor:
        # similarity: [..., N, N]
        if similarity is None:
            return None
        sim = similarity
        # affine rescale first (optional)
        if (abs(self.offset_similarity) > 1e-12) or (abs(self.scale_similarity - 1.0) > 1e-12):
            sim = (sim + self.offset_similarity) * self.scale_similarity
        for op in self.semantic_ops:
            if op == 'rmdiag':
                eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype)
                sim = sim * (1.0 - eye)
            elif op == 'topk':
                if self.topk and self.topk > 0:
                    vals, idx = sim.topk(min(self.topk, sim.shape[-1]), dim=-1)
                    mask = torch.zeros_like(sim)
                    mask.scatter_(-1, idx, 1.0)
                    # binarize mask
                    sim = sim * mask
            elif op == 'knn':
                if self.topk and self.topk > 0:
                    vals, idx = sim.topk(min(self.topk, sim.shape[-1]), dim=-1)
                    knn = torch.zeros_like(sim)
                    knn.scatter_(-1, idx, vals)
                    sim = knn
            elif op == 'clip':
                sim = torch.clamp(sim, min=0.0)
            elif op == 'upclip':
                sim = torch.clamp(sim, max=1.0)
            elif op == 'consistency':
                # make symmetric by keeping min of (i,j) and (j,i)
                sim = 0.5 * (sim + sim.transpose(-2, -1))
            else:
                # unsupported op name: skip
                pass
        return sim

    def _compute_rho(self, iterations: int, iterations_per_epoch: int) -> float:
        total = 10 * iterations_per_epoch
        current = float(max(0, min(iterations, total)))
        if self.rho_strategy == 'sigmoid':
            phase = 1.0 - current / total if total > 0 else 0.0
            ramp = float(torch.exp(torch.tensor(-5.0 * phase * phase))) if total > 0 else 1.0
        elif self.rho_strategy == 'linear':
            ramp = current / total if total > 0 else 1.0
        else:
            ramp = current / total if total > 0 else 1.0
        rho = ramp * (self.rho_upper - self.rho_base) + self.rho_base
        rho = float(max(0.0, min(1.0, rho)))
        return rho

    def _compute_gamma(self, iterations: int, iterations_per_epoch: int) -> float:
        if self.gamma_schedule is None:
            return self.gamma_base
        total = 10 * iterations_per_epoch
        current = float(max(0, min(iterations, total)))
        if self.gamma_schedule == 'linear':
            val = self.gamma_base + (self.gamma_upper - self.gamma_base) * (current / total if total > 0 else 1.0)
        else:
            val = self.gamma_base
        return float(val)

    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, logit, feature, iterations, iterations_per_epoch):
        """
        Parmas:
            logit : (N, K)
            feature : (N, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "shot":
            rho_value = self._compute_rho(iterations=iterations, iterations_per_epoch=iterations_per_epoch)
            gamma_value = self._compute_gamma(iterations=iterations, iterations_per_epoch=iterations_per_epoch)
            sk = SemiCurrSinkhornKnopp_stable(
                num_iters=self.sk_iter,
                epsilon=self.sk_epsilon,
                gamma=gamma_value,
                stoperr=1e-10,
                numItermax=self.sk_iter_limit,
                rho=rho_value,
                semi_use=self.semi_use,
                prior=self.prior
            )
            sk = MMOT(sk, lam1=self.mm_factor, numItermax=self.mm_iter_limit, lam_fix=False, ema=self.ema)
            feat_sim = self._get_feature_similarity(feature)
            feat_sim = self._apply_semantic_ops(feat_sim)
            pseudo_label = sk(logit, feat_sim)

            return pseudo_label
        
        else:
            raise NotImplementedError

    def forward(self, logit, feature, iterations=0, iterations_per_epoch=20):
        '''
        logit: (N, K) 或 (B, N, K)
        feature: (N, D) 或 (B, N, D)
        '''
        # 检查是否为批量输入
        if logit.dim() == 3:
            # 批量模式: logit shape is (B, N, K), feature shape is (B, N, D)
            batch_size = logit.shape[0]
            is_batch = True
        else:
            # 单样本模式: logit shape is (N, K), feature shape is (N, D)
            batch_size = 1
            is_batch = False
            # 将单样本转换为批量格式
            logit = logit.unsqueeze(0)
            feature = feature.unsqueeze(0)
        
        if is_batch:
            # 批量模式：直接批量处理
            # 重塑为 (B*N, K) 和 (B*N, D) 进行批量处理
            B, N, K = logit.shape
            D = feature.shape[-1]

            # 批量计算pseudo_label
            pseudo_label = self.OT(logit, feature, iterations, iterations_per_epoch)

            return pseudo_label
        else:
            # 单样本模式
            pseudo_label = self.OT(logit[0], feature[0], iterations, iterations_per_epoch)
            return pseudo_label