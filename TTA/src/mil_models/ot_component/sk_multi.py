import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools as it

from .sk_batch import SemiCurrSinkhornKnopp_stable, MMOT


class SKMultiLoss(nn.Module):
    """Parallel OT loss over multiple heads and optional multiple views.

    Inputs:
      - logits_by_view: List[List[Tensor]] where outer list = views, inner list = heads; each Tensor [B, N, K] or [B, K]
      - features_by_view: Optional[List[Tensor]] features per view, used to build similarity S when no global matrix
      - similarity_matrix: Optional[Callable] that maps data_idxs -> dense S submatrix (idx aligned with memory banks)
      - data_idxs: Optional[Tensor] sample indices of current batch

    Behavior:
      - Build one SK/MMOT solver per-head
      - If memory banks are provided, enqueue first view, align other views by not advancing write pointer
      - Compute pseudo labels either on batch or on memory bank window; slice back the current batch portion
      - Sum cross-view CE for all permutations within each head; return per-head losses [H]
    """
    def __init__(self,
                 num_heads: int,
                 sk_type: str = "sppot",
                 ot_frame: str = "mm",
                 sk_iter_limit: int = 1000,
                 epsilon: float = 0.1,
                 rho_base: float = 0.1,
                 rho_upper: float = 1.0,
                 rho_strategy: str = "sigmoid",
                 gamma_base: float = 1.0,
                 gamma_upper: float = 1.0,
                 gamma_schedule: str = None,
                 mm_factor: float = 0.5,
                 mm_iter_limit: int = 100,
                 ema_mm: float = 1.0,
                 logits_bank=None,
                 feature_bank=None,
                 total_iter: int = 100000,
                 start_iter: int = 0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.sk_type = sk_type
        self.ot_frame = ot_frame

        # Build per-head solvers
        self.sk = []
        for _ in range(self.num_heads):
            if sk_type in ["ppot", "sppot", "sppot_stable"]:
                sk = SemiCurrSinkhornKnopp_stable(gamma=gamma_upper, epsilon=epsilon, numItermax=sk_iter_limit, prior=None)
                if ot_frame == "mm":
                    # reduce MM outer loops for stability & speed in early epochs (SPPOT default ~10)
                    mm_loops = max(1, min(10, int(mm_iter_limit)))
                    sk = MMOT(sk, lam1=mm_factor, numItermax=mm_loops, lam_fix=False, ema=ema_mm)
                self.sk.append(sk)
            else:
                raise NotImplementedError(f"Unsupported sk_type: {sk_type}")

        self.logits_bank = logits_bank
        self.feature_bank = feature_bank
        # Note: SK produces soft pseudo-labels q \in R^{B\times K}. We'll use soft-label CE
        # loss = - E_q [log softmax(logits)]. Keep a CE instance unused for BC compatibility.
        self.ce = nn.CrossEntropyLoss()

        # schedules
        self.i = start_iter
        self.total_iter = total_iter
        self.rho_base = float(rho_base)
        self.rho_upper = float(rho_upper) - float(rho_base)
        self.rho_strategy = rho_strategy
        self.gamma_base = float(gamma_base)
        self.gamma_upper = float(gamma_upper) - float(gamma_base)
        self.gamma_schedule = gamma_schedule

        for sk in self.sk:
            sk.set_rho(self.rho_base)

    @torch.no_grad()
    def _set_rho(self, current):
        if self.rho_upper <= 0:
            return
        if self.rho_strategy == "sigmoid":
            # refer to common sigmoid rampup schedule
            cur = max(0, min(current, self.total_iter))
            phase = 1.0 - float(cur) / float(self.total_iter)
            ramp = float(torch.exp(torch.tensor(-5.0 * phase * phase)))
        elif self.rho_strategy == "linear":
            ramp = float(max(0.0, min(1.0, current / max(1, self.total_iter))))
        else:
            ramp = float(max(0.0, min(1.0, current / max(1, self.total_iter))))
        rho = ramp * self.rho_upper + self.rho_base
        for sk in self.sk:
            sk.set_rho(rho)

    @torch.no_grad()
    def _set_gamma(self, current):
        if self.gamma_schedule is None:
            return
        ramp = float(max(0.0, min(1.0, current / max(1, self.total_iter))))
        gamma = self.gamma_base + ramp * self.gamma_upper
        for sk in self.sk:
            sk.set_gamma(gamma)

    @staticmethod
    @torch.no_grad()
    def _feature_similarity(feat_bag: torch.Tensor) -> torch.Tensor:
        # feat_bag: [B, D]; use bag-level pooled features to build [B,B] similarity
        feat_bag = torch.nan_to_num(feat_bag)
        feat_bag = F.normalize(feat_bag, dim=-1, p=2, eps=1e-12)
        sim = feat_bag @ feat_bag.t()
        return torch.nan_to_num(sim)

    def forward(self, logits_by_view, features_by_view=None, similarity_matrix=None, data_idxs=None):
        # logits_by_view: List[List[Tensor]]; inner Tensor either [B, N, K] or [B, K]
        if len(logits_by_view) == 0:
            return torch.zeros(self.num_heads, device=logits_by_view)

        batch_size = logits_by_view[0][0].shape[0]
        # schedules
        self._set_rho(self.i)
        self._set_gamma(self.i)
        self.i += 1

        # Ensure logits are [B, K] before CE, while SK consumes memory/batch [*, K]
        def to_bag_logits(x):
            # if [B, N, K], pool over N by mean; otherwise return [B, K]
            if x.dim() == 3:
                return x.mean(dim=1)
            return x

        # compute pseudo labels per view/head (batched over [B,K])
        pseudo_labels = []
        for view_id, heads in enumerate(logits_by_view):
            pl_view = []
            # build similarity
            feat_sim = None
            if self.logits_bank is None:
                if similarity_matrix is not None and data_idxs is not None:
                    feat_sim = similarity_matrix(data_idxs)
                elif features_by_view is not None and features_by_view[view_id] is not None:
                    # features_by_view should provide [B, D] pooled bag features for speed
                    feat_sim = self._feature_similarity(features_by_view[view_id])
            for head_id, head_logits in enumerate(heads):
                # prepare logits block for SK
                if self.logits_bank is None:
                    # batch-only: use [B,K]
                    logits_for_sk = to_bag_logits(head_logits).detach()
                    logits_for_sk = torch.nan_to_num(logits_for_sk)
                    if feat_sim is None:
                        q = self.sk[head_id](logits_for_sk)
                    else:
                        q = self.sk[head_id](logits_for_sk, feat_sim)
                    pl_view.append(torch.nan_to_num(q))
                else:
                    # memory path: enqueue on first view only
                    memory, memory_idx, write_idx = self.logits_bank[head_id](to_bag_logits(head_logits), enqueue=True if view_id == 0 else False, data_idxs=data_idxs)
                    memory = torch.nan_to_num(memory)
                    if (features_by_view is None or features_by_view[view_id] is None) and similarity_matrix is not None:
                        S = similarity_matrix(memory_idx)
                        S = torch.nan_to_num(S)
                    else:
                        # feature bank preferred for S; features_by_view expected to be bag-level [B,D]
                        fmemory, fidx, f_write_idx = self.feature_bank(features_by_view[view_id], enqueue=True if view_id == 0 else False, data_idxs=data_idxs)
                        fmemory = torch.nan_to_num(fmemory)
                        S = self._feature_similarity(fmemory)
                    q_full = self.sk[head_id](memory, S)
                    q_full = torch.nan_to_num(q_full)
                    if write_idx == 0:
                        q = q_full[-batch_size:, :]
                    else:
                        q = q_full[write_idx - batch_size:write_idx, :]
                    pl_view.append(torch.nan_to_num(q))
            pseudo_labels.append(pl_view)

        # cross-view CE per head over all permutations
        loss_per_head = []
        # logits for CE need [B, K]; labels are soft distributions [B, K]
        def _safe_bag(x: torch.Tensor) -> torch.Tensor:
            x = to_bag_logits(x)
            return torch.nan_to_num(x)
        logits_bag = [[_safe_bag(h) for h in v] for v in logits_by_view]
        for head_id, (head_logits_across_views, head_labels_across_views) in enumerate(zip(zip(*logits_bag), zip(*pseudo_labels))):
            loss = 0.0
            V = len(head_logits_across_views)
            # helper: soft-label cross entropy
            def soft_ce(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
                logits = torch.nan_to_num(logits)
                log_prob = F.log_softmax(logits, dim=1)
                target_prob = torch.nan_to_num(target_prob)
                target_prob = target_prob.clamp_min(1e-8)
                # renormalize rows to sum 1 to avoid degenerate zeros after nan_to_num
                target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
                return -(target_prob * log_prob).sum(dim=1).mean()
            if V <= 1:
                # single view: use its own label
                loss = soft_ce(head_logits_across_views[0], head_labels_across_views[0])
            else:
                for a, b in it.permutations(range(V), 2):
                    loss = loss + soft_ce(head_logits_across_views[a], head_labels_across_views[b])
            # normalize by number of cross-view pairs to stabilize scale
            norm = max(1, V * (V - 1))
            loss_per_head.append(loss / norm)
        return torch.stack(loss_per_head)


