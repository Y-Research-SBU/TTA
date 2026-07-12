import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_component.ot_pseudolabel import OT_PseudoLabel
from .ot_component.sk_batch import SemiCurrSinkhornKnopp_stable
from .fusion import FusionHead
from .components import MMAttentionLayer, FeedForward, process_surv

# Numerical safety helpers for aggregation
def _safe_row_normalize(w: torch.Tensor) -> torch.Tensor:
    # Ensure rows are valid probability distributions
    w = torch.nan_to_num(w, 0.0, 0.0, 0.0)
    row_sum = w.sum(dim=-1, keepdim=True)
    bad = row_sum <= 1e-16
    if bad.any():
        # fallback to uniform to avoid all-zero rows
        uniform = torch.full_like(w, 1.0 / max(1, w.size(-1)))
        w = torch.where(bad, uniform, w)
        row_sum = w.sum(dim=-1, keepdim=True)
    return w / row_sum.clamp_min(1e-12)

def _safe_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Only sanitize weights; keep features raw to preserve distribution
    A = torch.nan_to_num(A, 0.0, 0.0, 0.0)
    return torch.bmm(A, B)

def _safe_soft_ce(logits: torch.Tensor, soft_targets: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Sanitize targets and lightly normalize valid rows (no extra scaling)
    t = torch.nan_to_num(soft_targets, 0.0, 0.0, 0.0).clamp_min(1e-12)
    t_sum = t.sum(dim=dim, keepdim=True)
    norm_mask = (t_sum > 1e-6)
    t = torch.where(norm_mask, t / t_sum.clamp_min(1e-12), t)
    logp = F.log_softmax(logits, dim=dim)
    ce = -(t * logp).sum(dim=dim)
    ce = torch.nan_to_num(ce, 0.0, 0.0, 0.0)
    return ce.mean()

def _safe_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    # Softmax with NaN guards (no manual fallback to uniform)
    out = torch.softmax(x, dim=dim)
    return torch.nan_to_num(out, 0.0, 0.0, 0.0)

@torch.no_grad()
def _batch_kmeans_assign(x: torch.Tensor, K: int, iters: int = 2) -> torch.Tensor:
    """
    Simple mini-batch KMeans to produce hard one-hot assignments per batch.
    x: [B, N, D] features
    Return: one-hot weights [B, N, K]
    """
    x = torch.nan_to_num(x)
    B, N, D = x.shape
    if K <= 0:
        raise ValueError("K must be > 0 for kmeans")
    # Init centroids by random subset (deterministic fallback if N<K)
    device = x.device
    weights = torch.zeros(B, N, K, device=device, dtype=x.dtype)
    for b in range(B):
        xb = x[b]  # [N, D]
        if N <= 0:
            continue
        if N >= K:
            idx = torch.randperm(N, device=device)[:K]
        else:
            # repeat indices if not enough tokens
            base = torch.arange(N, device=device)
            idx = base.repeat((K + N - 1) // N)[:K]
        centroids = xb.index_select(0, idx).clone()  # [K, D]
        for _ in range(max(1, iters)):
            # assign
            # distances: [N, K]
            d2 = torch.cdist(xb.unsqueeze(0), centroids.unsqueeze(0), p=2.0).squeeze(0)
            assign = torch.argmin(d2, dim=1)  # [N]
            # recompute
            new_centroids = []
            for k in range(K):
                mask = (assign == k)
                if mask.any():
                    new_centroids.append(xb[mask].mean(dim=0))
                else:
                    # re-init empty cluster to a random point
                    ridx = torch.randint(0, N, (1,), device=device)
                    new_centroids.append(xb[ridx].squeeze(0))
            centroids = torch.stack(new_centroids, dim=0)
        # one-hot
        onehot = torch.zeros(N, K, device=device, dtype=xb.dtype).scatter_(1, assign.unsqueeze(-1), 1.0)
        weights[b] = onehot
    return weights

class OTMIL_Label(nn.Module):
    """OTMIL pseudo-label model, supporting image and/or omics branches.

    - For image: instance -> K prototypes via logits and OT; optional instance-level CE with pseudo labels.
    - For omics: pathway tokens -> K prototypes similarly; optional instance-level CE.
    - Head: for histo-only, Linear(K->1) then cls; for multimodal, concat or mean then cls.
    """

    def __init__(self,
                 histo_in_dim=1024,
                 hidden_dim=256,
                 dropout=0.1,
                 num_classes=1,
                 ot_mode='ubot',
                 rho_fixed=0.1,
                 image_proto_num=16,
                 num_heads=1,
                 omics_proto_num=16,
                 use_image=True,
                 use_omics=False,
                 image_pseudo_label=0,
                 omics_pseudo_label=0,
                 image_ot_impl='batchot',
                 omics_ot_impl='batchot',
                 omic_sizes=None,
                 fusion_type='concat',
                 label_num_coattn_layers=1,
                 # modality-context refinement (anchor + self-attn)
                 enable_modality_refine=0,
                 modref_weight=0.0,
                 modref_tau=0.1,
                 modref_layers=1,
                 modref_apply_to_tokens=1,
                 # OT as aggregation weights
                 use_ot_as_weights=0,
                 ot_weight_strategy='mix',
                 ot_mix_coeff=0.3,
                 # Differentiable OT-attention (trainable transport weights)
                 ot_attn_trainable=0,
                 proto_ortho_weight=0.0,
                 # shared prototype switches (Scheme A)
                 shared_prototypes=0,
                 shared_proto_num=16,
                 shared_proto_dim=256,
                 shared_tau=1.0,
                 joint_ot_single_path=0,
                 # OT schedules 
                 rho_strategy='sigmoid',
                 rho_base=0.1,
                 rho_upper=1.0,
                 gamma_schedule=None,
                 gamma_base=1.0,
                 gamma_upper=1.0,
                 sk_epsilon=0.1,
                 sk_iter=3,
                 sk_iter_limit=1000,
                 mm_factor=0.5,
                 mm_iter_limit=100,
                 ema=1.0,
                 wsi_use_ce=0,
                 wsi_ce_weight=0.0,
                 omics_use_ce=0,
                 omics_ce_weight=0.0
                 ):
        super().__init__()

        self.use_image = use_image
        self.use_omics = use_omics
        self.image_pseudo_label = int(image_pseudo_label)
        self.omics_pseudo_label = int(omics_pseudo_label)
        self.num_heads = int(num_heads)
        # shared prototype configs
        self.shared_prototypes = int(shared_prototypes)
        self.shared_proto_num = int(shared_proto_num)
        self.shared_proto_dim = int(shared_proto_dim)
        self.shared_tau = float(shared_tau)
        self.joint_ot_single_path = int(joint_ot_single_path)
        self.rho_strategy = str(rho_strategy)
        self.rho_base = float(rho_base)
        self.rho_upper = float(rho_upper)
        self.gamma_schedule = None if gamma_schedule is None else str(gamma_schedule)
        self.gamma_base = float(gamma_base)
        self.gamma_upper = float(gamma_upper)
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iter = int(sk_iter)
        self.sk_iter_limit = int(sk_iter_limit)
        self.mm_factor = float(mm_factor)
        self.mm_iter_limit = int(mm_iter_limit)
        self.ema = float(ema)
        # consistency removed
        self.wsi_use_ce = int(wsi_use_ce)
        self.wsi_ce_weight = float(wsi_ce_weight)
        self.omics_use_ce = int(omics_use_ce)
        self.omics_ce_weight = float(omics_ce_weight)

        self.histo_in_dim = histo_in_dim
        self.path_proj_dim = hidden_dim
        self.num_classes = num_classes
        self.ot_mode = str(ot_mode)
        self.rho_fixed = float(rho_fixed)
        # No dynamic adapters; rely on explicit in_dim matching
        # Control for token-level attention in label route: only by fusion_type
        self.fusion_type = fusion_type
        self.label_use_coattn = (str(self.fusion_type).lower() == 'coattn')
        self.label_num_coattn_layers = int(label_num_coattn_layers)

        # Modality-context refinement configuration
        self.enable_modality_refine = int(enable_modality_refine)
        self.modref_weight = float(modref_weight)
        self.modref_tau = float(modref_tau)
        self.modref_layers = int(modref_layers)
        self.modref_apply_to_tokens = int(modref_apply_to_tokens)
        # OT-as-weights configuration
        self.use_ot_as_weights = int(use_ot_as_weights)
        self.ot_weight_strategy = str(ot_weight_strategy)
        self.ot_mix_coeff = float(ot_mix_coeff)
        # Differentiable OT-attn
        self.ot_attn_trainable = int(ot_attn_trainable)
        self.proto_ortho_weight = float(proto_ortho_weight)

        # Image branch 
        if self.use_image:
            self.image_proto_num = image_proto_num
            self.image_patch_encoder = nn.Sequential(
                nn.Linear(histo_in_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            # Learnable instance -> prototype classifiers (multi-head)
            if self.shared_prototypes == 0:
                if self.num_heads <= 1:
                    self.image_patch_classifiers = nn.ModuleList([
                        nn.Linear(hidden_dim, image_proto_num)
                    ])
                else:
                    self.image_patch_classifiers = nn.ModuleList([
                        nn.Linear(hidden_dim, image_proto_num) for _ in range(self.num_heads)
                    ])
            # Linear aggregator over prototypes K->1
            self.image_linear = nn.Sequential(
                nn.Linear((self.shared_proto_num if self.shared_prototypes == 1 else image_proto_num), 1),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            # Final classifier for survival
            self.image_head = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            # Pseudo-label generator (MMOT/SHOT)
            self.image_ot_pseudo = None
            if self.image_pseudo_label == 1 and OT_PseudoLabel is not None:
                # Align pseudo-label OT to outer ot_mode/rho settings
                _rho_strategy = self.rho_strategy
                _rho_base = self.rho_base
                _rho_upper = self.rho_upper
                _semi_use = True
                if self.ot_mode == 'ubot_fixed_rho':
                    _rho_strategy = 'linear'
                    _rho_base = self.rho_fixed
                    _rho_upper = self.rho_fixed
                elif self.ot_mode == 'balanced':
                    _semi_use = False
                self.image_ot_pseudo = OT_PseudoLabel(
                    impl='shot',
                    rho_strategy=_rho_strategy,
                    rho_base=_rho_base,
                    rho_upper=_rho_upper,
                    gamma_schedule=self.gamma_schedule,
                    gamma_base=self.gamma_base,
                    gamma_upper=self.gamma_upper,
                    sk_epsilon=self.sk_epsilon,
                    sk_iter=self.sk_iter,
                    sk_iter_limit=self.sk_iter_limit,
                    semi_use=_semi_use,
                    prior=None,
                    mm_factor=self.mm_factor,
                    mm_iter_limit=self.mm_iter_limit,
                    ema=self.ema,
                )
            # Shared prototype projection for image
            if self.shared_prototypes == 1:
                self.shared_proj_img = nn.Linear(hidden_dim, self.shared_proto_dim)
                # Enable multi-head with shared prototypes by adding per-head projections
                if self.num_heads > 1:
                    # head 0 uses shared_proj_img; extra heads use their own projections
                    self.shared_proj_img_extra_heads = nn.ModuleList([
                        nn.Linear(hidden_dim, self.shared_proto_dim) for _ in range(self.num_heads - 1)
                    ])
                else:
                    self.shared_proj_img_extra_heads = None

        # Omics branch 
        if self.use_omics:
            self.omics_proto_num = omics_proto_num
            # Pathway tokens are provided; use identity encoder
            self.omics_patch_encoder = nn.Identity()
            # Build per-path linear to hidden_dim to equalize dims when sizes vary
            self.omics_per_path = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(s, hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout)
                ) for s in (omic_sizes if isinstance(omic_sizes, (list, tuple)) else [])
            ])
            if self.shared_prototypes == 0:
                if self.num_heads <= 1:
                    self.omics_patch_classifiers = nn.ModuleList([
                        nn.Linear(hidden_dim, omics_proto_num)
                    ])
                else:
                    self.omics_patch_classifiers = nn.ModuleList([
                        nn.Linear(hidden_dim, omics_proto_num) for _ in range(self.num_heads)
                    ])
            self.omics_linear = nn.Sequential(
                nn.Linear((self.shared_proto_num if self.shared_prototypes == 1 else omics_proto_num), 1),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            self.omics_head = nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            self.omics_ot_pseudo = None
            if self.omics_pseudo_label == 1 and OT_PseudoLabel is not None:
                _rho_strategy = self.rho_strategy
                _rho_base = self.rho_base
                _rho_upper = self.rho_upper
                _semi_use = True
                if self.ot_mode == 'ubot_fixed_rho':
                    _rho_strategy = 'linear'
                    _rho_base = self.rho_fixed
                    _rho_upper = self.rho_fixed
                elif self.ot_mode == 'balanced':
                    _semi_use = False
                self.omics_ot_pseudo = OT_PseudoLabel(
                    impl='shot',
                    rho_strategy=_rho_strategy,
                    rho_base=_rho_base,
                    rho_upper=_rho_upper,
                    gamma_schedule=self.gamma_schedule,
                    gamma_base=self.gamma_base,
                    gamma_upper=self.gamma_upper,
                    sk_epsilon=self.sk_epsilon,
                    sk_iter=self.sk_iter,
                    sk_iter_limit=self.sk_iter_limit,
                    semi_use=_semi_use,
                    prior=None,
                    mm_factor=self.mm_factor,
                    mm_iter_limit=self.mm_iter_limit,
                    ema=self.ema,
                )
            # Shared prototype projection for omics
            if self.shared_prototypes == 1:
                self.shared_proj_omics = nn.Linear(hidden_dim, self.shared_proto_dim)
                if self.num_heads > 1:
                    self.shared_proj_omics_extra_heads = nn.ModuleList([
                        nn.Linear(hidden_dim, self.shared_proto_dim) for _ in range(self.num_heads - 1)
                    ])
                else:
                    self.shared_proj_omics_extra_heads = None

        # Shared prototype bank (learnable) for both modalities
        if self.shared_prototypes == 1:
            self.shared_proto_bank = nn.Parameter(torch.randn(self.shared_proto_num, self.shared_proto_dim))

        # Fusion / classifier
        self.fusion_type = fusion_type
        if self.use_image and self.use_omics:
            if str(self.fusion_type).lower() in ['concat', 'sum', 'mlp']:
                self.fusion_head = FusionHead(fusion_type=fusion_type, in_dim=hidden_dim, out_dim=num_classes, dropout=dropout)
                self.coattn_head = None
            elif str(self.fusion_type).lower() == 'coattn':
                # Token-level attention path will pool to two [B,D] features; classify on concatenation
                self.fusion_head = None
                self.coattn_head = nn.Linear(hidden_dim * 2, num_classes, bias=False)
            else:
                raise ValueError(f"Unsupported fusion_type for label route: {self.fusion_type}")
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

        # Build optional co-attention stack for label route
        if self.label_use_coattn:
            # Determine number of tokens for omics branch to separate after attention
            if self.shared_prototypes == 1:
                self._label_num_path_tokens = (self.shared_proto_num if self.use_omics else 0)
            else:
                self._label_num_path_tokens = self.omics_proto_num if self.use_omics else 0
            co_layers = []
            # Use a single MMAttentionLayer followed by simple FeedForward and LayerNorm
            # mirroring the lightweight setup used in proto path
            attn_mode = 'full' if (self.use_image and self.use_omics) else 'self'
            co_layers.append(MMAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=self.path_proj_dim,
                heads=1,
                residual=False,
                dropout=dropout,
                num_pathways=self._label_num_path_tokens,
                attn_mode=attn_mode
            ))
            co_layers.append(FeedForward(self.path_proj_dim, 1, dropout=dropout))
            co_layers.append(nn.LayerNorm(int(self.path_proj_dim)))
            self.label_coattn = nn.Sequential(*co_layers)
        else:
            self.label_coattn = None

        # Build optional modality-context refinement blocks and anchors
        # These are lightweight self-attention refiners applied to per-modality K tokens
        self.modality_anchor = nn.Parameter(torch.randn(2, self.path_proj_dim))
        def _build_refiner(n_layers: int):
            layers = []
            for _ in range(max(1, int(n_layers))):
                layers.append(MMAttentionLayer(
                    dim=self.path_proj_dim,
                    dim_head=self.path_proj_dim,
                    heads=1,
                    residual=False,
                    dropout=dropout,
                    num_pathways=0,
                    attn_mode='self'
                ))
                layers.append(FeedForward(self.path_proj_dim, 1, dropout=dropout))
                layers.append(nn.LayerNorm(int(self.path_proj_dim)))
            return nn.Sequential(*layers)
        self.img_refiner = _build_refiner(self.modref_layers)
        self.omics_refiner = _build_refiner(self.modref_layers)
        # small projection head for anchor alignment
        self.modref_proj = nn.Sequential(
            nn.Linear(self.path_proj_dim, self.path_proj_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.path_proj_dim, self.path_proj_dim)
        )

    def forward_no_loss(self, x_path, x_omics, return_attn=False, attn_mask=None, iterations=None, iterations_per_epoch=None, forward_pass=None, feature_archive=None, archive_k: int = 5, prototype_archive=None, proto_contrast_k: int = 0, proto_contrast_weight: float = 0.0):
        device = x_path[0].device if isinstance(x_path, list) and len(x_path) > 0 else (x_path.device if torch.is_tensor(x_path) else (x_omics[0].device if isinstance(x_omics, list) and len(x_omics) > 0 else torch.device('cpu')))
        iters = 0 if iterations is None else int(iterations)
        iters_per_ep = 20 if iterations_per_epoch is None else int(iterations_per_epoch)

        out = {}
        features = []
        # Optional token buffers for co-attn path
        img_tokens = None
        omic_tokens = None
        # Initialize pseudo distributions to avoid unbound references when branches are disabled
        pseudo_img = None
        pseudo_omics = None

        mask_img = None
        mask_tensor_cache = None
        if isinstance(attn_mask, list) and len(attn_mask) > 0:
            if torch.is_tensor(attn_mask[0]):
                mask_tensor_cache = torch.stack([m.float() for m in attn_mask], dim=0)
        elif torch.is_tensor(attn_mask):
            mask_tensor_cache = attn_mask.float()

        if self.use_image:
            feats = x_path if torch.is_tensor(x_path) else (x_path[0] if (isinstance(x_path, list) and len(x_path) == 1) else torch.stack(x_path, dim=0))
            encoded = self.image_patch_encoder(feats)
            encoded = torch.nan_to_num(encoded)
            # Ensure 3D tokens [B, N, D]
            encoded_3d = encoded.unsqueeze(0) if encoded.dim() == 2 else encoded
            if mask_tensor_cache is not None:
                mask_img = mask_tensor_cache.to(encoded_3d.device)
                if mask_img.dim() == 1:
                    mask_img = mask_img.unsqueeze(0)
                if mask_img.shape[1] != encoded_3d.shape[1]:
                    expanded_mask = torch.zeros(mask_img.size(0), encoded_3d.size(1), device=mask_img.device, dtype=mask_img.dtype)
                    copy_cols = min(mask_img.size(1), encoded_3d.size(1))
                    expanded_mask[:, :copy_cols] = mask_img[:, :copy_cols]
                    mask_img = expanded_mask
                mask_img = mask_img.unsqueeze(-1)
                encoded_3d = encoded_3d * mask_img
            # Multi-head patch logits [B, N, K]
            if self.shared_prototypes == 1:
                # project to shared space and compute logits via shared proto bank
                proto_norm = F.normalize(self.shared_proto_bank, dim=-1)
                patch_logits_heads = []
                # head 0 uses base shared projection
                proj_img_0 = self.shared_proj_img(encoded_3d)
                proj_img_0 = F.normalize(proj_img_0, dim=-1)
                patch_logit_0 = torch.einsum('bnd,kd->bnk', proj_img_0, proto_norm) / max(self.shared_tau, 1e-6)
                patch_logits_heads.append(patch_logit_0)
                # additional heads (if any) use extra per-head projections
                if (hasattr(self, 'shared_proj_img_extra_heads') and (self.shared_proj_img_extra_heads is not None)) and (self.num_heads > 1):
                    for lin in self.shared_proj_img_extra_heads:
                        proj_img_h = lin(encoded_3d)
                        proj_img_h = F.normalize(proj_img_h, dim=-1)
                        patch_logit_h = torch.einsum('bnd,kd->bnk', proj_img_h, proto_norm) / max(self.shared_tau, 1e-6)
                        patch_logits_heads.append(patch_logit_h)
            else:
                patch_logits_heads = [cls(encoded_3d) for cls in self.image_patch_classifiers]
            if mask_img is not None:
                mask_expand = mask_img.squeeze(-1).unsqueeze(-1)
                patch_logits_heads = [p.masked_fill(mask_expand == 0, float('-inf')) for p in patch_logits_heads]

            # Optional: expose encoded tokens for external OT (do not early return; allow omics branch to add its outputs too)
            if forward_pass == 'return_all':
                out['patch_logits_heads'] = patch_logits_heads  # list of [B, N, K]
                out['encoded_features'] = encoded_3d            # [B, N, D]
                out['bag_features'] = torch.nan_to_num(encoded_3d.mean(dim=1))
                if mask_img is not None:
                    out['attn_mask'] = mask_img.squeeze(-1)
            else:
                out['patch_logits_heads'] = patch_logits_heads
                out['encoded_features'] = encoded_3d
                if mask_img is not None:
                    out['attn_mask'] = mask_img.squeeze(-1)

            # Instance-level pseudo labels retained for potential downstream use, but CE is disabled
            pseudo_img = None
            # Select head: default 0, or use epoch-selected head from trainer (SPPOT-style)
            sel_head = int(getattr(self, 'selected_head', 0))
            if sel_head < 0 or sel_head >= len(patch_logits_heads):
                sel_head = 0
            patch_logit_primary = patch_logits_heads[sel_head]
            if (self.ot_attn_trainable == 0) and (self.image_pseudo_label == 1):
                if self.image_ot_pseudo is not None:
                    with torch.no_grad():
                        feat_for_ot = (proj_img_0 if self.shared_prototypes == 1 else encoded_3d)
                        if (self.shared_prototypes == 1) and (self.joint_ot_single_path == 1) and self.use_omics and (self.omics_pseudo_label == 1):
                            out['__img_logit_joint__'] = patch_logit_primary.detach()
                            out['__img_feat_joint__'] = feat_for_ot.detach()
                        else:
                            pseudo_img = self.image_ot_pseudo(patch_logit_primary.detach(), feat_for_ot.detach(), iterations=iters, iterations_per_epoch=iters_per_ep)
                else:
                    pseudo_img = torch.softmax(patch_logit_primary.detach() * 2.0, dim=-1)

            # Aggregate with primary head; optionally use OT as weights
            defer_img_agg = (
                (self.shared_prototypes == 1) and (self.joint_ot_single_path == 1) and self.use_omics and (
                    ((self.omics_pseudo_label == 1) and (self.use_ot_as_weights == 1) and (self.ot_attn_trainable == 0))
                    or (self.ot_attn_trainable == 1)
                )
            )
            if defer_img_agg:
                # Defer image aggregation until joint pseudo is computed in omics branch
                out['__img_defer_agg__'] = True
                out['__img_encoded_to_agg__'] = encoded_3d
                # store softmax weight only for non-trainable pseudo joint path
                if self.ot_attn_trainable == 0:
                    out['__img_weight_soft__'] = _safe_softmax(patch_logit_primary, dim=2)
                # store logits for trainable joint OT-attn
                out['__img_logit_train__'] = patch_logit_primary
                # do not set h_path/img_tokens/features yet
                h_path = None
            else:
                # KMeans-style hard assignment (ablation)
                if self.ot_mode == 'kmeans':
                    # run lightweight KMeans over encoded features to produce hard one-hot weights
                    weight = _batch_kmeans_assign(encoded_3d, (self.shared_proto_num if self.shared_prototypes == 1 else self.image_proto_num), iters=2)
                else:
                    weight = _safe_softmax(patch_logit_primary, dim=2)
                if self.ot_attn_trainable == 1:
                    cost_img = -patch_logit_primary
                    rho_value = (self.rho_fixed if self.ot_mode == 'ubot_fixed_rho' else self.rho_upper)
                    sk = SemiCurrSinkhornKnopp_stable(num_iters=self.sk_iter, epsilon=self.sk_epsilon, gamma=self.gamma_base,
                                                      stoperr=1e-10, numItermax=self.sk_iter_limit, rho=rho_value,
                                                      semi_use=(self.ot_mode != 'balanced'), prior=None)
                    weight = sk.cost_forward(cost_img, final=True)
                    weight = torch.nan_to_num(weight)
                elif (self.use_ot_as_weights == 1) and (pseudo_img is not None):
                    ot_w = torch.nan_to_num(pseudo_img)
                    if self.ot_weight_strategy == 'replace':
                        weight = ot_w
                    else:
                        beta = max(0.0, min(1.0, self.ot_mix_coeff))
                        weight = (1.0 - beta) * weight + beta * ot_w
                # Normalize rows and do safe bmm
                weight = _safe_row_normalize(weight)
                h_path = _safe_bmm(weight.transpose(1, 2), encoded_3d)
            # ===== Prototype-space contrast (optional, batched) =====
            # This branch computes a MONA-aligned contrastive signal over prototype assignments.
            proto_archive = prototype_archive
            proto_k = int(proto_contrast_k or 0)
            proto_w = float(proto_contrast_weight or 0.0)
            if (proto_archive is not None) and (proto_k > 0) and (proto_w > 0.0):
                # Use features in the same space as prototype logits
                feat_proto_space = proj_img_0  # [B, N, D']
                with torch.no_grad():
                    assign_ids = torch.argmax(patch_logit_primary.detach(), dim=-1)  # [B, N]
                # Always expose enqueue tensors; trainer will enqueue after optimizer step
                out['proto_enqueue_features'] = feat_proto_space.detach()
                out['proto_enqueue_ids'] = assign_ids.detach()
                # enqueue after loss to avoid self-match during this forward
                # Compute prototypes from current archive snapshot
                proto_bank, valid_mask = proto_archive.get_prototypes()
                if (proto_bank is not None) and torch.any(valid_mask):
                    proto_bank = F.normalize(proto_bank, dim=-1)
                    # Similarity of each token to prototypes [B,N,K]
                    sim = torch.einsum('bnd,kd->bnk', F.normalize(feat_proto_space, dim=-1), proto_bank)
                    # Gather positive (assigned proto) and hardest negatives via topk
                    pos_idx = assign_ids.unsqueeze(-1)
                    pos_sim = torch.gather(sim, dim=-1, index=pos_idx).squeeze(-1)  # [B,N]
                    # mask out the positive index then pick top-k negatives
                    b, n, k = sim.shape
                    pos_mask = F.one_hot(assign_ids, num_classes=k).bool()
                    sim_neg = sim.masked_fill(pos_mask, -1e6)
                    topk_vals, _ = torch.topk(sim_neg, k=min(proto_k, k-1), dim=-1)
                    # Contrastive hinge: maximize pos vs mean of top-k negatives
                    neg_mean = torch.mean(topk_vals, dim=-1)  # [B,N]
                    proto_contrast = torch.clamp(neg_mean - pos_sim, min=0.0)
                    out['proto_contrast_loss'] = proto_w * torch.mean(proto_contrast)
            if h_path is not None:
                if self.label_use_coattn:
                    # Keep tokens [B, K, D] for attention
                    img_tokens = h_path
                else:
                    # Reduce K->1 into [B, D]
                    h_img = self.image_linear(h_path.transpose(-1, -2)).squeeze(-1)
                    # keep raw h_img
                    features.append(h_img)

        if self.use_omics:
            if isinstance(x_omics, list) and len(x_omics) > 0 and len(self.omics_per_path) == len(x_omics):
                per_path = []
                for i, t in enumerate(x_omics):
                    per_path.append(self.omics_per_path[i](t.float()))
                omic_encoded = torch.stack(per_path, dim=1)
            else:
                omic_encoded = x_omics
            omic_encoded = torch.nan_to_num(omic_encoded)
            if self.shared_prototypes == 1:
                proj_omic_heads = []
                proj_omic_0 = self.shared_proj_omics(omic_encoded)
                proj_omic_0 = F.normalize(proj_omic_0, dim=-1)
                proj_omic_heads.append(proj_omic_0)
                if (hasattr(self, 'shared_proj_omics_extra_heads') and (self.shared_proj_omics_extra_heads is not None)) and (self.num_heads > 1):
                    for lin in self.shared_proj_omics_extra_heads:
                        proj_h = lin(omic_encoded)
                        proj_h = F.normalize(proj_h, dim=-1)
                        proj_omic_heads.append(proj_h)
                proto_norm = F.normalize(self.shared_proto_bank, dim=-1)
                omic_logits_heads = [torch.einsum('bpd,kd->bpk', ph, proto_norm) / max(self.shared_tau, 1e-6) for ph in proj_omic_heads]
            else:
                omic_logits_heads = [cls(omic_encoded) for cls in self.omics_patch_classifiers]
            if forward_pass == 'return_all':
                out['omics_patch_logits_heads'] = omic_logits_heads
                out['omics_encoded_features'] = omic_encoded
            else:
                out['omics_patch_logits_heads'] = omic_logits_heads
                out['omics_encoded_features'] = omic_encoded
            # Select primary head (default 0 or external selection) before pseudo-labels
            omics_sel_head = int(getattr(self, 'omics_selected_head', 0))
            if omics_sel_head < 0 or omics_sel_head >= len(omic_logits_heads):
                omics_sel_head = 0
            omic_logit_primary = omic_logits_heads[omics_sel_head]

            pseudo_omics = None
            if self.omics_pseudo_label == 1:
                if self.omics_ot_pseudo is not None:
                    with torch.no_grad():
                        feat_for_ot_o = (proj_omic_0 if self.shared_prototypes == 1 else omic_encoded)
                        if (self.shared_prototypes == 1) and (self.joint_ot_single_path == 1):
                            img_logit = out.get('__img_logit_joint__', None)
                            img_feat = out.get('__img_feat_joint__', None)
                            if (img_logit is not None) and (img_feat is not None):
                                joint_logit = torch.cat([img_logit, omic_logit_primary.detach()], dim=1)
                                joint_feat = torch.cat([img_feat, feat_for_ot_o.detach()], dim=1)
                                joint_pseudo = self.image_ot_pseudo(joint_logit, joint_feat, iterations=iters, iterations_per_epoch=iters_per_ep)
                                Bc, Nc_img, Kc = img_logit.shape
                                pseudo_img = joint_pseudo[:, :Nc_img, :]
                                pseudo_omics = joint_pseudo[:, Nc_img:, :]
                                out.pop('__img_logit_joint__', None)
                                out.pop('__img_feat_joint__', None)
                            else:
                                pseudo_omics = self.omics_ot_pseudo(omic_logit_primary.detach(), feat_for_ot_o.detach(), iterations=iters, iterations_per_epoch=iters_per_ep)
                        else:
                            pseudo_omics = self.omics_ot_pseudo(omic_logit_primary.detach(), feat_for_ot_o.detach(), iterations=iters, iterations_per_epoch=iters_per_ep)
                else:
                    if self.ot_mode == 'kmeans':
                        pseudo_omics = _batch_kmeans_assign(omic_encoded, (self.shared_proto_num if self.shared_prototypes == 1 else self.omics_proto_num), iters=2)
                    else:
                        # Sharpen fallback pseudo labels for CE only
                        pseudo_omics = torch.softmax(omic_logit_primary.detach() * 2.0, dim=-1)
            # If image aggregation was deferred due to joint OT, finish it now using joint pseudo labels (pseudo or trainable)
            if bool(out.get('__img_defer_agg__', False)):
                encoded_to_agg = out.pop('__img_encoded_to_agg__')
                weight_soft = out.pop('__img_weight_soft__', None)
                logit_img_train = out.pop('__img_logit_train__', None)
                out.pop('__img_defer_agg__', None)
                # combine OT weights per strategy (non-trainable) or compute trainable joint OT-attn
                if self.ot_attn_trainable == 1 and (logit_img_train is not None):
                    # Build joint cost from cached image logits and current omics logits
                    cost_joint = -torch.cat([logit_img_train, omic_logit_primary], dim=1)  # [B, Nimg+Ngene, K]
                    rho_value = self.rho_upper
                    sk = SemiCurrSinkhornKnopp_stable(num_iters=self.sk_iter, epsilon=self.sk_epsilon, gamma=self.gamma_base,
                                                      stoperr=1e-10, numItermax=self.sk_iter_limit, rho=rho_value,
                                                      semi_use=True, prior=None)
                    T_joint = sk.cost_forward(cost_joint, final=True)  # [B, Nimg+Ngene, K]
                    T_joint = torch.nan_to_num(T_joint)
                    # split back
                    n_img = logit_img_train.size(1)
                    weight_img = T_joint[:, :n_img, :]
                    omic_weight = T_joint[:, n_img:, :]
                else:
                    weight_img = weight_soft
                    if (self.use_ot_as_weights == 1) and (pseudo_img is not None):
                        ot_w = torch.nan_to_num(pseudo_img)
                        if self.ot_weight_strategy == 'replace':
                            weight_img = ot_w
                        else:
                            beta = max(0.0, min(1.0, self.ot_mix_coeff))
                            weight_img = (1.0 - beta) * weight_soft + beta * ot_w
                weight_img = _safe_row_normalize(torch.nan_to_num(weight_img))
                h_path_joint = _safe_bmm(weight_img.transpose(1, 2), encoded_to_agg)
                if self.label_use_coattn:
                    img_tokens = h_path_joint
                else:
                    h_img = self.image_linear(h_path_joint.transpose(-1, -2)).squeeze(-1)
                    # keep raw h_img
                    features.append(h_img)
            if self.ot_mode == 'kmeans':
                omic_weight = _batch_kmeans_assign(omic_encoded, (self.shared_proto_num if self.shared_prototypes == 1 else self.omics_proto_num), iters=2)
            else:
                omic_weight = _safe_softmax(omic_logit_primary, dim=2)
            if self.ot_attn_trainable == 1:
                cost_om = -omic_logit_primary
                rho_value = (self.rho_fixed if self.ot_mode == 'ubot_fixed_rho' else self.rho_upper)
                sk = SemiCurrSinkhornKnopp_stable(num_iters=self.sk_iter, epsilon=self.sk_epsilon, gamma=self.gamma_base,
                                                  stoperr=1e-10, numItermax=self.sk_iter_limit, rho=rho_value,
                                                  semi_use=(self.ot_mode != 'balanced'), prior=None)
                omic_weight = sk.cost_forward(cost_om, final=True)
                omic_weight = torch.nan_to_num(omic_weight)
            elif (self.use_ot_as_weights == 1) and (pseudo_omics is not None):
                ot_w_o = torch.nan_to_num(pseudo_omics)
                if self.ot_weight_strategy == 'replace':
                    omic_weight = ot_w_o
                else:
                    beta = max(0.0, min(1.0, self.ot_mix_coeff))
                    omic_weight = (1.0 - beta) * omic_weight + beta * ot_w_o
            # Normalize rows and do safe bmm
            omic_weight = _safe_row_normalize(omic_weight)
            h_omic = _safe_bmm(omic_weight.transpose(1, 2), omic_encoded)
            if self.label_use_coattn:
                omic_tokens = h_omic
            else:
                h_omic_agg = self.omics_linear(h_omic.transpose(-1, -2)).squeeze(-1)
                # keep raw h_omic_agg
                features.append(h_omic_agg)


        # Optional: soft CE losses with pseudo labels (not part of process_surv)
        if (self.wsi_use_ce == 1) and (self.image_pseudo_label == 1) and (pseudo_img is not None):
            ce_wsi = _safe_soft_ce(patch_logit_primary, pseudo_img, dim=-1)
            if torch.isfinite(ce_wsi):
                out['wsi_ce_loss'] = self.wsi_ce_weight * ce_wsi
        if (self.omics_use_ce == 1) and (self.omics_pseudo_label == 1) and (pseudo_omics is not None):
            ce_omics = _safe_soft_ce(omic_logit_primary, pseudo_omics, dim=-1)
            if torch.isfinite(ce_omics):
                out['omics_ce_loss'] = self.omics_ce_weight * ce_omics

        # Optional: per-modality context refinement + anchor alignment prior to co-attn
        if self.label_use_coattn and (self.enable_modality_refine == 1):
            B = None
            if img_tokens is not None:
                B = img_tokens.shape[0]
            elif omic_tokens is not None:
                B = omic_tokens.shape[0]
            if B is not None:
                # anchors expanded to batch
                mh = self.modality_anchor[0].unsqueeze(0).unsqueeze(1)  # [1,1,D]
                mg = self.modality_anchor[1].unsqueeze(0).unsqueeze(1)
                # refine image tokens
                if img_tokens is not None:
                    mh_b = mh.expand(img_tokens.size(0), 1, -1)
                    img_aug = torch.cat([img_tokens, mh_b], dim=1)
                    img_aug = self.img_refiner(img_aug)
                    img_refined = img_aug[:, :img_tokens.size(1), :]
                    # apply or keep original tokens
                    if self.modref_apply_to_tokens == 1:
                        img_tokens = img_refined
                    # mean over refined tokens for loss regardless
                    sh_bar = torch.mean(img_refined, dim=1)
                else:
                    sh_bar = None
                # refine omics tokens
                if omic_tokens is not None:
                    mg_b = mg.expand(omic_tokens.size(0), 1, -1)
                    gen_aug = torch.cat([omic_tokens, mg_b], dim=1)
                    gen_aug = self.omics_refiner(gen_aug)
                    omic_refined = gen_aug[:, :omic_tokens.size(1), :]
                    if self.modref_apply_to_tokens == 1:
                        omic_tokens = omic_refined
                    sg_bar = torch.mean(omic_refined, dim=1)
                else:
                    sg_bar = None
                # compute alignment loss only if we have projections
                modref_loss = 0.0
                if (sh_bar is not None):
                    shp = F.normalize(self.modref_proj(sh_bar), dim=-1)
                    mh_n = F.normalize(self.modality_anchor[0], dim=-1)
                    pos_h = torch.sum(shp * mh_n, dim=-1) / max(self.modref_tau, 1e-6)
                    if (omic_tokens is not None):
                        mg_n = F.normalize(self.modality_anchor[1], dim=-1)
                        neg_h = torch.sum(shp * mg_n, dim=-1) / max(self.modref_tau, 1e-6)
                        L_h = -torch.log(torch.exp(pos_h) / (torch.exp(pos_h) + torch.exp(neg_h) + 1e-6)).mean()
                    else:
                        L_h = (-pos_h).mean()
                    modref_loss = modref_loss + L_h
                if (sg_bar is not None):
                    sgp = F.normalize(self.modref_proj(sg_bar), dim=-1)
                    mg_n = F.normalize(self.modality_anchor[1], dim=-1)
                    pos_g = torch.sum(sgp * mg_n, dim=-1) / max(self.modref_tau, 1e-6)
                    if (img_tokens is not None):
                        mh_n = F.normalize(self.modality_anchor[0], dim=-1)
                        neg_g = torch.sum(sgp * mh_n, dim=-1) / max(self.modref_tau, 1e-6)
                        L_g = -torch.log(torch.exp(pos_g) / (torch.exp(pos_g) + torch.exp(neg_g) + 1e-6)).mean()
                    else:
                        L_g = (-pos_g).mean()
                    modref_loss = modref_loss + L_g
                out['modref_loss'] = modref_loss

        # Co-attn token path
        if self.label_use_coattn:
            # Expect at least one token set
            if (img_tokens is None) and (omic_tokens is None):
                raise ValueError("label_use_coattn is True but no modality tokens available")
            if (img_tokens is not None) and (omic_tokens is not None):
                tokens = torch.cat([omic_tokens, img_tokens], dim=1)
                split_idx = omic_tokens.shape[1]
            else:
                tokens = img_tokens if img_tokens is not None else omic_tokens
                split_idx = 0 if (omic_tokens is None) else omic_tokens.shape[1]

            # Pass through co-attn stack
            mm_embed = self.label_coattn(tokens) if isinstance(self.label_coattn, nn.Sequential) else tokens

            # Pool back to per-branch embeddings
            if (img_tokens is not None) and (omic_tokens is not None):
                gene_post = mm_embed[:, :split_idx, :]
                img_post = mm_embed[:, split_idx:, :]
                h_gene = torch.mean(gene_post, dim=1)
                h_img = torch.mean(img_post, dim=1)
                # fused feature for downstream archive
                fused_feature = torch.cat([h_img, h_gene], dim=1)
                if self.coattn_head is not None:
                    logits = self.coattn_head(fused_feature)
                elif self.fusion_head is not None:
                    logits = self.fusion_head(h_img, h_gene)
                else:
                    # Fallback to concat + linear if neither head exists
                    logits = nn.functional.linear(fused_feature, weight=torch.zeros(self.path_proj_dim * 2, self.num_classes, device=h_img.device).t())
            else:
                pooled = torch.mean(mm_embed, dim=1)
                logits = self.classifier(pooled)
                fused_feature = pooled
            out['logits'] = logits
            out['fused_feature'] = fused_feature

            # Structural KNN loss is computed in the trainer for better decoupling
            return out
        else:
            # Original late-fusion path
            if len(features) == 0:
                raise ValueError("OTMIL_Label requires at least one modality enabled")
            if self.use_image and self.use_omics:
                h_img, h_omic = features
                logits = self.fusion_head(h_img, h_omic)
                fused_feature = torch.cat([h_img, h_omic], dim=1)
            else:
                embedding = features[0]
                embedding = torch.nan_to_num(embedding)
                logits = self.image_head(embedding) if self.use_image else self.omics_head(embedding)
                fused_feature = embedding
            out['logits'] = logits
            out['fused_feature'] = fused_feature

            # Structural KNN loss is computed in the trainer for better decoupling
            return out

    def forward(self, x_path, x_omics, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None, iterations=None, iterations_per_epoch=None, forward_pass=None, feature_archive=None, archive_k: int = 5, prototype_archive=None, proto_contrast_k: int = 0, proto_contrast_weight: float = 0.0):
        out = self.forward_no_loss(
            x_path,
            x_omics,
            return_attn=return_attn,
            attn_mask=attn_mask,
            iterations=iterations,
            iterations_per_epoch=iterations_per_epoch,
            forward_pass=forward_pass,
            feature_archive=feature_archive,
            archive_k=archive_k,
            prototype_archive=prototype_archive,
            proto_contrast_k=proto_contrast_k,
            proto_contrast_weight=proto_contrast_weight,
        )
        # If only requesting return_all for external OT, skip survival loss assembly
        if forward_pass == 'return_all':
            return out, {}
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        # Prototype orthogonality loss for shared prototypes when OT-attn is trainable
        if (self.shared_prototypes == 1) and (self.ot_attn_trainable == 1) and (self.proto_ortho_weight > 0.0):
            P = F.normalize(self.shared_proto_bank, dim=-1)
            gram = torch.matmul(P, P.t())
            I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
            ortho = torch.mean((gram - I) ** 2)
            results_dict['proto_ortho_loss'] = ortho
            results_dict['loss'] = results_dict['loss'] + self.proto_ortho_weight * ortho
        # Merge optional modality-refine loss
        if ('modref_loss' in out) and (self.modref_weight > 0.0):
            results_dict['modref_loss'] = out['modref_loss']
            results_dict['loss'] = results_dict['loss'] + self.modref_weight * out['modref_loss']
        # Do not add per-branch CE or cross-modal consistency into total loss; SK loss is merged in trainer
        results_dict.update(out)
        return results_dict, log_dict


