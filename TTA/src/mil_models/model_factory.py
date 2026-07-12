import os
from mil_models.model_otmil_label import OTMIL_Label

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def create_multimodal_survival_model(args, omic_sizes=[]):
    if args.loss_fn in ['cox', 'rank']:
        num_classes = 1
    else:
        num_classes = args.n_label_bins

    # Always build OTMIL_Label (covers OT-attn and OT-as-weights aggregation)
    model = OTMIL_Label(
        histo_in_dim=getattr(args, 'feat_dim', 1024),
        hidden_dim=256,
        dropout=getattr(args, 'dropout', 0.1),
        num_classes=num_classes,
        ot_mode=getattr(args, 'ot_mode', 'ubot'),
        rho_fixed=getattr(args, 'rho_fixed', 0.1),
        image_proto_num=getattr(args, 'image_proto_num', 16),
        num_heads=getattr(args, 'num_heads', 1),
        omics_proto_num=getattr(args, 'omics_proto_num', 16),
        use_image=getattr(args, 'modality_type', 'gene') in ['histo','multi'],
        use_omics=getattr(args, 'modality_type', 'gene') in ['gene','multi'],
        image_pseudo_label=getattr(args, 'image_pseudo_label', 0),
        omics_pseudo_label=getattr(args, 'omics_pseudo_label', 0),
        image_ot_impl=getattr(args, 'image_ot_impl', 'batchot'),
        omics_ot_impl=getattr(args, 'omics_ot_impl', 'batchot'),
        omic_sizes=omic_sizes,
        fusion_type=getattr(args, 'fusion_type', 'coattn'),
        label_num_coattn_layers=getattr(args, 'label_num_coattn_layers', 1),
        # modality-context refinement
        enable_modality_refine=getattr(args, 'enable_modality_refine', 0),
        modref_weight=getattr(args, 'modref_weight', 0.0),
        modref_tau=getattr(args, 'modref_tau', 0.1),
        modref_layers=getattr(args, 'modref_layers', 1),
        modref_apply_to_tokens=getattr(args, 'modref_apply_to_tokens', 1),
        # OT as weights
        use_ot_as_weights=getattr(args, 'use_ot_as_weights', 0),
        ot_weight_strategy=getattr(args, 'ot_weight_strategy', 'mix'),
        ot_mix_coeff=getattr(args, 'ot_mix_coeff', 0.3),
        # Differentiable OT-attn
        ot_attn_trainable=getattr(args, 'ot_attn_trainable', 0),
        proto_ortho_weight=getattr(args, 'proto_ortho_weight', 0.0),
        rho_strategy=getattr(args, 'rho_strategy', 'sigmoid'),
        rho_base=getattr(args, 'rho_base', 0.1),
        rho_upper=getattr(args, 'rho_upper', 1.0),
        gamma_schedule=getattr(args, 'gamma_schedule', None),
        gamma_base=getattr(args, 'gamma_base', 1.0),
        gamma_upper=getattr(args, 'gamma_upper', 1.0),
        sk_epsilon=getattr(args, 'sk_epsilon', 0.1),
        sk_iter=getattr(args, 'sk_iter', 3),
        sk_iter_limit=getattr(args, 'sk_iter_limit', 1000),
        mm_factor=getattr(args, 'mm_factor', 0.5),
        mm_iter_limit=getattr(args, 'mm_iter_limit', 100),
        ema=getattr(args, 'ema', 1.0),
        # CE flags
        wsi_use_ce=getattr(args, 'wsi_use_ce', 0),
        wsi_ce_weight=getattr(args, 'wsi_ce_weight', 0.0),
        omics_use_ce=getattr(args, 'omics_use_ce', 0),
        omics_ce_weight=getattr(args, 'omics_ce_weight', 0.0),
        # shared prototype switches
        shared_prototypes=getattr(args, 'shared_prototypes', 0),
        shared_proto_num=getattr(args, 'shared_proto_num', 16),
        shared_proto_dim=getattr(args, 'shared_proto_dim', 256),
        shared_tau=getattr(args, 'shared_tau', 1.0),
        # joint OT switch
        joint_ot_single_path=getattr(args, 'joint_ot_single_path', 0),
    )
    return model

