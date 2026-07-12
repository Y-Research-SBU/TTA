import os
from os.path import join as j_
import pdb
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from sksurv.metrics import concordance_index_censored


from mil_models import create_multimodal_survival_model
from mil_models.ot_component.sk_multi import SKMultiLoss
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

# Optional wandb usage (no hard fail if missing)
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    # Resolve device lazily; avoid CUDA checks at import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optional anomaly detection for debugging
    try:
        if int(getattr(args, 'debug_nans', 0)) == 1:
            torch.autograd.set_detect_anomaly(True)
    except Exception:
        pass

    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric in ['loss', 'c_index']
    
    # support warmup: use NLL for first K epochs then switch to Cox
    if args.loss_fn == 'nll':
        base_loss = 'nll'
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        base_loss = 'cox'
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        base_loss = 'rank'
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    print('\nInit Model...', end=' ')
    modality_val = args.modality_type
    print(f"Model modality: {modality_val}; histo_OT={getattr(args, 'use_image_ot', False)}, omics_OT={getattr(args, 'use_omics_ot', False)}")

    ## Set the dimensionality for different inputs (only when omics is used)
    if modality_val != 'histo':
        args.omic_dim = datasets['train'].dataset.omics_data.shape[1]

    if modality_val != 'histo' and args.omics_modality in ['pathway', 'functional']:
        omic_sizes = datasets['train'].dataset.omic_sizes
    else:
        omic_sizes = []

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)
    model.to(device)
    
    print_network(model)

    # Register gradient NaN/Inf hooks when debugging
    try:
        if int(getattr(args, 'debug_nans', 0)) == 1:
            name_map = {id(p): n for n, p in model.named_parameters()}
            def _make_grad_hook(param_id):
                def _hook(grad):
                    try:
                        if grad is None:
                            return
                        if not torch.isfinite(grad).all():
                            n = name_map.get(param_id, f"param_id={param_id}")
                            gt = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).float()
                            print(f"[grad-nan] {n} shape={tuple(grad.shape)} "
                                  f"min={float(torch.min(gt)):.4e} max={float(torch.max(gt)):.4e} "
                                  f"mean={float(torch.mean(gt)):.4e} std={float(torch.std(gt)):.4e} "
                                  f"has_nan={bool(torch.isnan(grad).any())} has_inf={bool(torch.isinf(grad).any())}")
                    except Exception:
                        pass
                return _hook
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(_make_grad_hook(id(p)))
    except Exception:
        pass



    # ===== Optional: build parallel OT loss (multi-head) =====
    # Minimal switches with sane defaults
    args.num_heads = getattr(args, 'num_heads', 1)
    args.wsi_use_sk_multi = getattr(args, 'wsi_use_sk_multi', getattr(args, 'use_sk_multi', 0))
    args.wsi_sk_weight = getattr(args, 'wsi_sk_weight', getattr(args, 'sk_weight', 0.0))
    # omics SK toggles
    args.omics_use_sk_multi = getattr(args, 'omics_use_sk_multi', 0)
    args.omics_sk_weight = getattr(args, 'omics_sk_weight', 0.0)
    args.sk_type = getattr(args, 'sk_type', 'sppot')
    args.ot_frame = getattr(args, 'ot_frame', 'mm')
    args.sk_iter_limit = getattr(args, 'sk_iter_limit', 1000)
    args.sk_epsilon = getattr(args, 'sk_epsilon', 0.1)
    args.rho_base = getattr(args, 'rho_base', 0.1)
    args.rho_upper = getattr(args, 'rho_upper', 1.0)
    args.rho_strategy = getattr(args, 'rho_strategy', 'sigmoid')
    args.gamma_base = getattr(args, 'gamma_base', 1.0)
    args.gamma_upper = getattr(args, 'gamma_upper', 1.0)
    args.gamma_schedule = getattr(args, 'gamma_schedule', None)
    args.mm_factor = getattr(args, 'mm_factor', 0.5)
    args.mm_iter_limit = getattr(args, 'mm_iter_limit', 100)
    args.ema_mm = getattr(args, 'ema_mm', 1.0)
    # Speed knobs for SKMultiLoss (no CLI required):
    # - sk_every: compute multi-view consistency every N batches (default 3)
    # - max_sk_views: cap number of views to use per batch (default 2)
    # - sk_view_ratio: per-view random subsample ratio for WSI tokens (unused)
    # - sk_min_tokens: minimal tokens kept per view to avoid degenerate bags (default 256)
    args.sk_every = getattr(args, 'sk_every', 3)
    args.max_sk_views = getattr(args, 'max_sk_views', 2)
    # True multi-view knobs per WSI sample
    args.sk_view_keep_ratio = getattr(args, 'sk_view_keep_ratio', 0.8)
    args.sk_num_samples = getattr(args, 'sk_num_samples', 4)
    args.sk_view_ratio = getattr(args, 'sk_view_ratio', 0.5)
    args.sk_min_tokens = getattr(args, 'sk_min_tokens', 256)

    sk_loss = None
    logits_banks = None
    feature_bank = None
    if (args.wsi_use_sk_multi and (args.num_heads > 1) and (args.wsi_sk_weight > 0)):
            sk_loss = SKMultiLoss(
            num_heads=args.num_heads,
            sk_type=args.sk_type,
            ot_frame=args.ot_frame,
            sk_iter_limit=args.sk_iter_limit,
            epsilon=args.sk_epsilon,
            rho_base=args.rho_base,
            rho_upper=args.rho_upper,
            rho_strategy=args.rho_strategy,
            gamma_base=args.gamma_base,
            gamma_upper=args.gamma_upper,
            gamma_schedule=args.gamma_schedule,
            mm_factor=args.mm_factor,
            mm_iter_limit=args.mm_iter_limit,
            ema_mm=args.ema_mm,
            logits_bank=None,
            feature_bank=None,
            total_iter=args.max_epochs * len(datasets['train']) if hasattr(args, 'max_epochs') else 100000,
            start_iter=0,
        ).to(device)
        # Defer memory bank creation to bank_start_epoch to avoid cold-start instability

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None
    
    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        # warmup: if configured and base_loss is cox, use NLL for first K epochs
        effective_loss_fn = loss_fn
        if base_loss == 'cox' and getattr(args, 'loss_warmup_epochs', 0) > 0 and epoch < args.loss_warmup_epochs:
            effective_loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        # set model debug print cadence 
        if hasattr(model, 'debug_every'):
            model.debug_every = 100

        # ===== Modality-refine scheduling (freeze/apply/decay) =====
        # 1) Freeze/unfreeze modality_anchor for first K epochs (only if present)
        freeze_k = int(getattr(args, 'modref_freeze_epochs', 0))
        if hasattr(model, 'modality_anchor') and isinstance(model.modality_anchor, torch.nn.Parameter):
            model.modality_anchor.requires_grad = (epoch >= max(0, freeze_k))
        # 2) Optionally delay applying refined tokens
        switch_ep = int(getattr(args, 'modref_apply_switch_epoch', -1))
        if hasattr(model, 'modref_apply_to_tokens') and (switch_ep >= 0):
            model.modref_apply_to_tokens = 1 if (epoch >= switch_ep) else 0
        # 3) Linearly decay modref_weight to 0 after a start epoch over N epochs
        decay_start = int(getattr(args, 'modref_decay_start', -1))
        decay_epochs = int(getattr(args, 'modref_decay_epochs', 0))
        if hasattr(model, 'modref_weight'):
            # cache base once
            if not hasattr(model, '_modref_weight_base'):
                model._modref_weight_base = float(getattr(model, 'modref_weight', 0.0))
            if (decay_start >= 0) and (decay_epochs > 0) and (epoch >= decay_start):
                t = float(epoch - decay_start) / float(decay_epochs)
                if t < 0.0:
                    t = 0.0
                if t > 1.0:
                    t = 1.0
                model.modref_weight = float(model._modref_weight_base) * (1.0 - t)
            else:
                # ensure weight resets outside decay window
                model.modref_weight = float(model._modref_weight_base)

        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, effective_loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps, epoch=epoch,
                                            grad_clip_norm=getattr(args, 'grad_clip_norm', -1),
                                            sk_loss=sk_loss, sk_weight=args.wsi_sk_weight,
                                            use_sk_multi=args.wsi_use_sk_multi,
                                            sk_every=args.sk_every, max_sk_views=args.max_sk_views,
                                            sk_view_keep_ratio=args.sk_view_keep_ratio, sk_num_samples=args.sk_num_samples,
                                            omics_use_sk_multi=args.omics_use_sk_multi, omics_sk_weight=args.omics_sk_weight,
                                            debug_nans=int(getattr(args, 'debug_nans', 0)))


        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            # disable debug prints in validation
            if hasattr(model, 'debug_every'):
                model.debug_every = -1
            val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)

            # Epoch-level logging to wandb (validation metrics)
            if _WANDB_AVAILABLE and (wandb.run is not None):
                wandb.log({f"val/{k}": float(v) for k, v in val_results.items()}, step=(epoch + 1) * len(datasets['train']))
                

            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']
                elif args.es_metric == 'c_index':
                    score = val_results['c_index']
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')

                # Only allow saving the "best" checkpoint after es_min_epochs.
                # EarlyStopping still evaluates the metric each epoch, but actual disk save is deferred
                # until the warm-up period has passed. This prevents early-epoch noisy minima from being kept.
                def _maybe_save_checkpoint(**kwargs):
                    if epoch >= args.es_min_epochs:
                        return save_checkpoint(**kwargs)

                stop = early_stopper(epoch, score, _maybe_save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        # After epoch: if multi-head enabled, select lowest-loss head and set to model 
        if args.wsi_use_sk_multi and (args.num_heads > 1):
            wsi_head_losses = train_results.get('wsi_sk_head_losses', None)
            if isinstance(wsi_head_losses, list) and len(wsi_head_losses) > 0:
                sel = int(np.argmin(np.array(wsi_head_losses)))
                setattr(model, 'selected_head', sel)
                print(f"[head-select][wsi] Selected head={sel} via SK loss: {wsi_head_losses}")
                    
        if getattr(args, 'omics_use_sk_multi', 0) and (args.num_heads > 1):
            omics_head_losses = train_results.get('omics_sk_head_losses', None)
            if isinstance(omics_head_losses, list) and len(omics_head_losses) > 0:
                sel_o = int(np.argmin(np.array(omics_head_losses)))
                setattr(model, 'omics_selected_head', sel_o)
                print(f"[head-select][omics] Selected head={sel_o} via SK loss: {omics_head_losses}")
                    

        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of epoch: Save/Eval logic for best vs last epoch
    ckpt_best = j_(args.results_dir, f"s_checkpoint.pth")
    ckpt_last = j_(args.results_dir, f"s_checkpoint_last.pth")

    es_and_val = bool(args.early_stopping) and ('val' in datasets.keys()) and os.path.isfile(ckpt_best)

    if es_and_val:
        # Save the final (last-epoch) model separately and evaluate both last and best
        try:
            save_checkpoint(config=vars(args), epoch=epoch, model=model, score=float('nan'), save_dir=args.results_dir, fname='s_checkpoint_last.pth')
        except Exception:
            # Fallback: save raw state_dict if structured save fails
            torch.save({'model': model.state_dict()}, ckpt_last)

        # 1) Evaluate LAST epoch model (current in-memory)
        results_last, dumps_last = {}, {}
        for split_name, loader in datasets.items():
            if split_name == 'val':
                continue
            print(f'End of training. Evaluating LAST on Split {split_name.upper()}...:')
            return_attn = True
            split_res, split_dump = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                      dump_results=True, return_attn=return_attn, verbose=False)
            results_last[split_name] = split_res
            dumps_last[split_name] = split_dump
            # Log LAST metrics per split as scalars for visibility (in addition to summary)
            if _WANDB_AVAILABLE and (wandb.run is not None):
                wandb.log({f"final/{split_name}/{k}_last": float(v) for k, v in split_res.items()})
                
        if 'train' in results_last:
            _ = results_last.pop('train')

        # 2) Load BEST and evaluate
        model.load_state_dict(torch.load(ckpt_best)['model'])
        results_best, dumps_best = {}, {}
        for split_name, loader in datasets.items():
            if split_name == 'val':
                continue
            print(f'End of training. Evaluating BEST on Split {split_name.upper()}...:')
            return_attn = True
            split_res, split_dump = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                      dump_results=True, return_attn=return_attn, verbose=False)
            results_best[split_name] = split_res
            dumps_best[split_name] = split_dump
            # Log BEST metrics per split as scalars
            if _WANDB_AVAILABLE and (wandb.run is not None):
                wandb.log({f"final/{split_name}/{k}_best": float(v) for k, v in split_res.items()})
                
        if 'train' in results_best:
            _ = results_best.pop('train')

        # 3) Merge metrics with suffixes _best and _last for each split
        merged_results, merged_dumps = {}, {}
        for split_name in set(list(results_best.keys()) + list(results_last.keys())):
            merged_results[split_name] = {}
            if split_name in results_best:
                for m, v in results_best[split_name].items():
                    merged_results[split_name][f"{m}_best"] = v
            if split_name in results_last:
                for m, v in results_last[split_name].items():
                    merged_results[split_name][f"{m}_last"] = v

            # organize dumps for clarity
            md = {}
            if split_name in dumps_best:
                md['best'] = dumps_best[split_name]
            if split_name in dumps_last:
                md['last'] = dumps_last[split_name]
            merged_dumps[split_name] = md

        # 4) Log final summary metrics (best/last) to wandb.summary
        if _WANDB_AVAILABLE and (wandb.run is not None):
            for split_name, metrics in merged_results.items():
                for mk, mv in metrics.items():
                    try:
                        wandb.summary[f"final/{split_name}/{mk}"] = float(mv)
                    except Exception:
                        continue
            

        return merged_results, merged_dumps
    else:
        # No early-stopping+val: keep previous behavior
        if not os.path.isfile(ckpt_best):
            # Save current model if early stopping wasn't used or no val split / no checkpoint was created
            torch.save(model.state_dict(), ckpt_best)

        results, dumps = {}, {}
        for k, loader in datasets.items():
            if k == 'val':
                continue
            print(f'End of training. Evaluating on Split {k.upper()}...:')
            return_attn = True
            results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                         dump_results=True, return_attn=return_attn, verbose=False)

            if k == 'train':
                _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        # Log last-only final summary when no best model available
        if _WANDB_AVAILABLE and (wandb.run is not None):
            for split_name, metrics in results.items():
                for mk, mv in metrics.items():
                    try:
                        wandb.summary[f"final/{split_name}/{mk}_last"] = float(mv)
                    except Exception:
                        continue
            
        return results, dumps

## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32, epoch=0, grad_clip_norm=-1,
                        sk_loss=None, sk_weight: float = 0.0, use_sk_multi: int = 0,
                        sk_every: int = 3, max_sk_views: int = 2,
                        sk_view_keep_ratio: float = 0.8, sk_num_samples: int = 4,
                        omics_use_sk_multi: int = 0, omics_sk_weight: float = 0.0,
                        debug_nans: int = 0):
    
    model.train()
    device = next(model.parameters()).device  # Get device from model

    # Helper for quick tensor stats in debug mode
    def _tstats(name, t):
        try:
            if not torch.is_tensor(t):
                print(f"{name}: <non-tensor>")
                return
            tt = torch.nan_to_num(t.detach(), nan=0.0, posinf=0.0, neginf=0.0).float()
            print((f"{name}: shape={tuple(t.shape)} "
                   f"min={float(torch.min(tt)):.4e} "
                   f"max={float(torch.max(tt)):.4e} "
                   f"mean={float(torch.mean(tt)):.4e} "
                   f"std={float(torch.std(tt)):.4e} "
                   f"has_nan={bool(torch.isnan(t).any())} "
                   f"has_inf={bool(torch.isinf(t).any())}"))
        except Exception as _e:
            print(f"{name}: <stats_err {str(_e)}>")

    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    total_batches = len(loader)
    # Track per-head SK losses across batches to enable SPPO T-style lowest-loss head selection
    # Separate per-head SK loss trackers for WSI and OMICS
    wsi_head_sum = None
    wsi_head_count = 0
    omics_head_sum = None
    omics_head_count = 0
    
    # End of epoch: ensure any remaining accumulated gradients are applied
    # This handles the case where the last batch doesn't align with accum_steps
    # Note: We track if we have pending gradients via a flag
    has_pending_grads = False
    for batch_idx, batch in enumerate(loader):
        data = safe_list_to(batch['img'], device)
        label = safe_list_to(batch['label'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'] if ('attn_mask' in batch) else None
        if isinstance(attn_mask, list):
            attn_mask = [m.to(device) for m in attn_mask]
        elif attn_mask is not None:
            attn_mask = attn_mask.to(device)

        omics = safe_list_to(batch.get('omics', []), device)


        # Use global iteration for OT rho ramp-up
        global_iter = epoch * total_batches + batch_idx
        # Default survival forward
        out, log_dict = model(
            data,
            omics,
            attn_mask=attn_mask,
            label=label,
            censorship=censorship,
            loss_fn=loss_fn,
            iterations=global_iter,
            iterations_per_epoch=total_batches,
        )

        
        
        

        # Optional: multi-head parallel OT loss (WSI)
        if use_sk_multi and (sk_loss is not None) and (sk_weight > 0):
            # compute only every sk_every batches to reduce overhead
            _sk_every = int(sk_every) if sk_every is not None else 1
            if _sk_every < 1:
                _sk_every = 1
            if (batch_idx % _sk_every) == 0:
                # === Two views via token indices; reuse single forward ===
                def _two_view_indices(sample_tensor: torch.Tensor, keep_ratio: float = 0.8):
                    if not torch.is_tensor(sample_tensor) or sample_tensor.dim() < 2:
                        return None, None
                    num_patches = int(sample_tensor.size(0))
                    if num_patches <= 1:
                        return None, None
                    keep = max(1, int(num_patches * float(keep_ratio)))
                    perm = torch.randperm(num_patches, device=sample_tensor.device)
                    idx1 = perm[:keep]
                    replace_n = max(1, int(0.2 * keep))
                    remain = perm[keep:]
                    if remain.numel() < replace_n:
                        idx2 = torch.randperm(num_patches, device=sample_tensor.device)[:keep]
                    else:
                        pos = torch.randperm(keep, device=sample_tensor.device)[:replace_n]
                        idx2 = idx1.clone(); idx2[pos] = remain[:replace_n]
                    return idx1, idx2

                # choose subset of WSI samples in this batch
                if isinstance(data, list):
                    total_samples = len(data)
                    select_n = int(max(1, min(total_samples, sk_num_samples)))
                    sel_idx = torch.randperm(total_samples)[:select_n].tolist()
                else:
                    sel_idx = [0]

                # Accumulate per-sample slices, then run one SK over a mini-batch (align SPPOT)
                perhead_view1_slices = None
                perhead_view2_slices = None
                v1_feats = []
                v2_feats = []
                for si in sel_idx:
                    img_sample = data[si] if isinstance(data, list) else data
                    if isinstance(omics, list) and len(omics) > 0 and isinstance(data, list):
                        omics_sample = [t[si:si+1] for t in omics]
                    else:
                        omics_sample = omics

                    base_out, _ = model(img_sample, omics_sample, attn_mask=None, iterations=global_iter, iterations_per_epoch=total_batches, forward_pass='return_all')
                    heads_full = base_out.get('patch_logits_heads', []) or []
                    enc_full = base_out.get('encoded_features', None)  # [B,N,D], here B=1
                    idx1, idx2 = _two_view_indices(img_sample, keep_ratio=float(sk_view_keep_ratio))
                    if (idx1 is None) or (len(heads_full) == 0) or (enc_full is None):
                        continue
                    view1_heads = [h.index_select(1, idx1) for h in heads_full]
                    view2_heads = [h.index_select(1, idx2) for h in heads_full]
                    v1_feat = torch.nan_to_num(enc_full.index_select(1, idx1).mean(dim=1))  # [1,D]
                    v2_feat = torch.nan_to_num(enc_full.index_select(1, idx2).mean(dim=1))  # [1,D]
                    if perhead_view1_slices is None:
                        perhead_view1_slices = [[] for _ in range(len(view1_heads))]
                        perhead_view2_slices = [[] for _ in range(len(view2_heads))]
                    for h_idx in range(len(view1_heads)):
                        perhead_view1_slices[h_idx].append(view1_heads[h_idx])
                        perhead_view2_slices[h_idx].append(view2_heads[h_idx])
                    v1_feats.append(v1_feat)
                    v2_feats.append(v2_feat)

                sk_multi_accum = None
                if (perhead_view1_slices is not None) and (len(perhead_view1_slices) > 0):
                    # cat along batch to form [B, N_keep, K]
                    view1_heads_b = [torch.cat(slices, dim=0) for slices in perhead_view1_slices]
                    view2_heads_b = [torch.cat(slices, dim=0) for slices in perhead_view2_slices]
                    v1_feat_b = torch.cat(v1_feats, dim=0) if len(v1_feats) > 0 else None
                    v2_feat_b = torch.cat(v2_feats, dim=0) if len(v2_feats) > 0 else None
                    logits_by_view = [view1_heads_b, view2_heads_b]
                    features_by_view = [v1_feat_b, v2_feat_b]
                    loss_heads = sk_loss(logits_by_view, features_by_view, similarity_matrix=None, data_idxs=None)
                    loss_heads = torch.nan_to_num(loss_heads, nan=0.0, posinf=0.0, neginf=0.0)
                    # accumulate per-head losses for WSI head selection (one count per SK batch)
                    lh_cpu = loss_heads.detach().cpu()
                    if wsi_head_sum is None:
                        wsi_head_sum = lh_cpu.clone()
                    else:
                        if wsi_head_sum.numel() != lh_cpu.numel():
                            wsi_head_sum = lh_cpu.clone()
                        else:
                            wsi_head_sum += lh_cpu
                    wsi_head_count += 1
                    
                    # weight with configured sk_weight (optionally could gate by rho)
                    per_batch_raw = loss_heads.mean()
                    per_batch_val = sk_weight * per_batch_raw
                    sk_multi_accum = per_batch_val

                if sk_multi_accum is not None:
                    if torch.isfinite(sk_multi_accum):
                        if 'wsi_instance_loss' in out and torch.is_tensor(out['wsi_instance_loss']):
                            out['wsi_instance_loss'] = out['wsi_instance_loss'] + sk_multi_accum
                        else:
                            out['wsi_instance_loss'] = sk_multi_accum
                        if torch.is_tensor(out.get('loss', None)):
                            out['loss'] = out['loss'] + sk_multi_accum
                        else:
                            out['loss'] = sk_multi_accum

        

        # Merge optional CE losses (computed in model)
        # Ensure loss is initialized before adding components
        if 'loss' not in out or out['loss'] is None:
            # Initialize loss to zero if not present
            device = next(model.parameters()).device
            out['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        if torch.is_tensor(out.get('wsi_ce_loss', None)):
            ce_wsi = out['wsi_ce_loss']
            if torch.isfinite(ce_wsi):
                out['loss'] = out['loss'] + ce_wsi
        if torch.is_tensor(out.get('omics_ce_loss', None)):
            ce_o = out['omics_ce_loss']
            if torch.isfinite(ce_o):
                out['loss'] = out['loss'] + ce_o

        # Optional: multi-head parallel OT loss (OMICS)
        if omics_use_sk_multi and (sk_loss is not None) and (omics_sk_weight > 0):
            # Two-view indices over pathways; reuse single forward 
            def _omics_two_view_indices(sample_tensor: torch.Tensor, keep_ratio: float = 0.8, min_keep: int = 4):
                if not torch.is_tensor(sample_tensor) or sample_tensor.dim() < 2:
                    return None, None
                # sample_tensor: [B, P, D] or [P, D]
                if sample_tensor.dim() == 3:
                    P = int(sample_tensor.shape[1])
                else:
                    P = int(sample_tensor.shape[0])
                if P <= 1:
                    return None, None
                keep = max(min_keep, int(P * float(keep_ratio)))
                perm = torch.randperm(P, device=sample_tensor.device)
                idx1 = perm[:keep]
                replace_n = max(1, int(0.2 * keep))
                remain = perm[keep:]
                if remain.numel() < replace_n:
                    idx2 = torch.randperm(P, device=sample_tensor.device)[:keep]
                else:
                    pos = torch.randperm(keep, device=sample_tensor.device)[:replace_n]
                    idx2 = idx1.clone(); idx2[pos] = remain[:replace_n]
                return idx1, idx2

            # select subset of samples
            if isinstance(omics, list) and len(omics) > 0:
                total_samples = len(omics[0]) if torch.is_tensor(omics[0]) else len(omics)
                sel_idx = torch.randperm(total_samples)[:int(max(1, min(total_samples, sk_num_samples)))].tolist()
            else:
                sel_idx = [0]

            # Accumulate per-sample slices, then run one SK over a mini-batch (align SPPOT)
            perhead_view1_slices_o = None
            perhead_view2_slices_o = None
            v1_feats_o = []
            v2_feats_o = []
            for si in sel_idx:
                if isinstance(omics, list) and len(omics) > 0 and isinstance(data, list):
                    v_out, _ = model(data[si], [t[si:si+1] for t in omics], attn_mask=None, iterations=global_iter, iterations_per_epoch=total_batches, forward_pass='return_all')
                else:
                    v_out, _ = model(data if not isinstance(data, list) else data[si], omics if not isinstance(omics, list) else [t[si:si+1] for t in omics], attn_mask=None, iterations=global_iter, iterations_per_epoch=total_batches, forward_pass='return_all')
                omics_heads = v_out.get('omics_patch_logits_heads', []) or []
                omics_feat = v_out.get('omics_encoded_features', None)  # [B,P,D], here B=1
                if (len(omics_heads) == 0) or (omics_feat is None) or (not torch.is_tensor(omics_feat)):
                    continue
                idx1, idx2 = _omics_two_view_indices(omics_feat, keep_ratio=float(sk_view_keep_ratio), min_keep=4)
                if (idx1 is None):
                    continue
                view1_heads = [h.index_select(1, idx1) for h in omics_heads]
                view2_heads = [h.index_select(1, idx2) for h in omics_heads]
                v1_feat = torch.nan_to_num(omics_feat.index_select(1, idx1).mean(dim=1))  # [1,D]
                v2_feat = torch.nan_to_num(omics_feat.index_select(1, idx2).mean(dim=1))  # [1,D]
                if perhead_view1_slices_o is None:
                    perhead_view1_slices_o = [[] for _ in range(len(view1_heads))]
                    perhead_view2_slices_o = [[] for _ in range(len(view2_heads))]
                for h_idx in range(len(view1_heads)):
                    perhead_view1_slices_o[h_idx].append(view1_heads[h_idx])
                    perhead_view2_slices_o[h_idx].append(view2_heads[h_idx])
                v1_feats_o.append(v1_feat)
                v2_feats_o.append(v2_feat)

            if (perhead_view1_slices_o is not None) and (len(perhead_view1_slices_o) > 0):
                view1_heads_b_o = [torch.cat(slices, dim=0) for slices in perhead_view1_slices_o]
                view2_heads_b_o = [torch.cat(slices, dim=0) for slices in perhead_view2_slices_o]
                v1_feat_b_o = torch.cat(v1_feats_o, dim=0) if len(v1_feats_o) > 0 else None
                v2_feat_b_o = torch.cat(v2_feats_o, dim=0) if len(v2_feats_o) > 0 else None
                logits_by_view_o = [view1_heads_b_o, view2_heads_b_o]
                features_by_view_o = [v1_feat_b_o, v2_feat_b_o]
                omics_loss_heads = sk_loss(logits_by_view_o, features_by_view_o, similarity_matrix=None, data_idxs=None)
                omics_loss_heads = torch.nan_to_num(omics_loss_heads, nan=0.0, posinf=0.0, neginf=0.0)
                olh_cpu = omics_loss_heads.detach().cpu()
                if omics_head_sum is None:
                    omics_head_sum = olh_cpu.clone()
                else:
                    if omics_head_sum.numel() != olh_cpu.numel():
                        omics_head_sum = olh_cpu.clone()
                    else:
                        omics_head_sum += olh_cpu
                omics_head_count += 1

                omics_raw = omics_loss_heads.mean()
                omics_val = omics_sk_weight * omics_raw
                if torch.isfinite(omics_val):
                    out['omics_instance_loss'] = omics_val if ('omics_instance_loss' not in out) else (out['omics_instance_loss'] + omics_val)
                    if torch.is_tensor(out.get('loss', None)):
                        out['loss'] = out['loss'] + omics_val
                    else:
                        out['loss'] = omics_val

            # memory bank removed



        # Merge structural loss (archive) into total loss if enabled and after optional warmup
        


        # Log total loss and auxiliary instance/CE losses into meters for periodic prints
        # determine batch size for averaging
        if torch.is_tensor(data):
            n_samples = len(data)
        elif isinstance(data, list):
            n_samples = len(data)
        else:
            n_samples = 0

        # total loss
        total_loss_val = out.get('loss', None)
        if total_loss_val is not None:
            total_loss_scalar = total_loss_val.item() if isinstance(total_loss_val, torch.Tensor) else total_loss_val
            if 'loss' not in meters:
                meters['loss'] = AverageMeter()
            meters['loss'].update(total_loss_scalar, n=n_samples)

       

        # wsi instance loss
        if 'wsi_instance_loss' in out:
            if 'wsi_instance_loss' not in meters:
                meters['wsi_instance_loss'] = AverageMeter()
            meters['wsi_instance_loss'].update(out['wsi_instance_loss'].item(), n=n_samples)

        # omics instance loss
        if 'omics_instance_loss' in out:
            if 'omics_instance_loss' not in meters:
                meters['omics_instance_loss'] = AverageMeter()
            meters['omics_instance_loss'].update(out['omics_instance_loss'].item(), n=n_samples)
        # CE losses
        if 'wsi_ce_loss' in out:
            if 'wsi_ce_loss' not in meters:
                meters['wsi_ce_loss'] = AverageMeter()
            meters['wsi_ce_loss'].update(float(out['wsi_ce_loss'].item()), n=n_samples)
        if 'omics_ce_loss' in out:
            if 'omics_ce_loss' not in meters:
                meters['omics_ce_loss'] = AverageMeter()
            meters['omics_ce_loss'].update(float(out['omics_ce_loss'].item()), n=n_samples)
        # modality-context refinement contrast loss
        if torch.is_tensor(out.get('modref_loss', None)):
            if 'modref_loss' not in meters:
                meters['modref_loss'] = AverageMeter()
            meters['modref_loss'].update(float(out['modref_loss'].item()), n=n_samples)
    
        

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        # Check for nan/inf before backward pass
        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss detected at batch {batch_idx}, skipping backward pass")
            if int(debug_nans) == 1:
                print("=== NaN/Inf diagnostics ===")
                # Core outputs
                for k in ['loss', 'wsi_instance_loss', 'omics_instance_loss', 'wsi_ce_loss', 'omics_ce_loss',
                          'modref_loss', 'logits', 'risk']:
                    v = out.get(k, None)
                    if v is not None:
                        _tstats(f"out[{k}]", v)
                # Labels
                _tstats("label", label)
                _tstats("censorship", censorship)
                _tstats("event_time", event_time)
                # Inputs
                if isinstance(data, list) and len(data) > 0 and torch.is_tensor(data[0]):
                    _tstats("img[0]", data[0])
                elif torch.is_tensor(data):
                    _tstats("img", data)
                if isinstance(omics, list) and len(omics) > 0 and torch.is_tensor(omics[0]):
                    _tstats("omics[0]", omics[0])
                elif torch.is_tensor(omics):
                    _tstats("omics", omics)
                print("=== End diagnostics ===")
            continue
        
        loss = loss / accum_steps
        try:
            loss.backward()
        except RuntimeError as e:
            if int(debug_nans) == 1:
                print("=== Backward anomaly (loss.backward) ===")
                print(str(e))
                # Dump same diagnostics to associate with the failing batch
                for k in ['loss', 'wsi_instance_loss', 'omics_instance_loss', 'wsi_ce_loss', 'omics_ce_loss',
                          'modref_loss', 'logits', 'risk']:
                    v = out.get(k, None)
                    if v is not None:
                        _tstats(f"out[{k}]", v)
                if isinstance(data, list) and len(data) > 0 and torch.is_tensor(data[0]):
                    _tstats("img[0]", data[0])
                elif torch.is_tensor(data):
                    _tstats("img", data)
                if isinstance(omics, list) and len(omics) > 0 and torch.is_tensor(omics[0]):
                    _tstats("omics[0]", omics[0])
                elif torch.is_tensor(omics):
                    _tstats("omics", omics)
                print("=== End backward anomaly ===")
            raise
        has_pending_grads = True  # Mark that we have accumulated gradients
        if (batch_idx + 1) % accum_steps == 0:
            # Optional gradient clipping (apply before optimizer step)
            if isinstance(grad_clip_norm, (int, float)) and grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            has_pending_grads = False  # Reset flag after updating

            # After parameter update, update feature archive with current batch fused features
            
            

            # Periodic train logging aligned with print_every
            if _WANDB_AVAILABLE and (wandb.run is not None):
                try:
                    step_index = epoch * total_batches + batch_idx
                    log_items = {
                        'train/loss': float((meters.get('loss') or AverageMeter()).avg) if 'loss' in meters else float(loss.item()),
                        'train/lr': float(lr_scheduler.optimizer.param_groups[0]['lr'] if lr_scheduler else optimizer.param_groups[0]['lr'])
                    }
                    
                    if 'sk_multi_loss' in meters:
                        log_items['train/sk_multi_loss'] = float(meters['sk_multi_loss'].avg)
                    
                    if 'sk_multi_raw' in meters:
                        log_items['train/sk_multi_raw'] = float(meters['sk_multi_raw'].avg)
                    if 'wsi_instance_loss' in meters:
                        log_items['train/wsi_instance_loss'] = float(meters['wsi_instance_loss'].avg)
                    if 'omics_instance_loss' in meters:
                        log_items['train/omics_instance_loss'] = float(meters['omics_instance_loss'].avg)
                    if 'wsi_ce_loss' in meters:
                        log_items['train/wsi_ce_loss'] = float(meters['wsi_ce_loss'].avg)
                    if 'omics_ce_loss' in meters:
                        log_items['train/omics_ce_loss'] = float(meters['omics_ce_loss'].avg)
                    if 'modref_loss' in meters:
                        log_items['train/modref_loss'] = float(meters['modref_loss'].avg)
                    wandb.log(log_items, step=step_index)
                except Exception:
                    pass

        # End of iteration survival-specific metrics to calculate / log
        # Use log-risk for Cox to avoid exp overflow in c-index; keep monotonicity
        try:
            if isinstance(loss_fn, CoxLoss):
                estimate_np = out['logits'].detach().cpu().numpy()
            else:
                estimate_np = out['risk'].detach().cpu().numpy()
            # Cast to float64 and sanitize to avoid inf/nan in metric
            estimate_np = np.nan_to_num(estimate_np.astype(np.float64), posinf=1e12, neginf=-1e12)
            all_risk_scores.append(estimate_np)
        except Exception:
            # Fallback to risk if logits missing
            estimate_np = np.nan_to_num(out['risk'].detach().cpu().numpy().astype(np.float64), posinf=1e12, neginf=-1e12)
            all_risk_scores.append(estimate_np)
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        if torch.is_tensor(data):
            bag_len = data.size(1)
            n_samples = len(data)
        elif isinstance(data, list):
            bag_lens = [t.size(0) for t in data]
            bag_len = int(np.mean(bag_lens)) if len(bag_lens) > 0 else 0
            n_samples = len(data)
        else:
            bag_len = 0
            n_samples = 0
        bag_size_meter.update(bag_len, n=n_samples)
        

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch: ensure any remaining accumulated gradients are applied
    # This handles the case where the last batch doesn't align with accum_steps
    if has_pending_grads:
        # There are accumulated gradients that haven't been applied yet
        if isinstance(grad_clip_norm, (int, float)) and grad_clip_norm > 0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    # attach per-head SK loss averages for head selection (separate for WSI/OMICS)
    if wsi_head_sum is not None and wsi_head_count > 0:
        results['wsi_sk_head_losses'] = (wsi_head_sum / float(wsi_head_count)).tolist()
    if omics_head_sum is not None and omics_head_count > 0:
        results['omics_sk_head_losses'] = (omics_head_sum / float(omics_head_count)).tolist()
    
    results['lr'] = optimizer.param_groups[0]['lr']
    # Return basic info to optionally drive validation ramp-up if needed
    results['iterations_per_epoch'] = total_batches
    results['iterations'] = (epoch + 1) * total_batches - 1
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,
                      ):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []
    all_fused_features = []

    for batch_idx, batch in enumerate(loader):
        img = batch['img']
        if isinstance(img, list):
            data = [t.to(device) for t in img]
        else:
            data = img.to(device)
        label = batch['label'].to(device)
        # omics may be list of tensors per path
        omics = batch.get('omics', None)
        if isinstance(omics, list):
            omics = [t.to(device) for t in omics]
        else:
            omics = safe_list_to(omics, device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch.get('attn_mask', None)
        if isinstance(attn_mask, list):
            attn_mask = [t.to(device) for t in attn_mask]
        elif attn_mask is not None:
            attn_mask = attn_mask.to(device)
        

        # Cox-only: do not flip per-batch censorship; rely on consistent global coding

        out, log_dict = model(
            data,
            omics,
            attn_mask=attn_mask,
            label=label,
            censorship=censorship,
            loss_fn=loss_fn,
            return_attn=return_attn,
        )

        
        # Log total loss and auxiliary instance losses into meters for periodic prints
        if torch.is_tensor(data):
            n_samples = len(data)
        elif isinstance(data, list):
            n_samples = len(data)
        else:
            n_samples = 0

        # total loss
        total_loss_val = out.get('loss', None)
        if total_loss_val is not None:
            total_loss_scalar = total_loss_val.item() if isinstance(total_loss_val, torch.Tensor) else total_loss_val
            if 'loss' not in meters:
                meters['loss'] = AverageMeter()
            meters['loss'].update(total_loss_scalar, n=n_samples)

        # wsi instance loss
        if 'wsi_instance_loss' in out:
            if 'wsi_instance_loss' not in meters:
                meters['wsi_instance_loss'] = AverageMeter()
            meters['wsi_instance_loss'].update(out['wsi_instance_loss'].item(), n=n_samples)

        # omics instance loss
        if 'omics_instance_loss' in out:
            if 'omics_instance_loss' not in meters:
                meters['omics_instance_loss'] = AverageMeter()
            meters['omics_instance_loss'].update(out['omics_instance_loss'].item(), n=n_samples)
        
        
        if return_attn:
            # Some model heads may not produce all attention matrices
            if 'omic_attn' in out:
                all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
            if 'cross_attn' in out:
                all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
            if 'path_attn' in out:
                all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        # End of iteration survival-specific metrics to calculate / log
        if torch.is_tensor(data):
            bag_len = data.size(1)
            n_samples = len(data)
        elif isinstance(data, list):
            bag_lens = [t.size(0) for t in data]
            bag_len = int(np.mean(bag_lens)) if len(bag_lens) > 0 else 0
            n_samples = len(data)
        else:
            bag_len = 0
            n_samples = 0
        bag_size_meter.update(bag_len, n=n_samples)
        
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        try:
            if isinstance(loss_fn, CoxLoss):
                estimate_np = out['logits'].detach().cpu().numpy()
            else:
                estimate_np = out['risk'].detach().cpu().numpy()
            estimate_np = np.nan_to_num(estimate_np.astype(np.float64), posinf=1e12, neginf=-1e12)
            all_risk_scores.append(estimate_np)
        except Exception:
            estimate_np = np.nan_to_num(out['risk'].detach().cpu().numpy().astype(np.float64), posinf=1e12, neginf=-1e12)
            all_risk_scores.append(estimate_np)
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())
        # Collect fused features for downstream visualization if provided
        fused_feature = out.get('fused_feature', None)
        if torch.is_tensor(fused_feature):
            all_fused_features.append(fused_feature.detach().cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        # Only stack attentions that were actually collected
        if len(all_omic_attn) > 0:
            if len(all_omic_attn[0].shape) == 2:
                all_omic_attn = np.stack(all_omic_attn)
            else:
                all_omic_attn = np.vstack(all_omic_attn)
        if len(all_cross_attn) > 0:
            if len(all_cross_attn[0].shape) == 2:
                all_cross_attn = np.stack(all_cross_attn)
            else:
                all_cross_attn = np.vstack(all_cross_attn)
        if len(all_path_attn) > 0:
            if len(all_path_attn[0].shape) == 2:
                all_path_attn = np.stack(all_path_attn)
            else:
                all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        # CoxLoss expects log-risk (logits). We collected risk=exp(logits) above; convert back safely.
        log_risk = torch.log(torch.tensor(all_risk_scores).clamp_min(1e-12)).unsqueeze(1)
        surv_loss_dict = loss_fn(logits=log_risk,
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        # Save fused features if available
        if len(all_fused_features) > 0:
            try:
                dumps['fused_feature'] = np.concatenate(all_fused_features, axis=0)
            except Exception:
                dumps['fused_feature'] = np.vstack(all_fused_features)
        if return_attn:
            if isinstance(all_omic_attn, np.ndarray):
                dumps['all_omic_attn'] = all_omic_attn
            if isinstance(all_cross_attn, np.ndarray):
                dumps['all_cross_attn'] = all_cross_attn
            if isinstance(all_path_attn, np.ndarray):
                dumps['all_path_attn'] = all_path_attn
    return results, dumps


