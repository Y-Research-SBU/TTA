#!/bin/bash
export WANDB_MODE=disabled

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

case "${task}" in
  *_uni_survival)
    feat='uni'
    ;;
  *_resnet50_survival)
    feat='resnet50'
    ;;
esac
input_dim=1024
# Set input_dim by feature family
if [ "${feat}" = "resnet50" ]; then
  input_dim=768
elif [ "${feat}" = "uni" ]; then
  input_dim=1024
fi
# Optional external override (e.g., for specific datasets)
if [ -n "${IN_DIM_OVERRIDE}" ]; then
  input_dim=${IN_DIM_OVERRIDE}
fi
mag='20x'
patch_size=256
seed=1
num_workers=8
es_metric='loss'

# Fixed WSI bag size for BatchedSK (pad/cut). Use 4096 or adjust by GPU.
bag_size='4096'
# Use batch size 16 with grad accumulation to get effective 64
batch_size=32
max_epoch=30
lr=0.0001
wd=0.00001
lr_scheduler='cosine'
opt='adamW'
grad_accum=1
dropout=0.3
loss_fn='cox'
n_label_bin=4
alpha=0.5
es_flag=1
es_min_epochs=5
es_patience=30
save_dir_root=results
loss_warmup_epochs=0

# ===== SK multi-head =====
# If num_heads<=1, multi-head is disabled. Each modality can toggle its own SK loss.
num_heads=5
wsi_use_sk_multi=1
wsi_sk_weight=0.01
omics_use_sk_multi=1
omics_sk_weight=0.01



# Multimodal args (coattn for multimodal; set to 'gene' or 'histo' for ablations)
modality_type='multi'
# model_type: 'otmil_label'
model_type='otmil_label'
# fusion type for multimodal
fusion_type='coattn'
num_coattn_layers=1

# ===== Modality-context refinement (anchor+self-attn) =====
enable_modality_refine=1
modref_weight=0.5
modref_tau=0.1
modref_layers=1
modref_apply_to_tokens=1
# Scheduling (default off)
modref_freeze_epochs=0
modref_apply_switch_epoch=-1
modref_decay_start=-1
modref_decay_epochs=0

# ===== OT-as-weights (safe defaults: mix, beta=0.3) =====
use_ot_as_weights=1
ot_weight_strategy='mix'
ot_mix_coeff=0.5

ot_kl_weight=0.1

# ===== OT ablations =====
# for kmeans ablations, set image_pseudo_label=0,omics_pseudo_label=0, use_ot_as_weights=0,ot_attn_trainable=0 at the same time
ot_mode='ubot'          # ubot | ubot_fixed_rho | balanced | kmeans
rho_fixed=1           # used when ot_mode=ubot_fixed_rho

# Optional environment overrides
if [ -n "${OT_MODE_OVERRIDE}" ]; then
  ot_mode=${OT_MODE_OVERRIDE}
fi
if [ -n "${RHO_FIXED_OVERRIDE}" ]; then
  rho_fixed=${RHO_FIXED_OVERRIDE}
fi

if [ -n "${LR_OVERRIDE}" ]; then
  lr=${LR_OVERRIDE}
fi
if [ -n "${FUSION_TYPE_OVERRIDE}" ]; then
  fusion_type=${FUSION_TYPE_OVERRIDE}
fi
if [ -n "${DROPOUT_OVERRIDE}" ]; then
  dropout=${DROPOUT_OVERRIDE}
fi
if [ -n "${ES_MIN_EPOCHS_OVERRIDE}" ]; then
  es_min_epochs=${ES_MIN_EPOCHS_OVERRIDE}
fi

# Omics OT args
omics_use_ot=1
omics_proto_num=16
omics_ot_impl='batchot'
omics_rho_ramp=1

# Image OT args
image_use_ot=1
image_proto_num=16
image_ot_impl='batchot'
image_rho_ramp=1
image_feat_norm='none'
## Histo head: 'coattn' or 'otsurv'
histo_head='coattn'

ot_attn_trainable=0
proto_ortho_weight=0.0

# Pseudo-label generation 
image_pseudo_label=1
omics_pseudo_label=1
# Soft CE controls: disable and zero weights to avoid redundancy with SK
wsi_use_ce=1
wsi_ce_weight=0.5
omics_use_ce=1
omics_ce_weight=0.5

# ===== Shared Prototypes (Scheme A) switches =====
shared_prototypes=1
shared_proto_num=32
shared_proto_dim=256
shared_tau=0.5

# ===== Joint OT (Scheme B) switch =====
joint_ot_single_path=1

feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

# Scheduler & warmup policy
# Respect selected lr_scheduler by default; apply a gentle warmup only when scheduler is not 'constant'.
curr_lr_scheduler=$lr_scheduler
warmup=0
# Allow explicit override via env var WARMUP_EPOCHS_OVERRIDE
if [ -n "${WARMUP_EPOCHS_OVERRIDE}" ]; then
  warmup=${WARMUP_EPOCHS_OVERRIDE}
else
  if [ "$curr_lr_scheduler" != 'constant' ]; then
    # default gentle warmup for cosine/linear
    warmup=15
  fi
fi

# Identify feature paths
all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir
  then
    continue
  fi

  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done

echo $feat_dir

# Actual command 
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--warmup_epochs ${warmup} \\
--accum_steps ${grad_accum} \\
--dropout ${dropout} \\
--wd ${wd} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${bag_size} \\
--val_bag_size ${bag_size} \\
--batch_size ${batch_size} \\
--seed ${seed} \\
--num_workers ${num_workers} \\
--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
--early_stopping ${es_flag} \\
--es_min_epochs ${es_min_epochs} \\
--es_patience ${es_patience} \\
--es_metric ${es_metric} \\
--num_coattn_layers ${num_coattn_layers} \\
--modality_type ${modality_type} \\
--model_type ${model_type} \\
--fusion_type ${fusion_type} \\
--label_num_coattn_layers ${num_coattn_layers} \\
--enable_modality_refine ${enable_modality_refine} \\
--modref_weight ${modref_weight} \\
--modref_tau ${modref_tau} \\
--modref_layers ${modref_layers} \\
--modref_apply_to_tokens ${modref_apply_to_tokens} \\
--modref_freeze_epochs ${modref_freeze_epochs} \\
--modref_apply_switch_epoch ${modref_apply_switch_epoch} \\
--modref_decay_start ${modref_decay_start} \\
--modref_decay_epochs ${modref_decay_epochs} \\
--use_ot_as_weights ${use_ot_as_weights} \\
--ot_weight_strategy ${ot_weight_strategy} \\
--ot_mix_coeff ${ot_mix_coeff} \\
--ot_kl_weight ${ot_kl_weight} \\
--loss_warmup_epochs ${loss_warmup_epochs} \\
--grad_clip_norm 5.0 \\
"

# Add Omics OT flags
if [ "$omics_use_ot" -eq 1 ]; then
  cmd+=" --use_omics_ot --gene_simple_head"
fi

cmd+=" \\
--omics_proto_num ${omics_proto_num} \\
--omics_ot_impl ${omics_ot_impl} \\
"

# Add Omics OT rho ramp flag
if [ "$omics_rho_ramp" -eq 1 ]; then
  cmd+=" --enable_omics_rho_ramp"
fi

if [ "$image_rho_ramp" -eq 1 ]; then
  cmd+=" --enable_image_rho_ramp"
fi

# Add Image OT flags
if [ "$image_use_ot" -eq 1 ]; then
  cmd+=" --use_image_ot"
fi

cmd+=" \\
--image_proto_num ${image_proto_num} \\
--image_ot_impl ${image_ot_impl} \\
--image_feat_norm ${image_feat_norm} \\
--histo_head ${histo_head} \\
--image_pseudo_label ${image_pseudo_label} \\
--omics_pseudo_label ${omics_pseudo_label} \\
--wsi_use_ce ${wsi_use_ce} \\
--wsi_ce_weight ${wsi_ce_weight} \\
--omics_use_ce ${omics_use_ce} \\
--omics_ce_weight ${omics_ce_weight} \\
--ot_attn_trainable ${ot_attn_trainable} \\
--proto_ortho_weight ${proto_ortho_weight} \\
"

# Append WSI SK multi-head flags
cmd+=" --num_heads ${num_heads} --wsi_use_sk_multi ${wsi_use_sk_multi} --wsi_sk_weight ${wsi_sk_weight} --omics_use_sk_multi ${omics_use_sk_multi} --omics_sk_weight ${omics_sk_weight}"
cmd+=" --sk_every 3"

# Append Shared Prototypes (Scheme A) and Joint OT (Scheme B) flags
cmd+=" --shared_prototypes ${shared_prototypes} --shared_proto_num ${shared_proto_num} --shared_proto_dim ${shared_proto_dim} --shared_tau ${shared_tau}"
cmd+=" --joint_ot_single_path ${joint_ot_single_path}"

cmd+=" --ot_mode ${ot_mode}"
cmd+=" --rho_fixed ${rho_fixed}"

cmd+=" --debug_nans 0"

eval "$cmd"