# Together, Then Apart: Balancing Alignment and Distinctiveness for Multimodal Survival Analysis

[![Paper](https://img.shields.io/badge/Paper-ArXiv%202511.18089-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2511.18089)
[![Project](https://img.shields.io/badge/Project-Website-0877c9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://y-research-sbu.github.io/TTA)
[![Code](https://img.shields.io/badge/GitHub-Code-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Y-Research-SBU/TTA)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/LIUWJ/Data_TMI)

TTA is a multimodal survival analysis framework for Whole Slide Images (WSIs) and transcriptomics. In multimodal prognosis, histopathology and genomic profiles should agree on shared survival-relevant factors, but each modality also carries distinct evidence. Over-aligning the two modalities can erase modality-specific prognostic signals and produce less discriminative patient representations.

TTA addresses this with a **Together, Then Apart** design. The **Together** stage aligns WSI patch tokens and omics tokens through shared prototypes and joint unbalanced optimal transport, producing structured prototype-level representations instead of relying on noisy token-level matching. The **Apart** stage then preserves modality-specific information with modality-context refinement and distinctiveness regularization before multimodal survival prediction. This repository contains the code, example outputs, and code documentation for reproducing the paper experiments.

![TTA method overview](docs/assets/TTA-method.png)

## Repository Structure

```text
.
|-- env.yaml                  # conda environment
|-- data/                     # local WSI feature root
|-- docs/                     # GitHub Pages project website
`-- TTA/                      # runnable code
    `-- src/
        |-- scripts/          # reference run scripts and default TTA config
        |-- training/         # training and evaluation entry points
        |-- mil_models/       # TTA model and fusion modules
        |-- wsi_datasets/     # WSI/omics survival dataset loaders
        |-- utils/            # losses and helper utilities
        |-- splits/           # survival split files
        |-- data_csvs/        # omics metadata and RNA CSVs
        `-- results/          # example outputs and default run directory
```

The runnable code is under `TTA/src`. The root-level `data/` directory is the default location for local WSI feature folders.

## Installation

Create the conda environment from the repository root:

```bash
conda env create -f env.yaml
conda activate tta
```

## Data Preparation

The code expects two data sources:

- WSI patch features in `.h5` or `.pt` format.
- Omics CSV files and survival split CSVs used by the training pipeline.

### WSI Features

Preprocessed WSI features are available from [LIUWJ/Data_TMI](https://huggingface.co/datasets/LIUWJ/Data_TMI). The release contains feature files extracted with UNI and ResNet-50 encoders. UNI features are extracted with the UNI preprocessing pipeline provided by [mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT).

Place downloaded features under the root-level `data/` directory:

```text
data/
  tcga_brca_uni/
    extracted_mag20x_patch256_fp/
      uni/
        feats_h5/
          <slide_id>.h5
```

For ResNet-50 features, use the corresponding feature folder:

```text
data/
  tcga_brca_resnet50/
    extracted_mag20x_patch256_fp/
      resnet50/
        feats_h5/
          <slide_id>.h5
```

The feature family is inferred from the task suffix in `TTA/src/scripts/survival/tta.sh`:

```text
*_uni_survival       -> uni
*_resnet50_survival  -> resnet50
```

The default WSI preprocessing convention is `mag=20x` and `patch_size=256`. The default input dimensions are `1024` for UNI features and `768` for ResNet-50 features.

### Splits and Omics CSVs

Survival splits and omics CSVs follow the organization used by [MMP](https://github.com/mahmoodlab/MMP). Place the split files under `TTA/src/splits/` and the omics files under `TTA/src/data_csvs/`, keeping the same directory structure and CSV format as MMP.

## Running Experiments

Run commands from the source directory:

```bash
cd TTA/src
```

Run BRCA with the default TTA configuration:

```bash
bash ./scripts/survival/brca_uni_surv.sh 0 tta
```

The first argument is the GPU id and the second argument is the experiment configuration script name. The command above calls:

```text
scripts/survival/tta.sh
```

To use a custom data location, pass it as the third argument:

```bash
bash ./scripts/survival/brca_uni_surv.sh 0 tta /path/to/data
```

or set:

```bash
export TTA_DATA_ROOT=/path/to/data
bash ./scripts/survival/brca_uni_surv.sh 0 tta
```

## Important Configuration Switches

Most experiment settings can be changed directly in `TTA/src/scripts/survival/tta.sh`. This section highlights the switches most relevant to the TTA pipeline and ablations; ordinary runtime defaults are listed afterward.

### Core TTA Switches

| Argument | Main value | Description |
| --- | --- | --- |
| `modality_type` | `multi` | Use both WSI and omics modalities.|
| `model_type` | `otmil_label` | TTA model implementation. |
| `fusion_type` | `coattn` | Apply co-attention over prototype tokens. `concat`, `sum`, and `mlp` are late-fusion alternatives. |
| `shared_prototypes` | `1` | Use one shared prototype bank for WSI and omics tokens in the Together stage. |
| `shared_proto_num` | `32` | Number of shared prototypes used by the main model. |
| `joint_ot_single_path` | `1` | Compute OT assignments once on concatenated WSI and omics tokens, then split assignments back by modality. Setting `0` computes modality-specific OT assignments separately. |
| `ot_mode` | `ubot` | Use unbalanced OT. `balanced` disables the semi-relaxed/unbalanced behavior; `ubot_fixed_rho` uses a fixed rho value; `kmeans` is the hard-assignment ablation. |
| `image_pseudo_label` | `1` | Enable OT-based soft assignments for WSI tokens. For the `kmeans` ablation, set this to `0`. |
| `omics_pseudo_label` | `1` | Enable OT-based soft assignments for omics tokens. For the `kmeans` ablation, set this to `0`. |
| `use_ot_as_weights` | `1` | Use OT-produced assignments as token-to-prototype aggregation weights. This changes how tokens are pooled into prototype tokens. |
| `ot_weight_strategy` | `mix` | Blend OT assignment weights with softmax assignment weights for stability. With `replace`, OT weights fully replace softmax weights. |
| `enable_modality_refine` | `1` | Enable the Apart-stage modality-context refinement. |
| `modref_apply_to_tokens` | `1` | Use refined tokens for downstream prediction. If set to `0`, the refinement loss is computed but the original tokens are kept for downstream prediction. |

OT pseudo labels provide structured soft assignments from tokens to prototypes, supporting the Together-stage prototype alignment described in the paper.

### Ablation-Related Notes

- `fusion_type`: `coattn` is the main setting; `concat`, `sum`, and `mlp` are late-fusion variants.
- `ot_mode`: `ubot` is the main setting; `balanced`, `ubot_fixed_rho`, and `kmeans` correspond to OT assignment ablations.
- `kmeans` ablation: disable the OT pseudo-label and OT-as-weights paths together (`image_pseudo_label=0`, `omics_pseudo_label=0`, `use_ot_as_weights=0`).
- `use_ot_as_weights` and soft CE are different mechanisms: `use_ot_as_weights` changes token pooling weights, while soft CE adds an auxiliary training signal that encourages assignment logits to match OT pseudo labels.
- `num_heads<=1` disables the multi-head Sinkhorn consistency path.

### Additional Default Settings

The following values are default implementation and training settings used by the main experiments. They have clear runtime meanings, but they are not the primary switches behind the reported method comparisons.

| Argument | Main value | Description |
| --- | --- | --- |
| `shared_proto_dim` | `256` | Shared prototype dimension. |
| `shared_tau` | `0.5` | Temperature for shared prototype assignment. |
| `image_proto_num` | `16` | Number of image prototypes. |
| `omics_proto_num` | `16` | Number of omics prototypes. |
| `image_ot_impl` | `batchot` | Image-side OT solver. |
| `omics_ot_impl` | `batchot` | Omics-side OT solver. |
| `enable_image_rho_ramp` | `1` | Enable image-side rho ramp for stable OT assignment. |
| `enable_omics_rho_ramp` | `1` | Enable omics-side rho ramp for stable OT assignment. |
| `image_feat_norm` | `none` | Image feature normalization. |
| `ot_mix_coeff` | `0.5` | Mixing coefficient between softmax weights and OT weights. |
| `ot_kl_weight` | `0.1` | OT regularization weight. |
| `wsi_use_ce` | `1` | Add WSI auxiliary soft cross-entropy against OT assignments. |
| `omics_use_ce` | `1` | Add omics auxiliary soft cross-entropy against OT assignments. |
| `wsi_ce_weight` | `0.5` | WSI auxiliary CE weight. |
| `omics_ce_weight` | `0.5` | Omics auxiliary CE weight. |
| `modref_weight` | `0.5` | Modality-context refinement loss weight. |
| `modref_tau` | `0.1` | Modality-context refinement temperature. |
| `modref_layers` | `1` | Number of refinement layers. |
| `num_heads` | `5` | Number of Sinkhorn heads. |
| `wsi_use_sk_multi` | `1` | Enable WSI multi-head Sinkhorn consistency. |
| `omics_use_sk_multi` | `1` | Enable omics multi-head Sinkhorn consistency. |
| `wsi_sk_weight` | `0.01` | Small auxiliary weight for enabling WSI multi-head Sinkhorn consistency without letting it dominate the Cox survival objective. |
| `omics_sk_weight` | `0.01` | Small auxiliary weight for enabling omics multi-head Sinkhorn consistency without letting it dominate the Cox survival objective. |
| `sk_every` | `3` | Compute the consistency loss every 3 batches to reduce training overhead. |
| `histo_head` | `coattn` | Histology branch head. |
| `num_coattn_layers` | `1` | Multimodal co-attention depth. |
| `label_num_coattn_layers` | `1` | Label/prototype co-attention depth. |
| `batch_size` | `32` | Training batch size. |
| `train_bag_size` | `4096` | WSI token sampling or padding size for training. |
| `val_bag_size` | `4096` | WSI token sampling or padding size for validation/testing. |
| `max_epochs` | `30` | Maximum number of epochs. |
| `lr` | `1e-4` | Learning rate. |
| `wd` | `1e-5` | Weight decay. |
| `opt` | `adamW` | Optimizer. |
| `lr_scheduler` | `cosine` | Learning-rate scheduler. |
| `warmup_epochs` | `15` | Warmup used with non-constant schedulers. |
| `dropout` | `0.3` | Dropout rate. |
| `loss_fn` | `cox` | Survival objective. |
| `n_label_bins` | `4` | Number of discrete survival bins. |
| `nll_alpha` | `0.5` | NLL alpha value passed by the wrapper. |
| `early_stopping` | `1` | Enable early stopping. |
| `es_min_epochs` | `5` | Minimum epochs before early stopping. |
| `es_patience` | `30` | Early-stopping patience. |
| `es_metric` | `loss` | Early-stopping metric. |
| `seed` | `1` | Random seed. |
| `num_workers` | `8` | Data-loader workers. |
| `grad_clip_norm` | `5.0` | Gradient clipping norm. |
| `save_dir_root` | `results` | Output directory root under `TTA/src`. |
| `IN_DIM_OVERRIDE` | optional | Override feature dimension when using another encoder. |

## Outputs

By default, outputs are saved under:

```text
TTA/src/results/
```

Each run creates a timestamped directory:

```text
results/<task>/<k-fold>/<exp_code>/<exp_code>::<timestamp>/
```

Typical files include:

```text
config.json
train.log
summary.csv
summary.csv.json
test_results.pkl
all_dumps.h5
s_checkpoint.pth
```

After the fifth fold (`k=4`), fold-level summaries are aggregated under:

```text
results/<task>/k=agg/<exp_code>/
```

Example result files are included under `TTA/src/results/`.

## Citation

```bibtex
@inproceedings{liu2026together,
  title     = {Together, Then Apart: Balancing Alignment and Distinctiveness for Multimodal Survival Analysis},
  author    = {Liu, Wenjing and Ren, Qin and Zhang, Wen and Lin, Yuewei and You, Chenyu},
  booktitle = {Proceedings of the 19th European Conference on Computer Vision},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [MMP: Multimodal Prototyping for Cancer Survival Prediction](https://github.com/mahmoodlab/MMP) and related open-source work in computational pathology and multimodal survival analysis.
