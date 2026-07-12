# Data Directory

Use this directory as the default local root for WSI feature data used by TTA.

Preprocessed `.h5` feature files extracted with UNI and ResNet-50 are available from [LIUWJ/Data_TMI](https://huggingface.co/datasets/LIUWJ/Data_TMI). UNI features were extracted with the UNI preprocessing pipeline provided by [mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT).

The training scripts look for downloaded feature folders using the layout below.

## Layout

The training scripts expect one folder per cohort and feature family. For BRCA with UNI features:

```text
data/
  tcga_brca_uni/
    extracted_mag20x_patch256_fp/
      uni/
        feats_h5/
          <slide_id>.h5
```

For ResNet-50 features:

```text
data/
  tcga_brca_resnet50/
    extracted_mag20x_patch256_fp/
      resnet50/
        feats_h5/
          <slide_id>.h5
```

The feature family is inferred from the task suffix:

```text
*_uni_survival       -> uni
*_resnet50_survival  -> resnet50
```

The default patching settings are `mag=20x` and `patch_size=256`, as configured in `TTA/src/scripts/survival/tta.sh`.

Feature folders can be placed under this `data/` directory or under an external data root passed to the run script.

## Metadata

Survival splits and omics metadata are stored with the code:

```text
TTA/src/splits/
TTA/src/data_csvs/
```

These files follow the data organization used by MMP.

## Example

From `TTA/src`, run:

```bash
./scripts/survival/brca_uni_surv.sh 0 tta ../../data
```

The third argument points to this `data/` directory.
