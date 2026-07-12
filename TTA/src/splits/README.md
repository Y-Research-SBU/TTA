# Survival Splits

This directory stores the train, validation, and test split CSVs used by the survival experiments.

The split files follow the same format as [MMP: Multimodal Prototyping for Cancer Survival Prediction](https://github.com/mahmoodlab/MMP). Download the corresponding survival split files from the MMP repository and place them under this directory.

The cohort wrappers in `scripts/survival/` expect split folders named by cohort and fold, for example:

```text
splits/
  survival/
    TCGA_BRCA_overall_survival_k=0/
      train.csv
      val.csv
      test.csv
```

Each split CSV should contain the patient identifiers and survival annotations required by `training/main_survival.py`.
