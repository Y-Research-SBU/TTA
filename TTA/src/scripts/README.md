# Experiment Scripts

This directory contains shell scripts for running TTA survival experiments.

The main configuration script is `scripts/survival/tta.sh`. Several cohort-level wrapper scripts are also included as examples of how to launch five-fold survival experiments on specific datasets:

```text
scripts/
  survival/
    brca_uni_surv.sh
    blca_uni_surv.sh
    stad_uni_surv.sh
    kirc_uni_surv.sh
    tta.sh
```

Run the scripts from `TTA/src`. The first argument is the GPU id, the second argument is the configuration script name, and the optional third argument is the data root:

```bash
bash ./scripts/survival/brca_uni_surv.sh 0 tta ../../data
```

If the third argument is omitted, the wrappers use `TTA_DATA_ROOT` when it is set, otherwise they default to `../../data`.
