# Results

This directory contains example output files.

- `fold_scores*` and `fold_summary*` show the aggregate result CSV format.
- `BLCA/k=4/` shows a per-fold output directory with `config.json`, `train.log`, `summary.csv`, `test_results.pkl`, `all_dumps.h5`, and `s_checkpoint.pth`.

New runs are saved here by default. To change the output path, modify `save_dir_root` in `scripts/survival/tta.sh` or pass a different `--results_dir` through the training entry point.
