# TabPFN CC18 Reproduction

This is the minimal GitHub folder for the TabPFN reproduction report.

## Files

- `reproduce_cc18.py`: one-file reproduction script.
- `tabpfn_cc18_safe_corrected_ovo_results.csv`: reference result CSV from the report.
- `requirements.txt`: dependencies.
- `.gitignore`: ignores caches, logs, and virtual environments.

## Run

```bash
pip install -r requirements.txt
python reproduce_cc18.py --device cuda --overwrite
```

Use `--device cpu` if CUDA is unavailable. Without `--overwrite`, the script resumes from the existing CSV and skips finished datasets.

## Protocol

- 18 OpenML-CC18 small classification datasets.
- 5 stratified 50/50 train-test splits.
- Random seeds: 0, 1, 2, 3, 4.
- Binary tasks use ROC AUC.
- Multiclass tasks use one-vs-one ROC AUC.

## Result

The reference CSV reports mean ROC AUC `0.9402` across the 18 datasets.
