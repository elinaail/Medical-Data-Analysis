# Clinical CatBoost experiment

This experiment is separate from the FastAPI service. It does not change
`med_data_analysis-main/app`, the production model, or the API feature order.

## Goal

Compare two CatBoost setups:

1. Baseline: existing statistical/frequency ECG features from `pipeline/features.py`.
2. Clinical: baseline features plus additional rhythm, HRV-like, and QRS-voltage features
   from `pipeline/clinical_features.py`.

The extra features include:

- R-peak count.
- RR interval statistics.
- Mean and variability of heart rate.
- SDNN, RMSSD, pNN50.
- R-peak amplitudes.
- PR, QRS, QT, QTc, ST, and T-wave interval features from P-QRS-T delineation.
- Q-wave depth/width, Q/R ratio, approximate pathological-Q ratio.
- ST deviation and ST slope.
- T-wave amplitude and T-wave inversion ratio.
- Approximate QRS voltage features around detected R-peaks.
- Approximate Sokolow-Lyon and Cornell voltage criteria.

## What to open

Open the repository folder:

```text
work/repo
```

The main files for this experiment are:

```text
pipeline/clinical_features.py
pipeline/clinical_catboost_experiment.py
requirements-clinical.txt
docs/clinical_catboost_experiment.md
```

## Where to put the dataset

Download the PTB-XL pickle file from the project Google Drive and put it here:

```text
datasets/raw/clean_ptbxl_with_ecg_n_diagnostic_superclass.pkl
```

The script uses this path by default.

If your file is somewhere else, pass it explicitly with `--data-path`.

## Install experiment dependencies

From the repository root (`work/repo`):

```bash
uv pip install --python med_data_analysis-main/.venv/Scripts/python.exe -r requirements-clinical.txt
```

CatBoost is needed for training. NeuroKit2 is used for ECG R-peak detection when
available. If NeuroKit2 fails on a signal, the code falls back to a SciPy peak detector.

## Quick smoke run

Use a small subset first:

```bash
uv run --project med_data_analysis-main python -m pipeline.clinical_catboost_experiment --sample-rows 500 --iterations 50 --task both
```

This checks that data loading, feature generation, and model training work.

## Full runs

Run only the 5 binary superclass models:

```bash
uv run --project med_data_analysis-main python -m pipeline.clinical_catboost_experiment --task binary
```

Run only the 16-class multiclass model:

```bash
uv run --project med_data_analysis-main python -m pipeline.clinical_catboost_experiment --task multiclass --output-dir outputs/clinical_catboost_multiclass
```

Run both binary and multiclass experiments:

```bash
uv run --project med_data_analysis-main python -m pipeline.clinical_catboost_experiment --task both
```

Outputs are saved to:

```text
outputs/clinical_catboost/
```

Important files:

```text
comparison_metrics.csv
comparison_metrics.json
baseline_catboost_metrics.json
clinical_catboost_metrics.json
baseline_catboost_multiclass_metrics.json
clinical_catboost_multiclass_metrics.json
*.cbm
```

## How to explain this at defense

We did not change the production FastAPI pipeline. Instead, we created a separate
research experiment with clinically interpretable ECG features. The experiment compares
the existing baseline feature set with an extended feature set that includes rhythm,
HRV-like, P-QRS-T interval, ST/T/Q-wave morphology, and QRS-voltage features. We also
evaluate two task formulations: 5 binary superclass classifiers and one 16-class
multiclass classifier for the combined `combo_class` target. This makes the change safe:
if quality improves, we can later retrain and version a new production model; if it does
not, the current service remains unaffected.
