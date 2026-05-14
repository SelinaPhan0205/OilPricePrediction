# Oil Price Prediction

Daily crude-oil direction prediction using market, macro, supply, and geopolitical signals.

The current project target is **daily T -> T+1 oil return direction**.
For review/submission, start with `final_results/`; it contains the final model artifacts, final reports, and compact metadata files.

- Target column: `oil_return_fwd1`
- Target rule: `oil_return_fwd1 > 0`
- Target date column: `oil_return_fwd1_date`
- Main final model artifact: `ml/classification/results_daily_selected_model_v1/selected_daily_model.joblib`

Weekly experiments are kept as background analysis only. The main deliverable is the daily classification workflow.

## Current Final Result

Best saved daily model:

| Item | Value |
| --- | --- |
| Model | `LGBMClassifier` |
| Feature count | 13 |
| Added regime features | `ovx_high_regime_lag1`, `yield_10y_3y_change_lag1_slog1p`, `oil_high_vol_regime_lag1` |
| Test split | from `2023-01-01` onward |
| Test accuracy | `0.5548` |
| Test F1 macro | `0.5442` |
| Test ROC-AUC | `0.5697` |

Artifacts:

- `ml/classification/results_daily_selected_model_v1/selected_daily_model.joblib`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_metrics.csv`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_features.csv`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_predictions.csv`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_summary.md`

The same files are copied into `final_results/model/` so the final deliverable is easy to find.

## Repository Layout

```text
data/
  raw/                         # Source market, macro, supply, GDELT, and extended regime data
  processed/                   # Final leakage-safe model datasets kept for reproducibility
docs/
  summary/                     # Stable pipeline summaries
  FINAL_DAILY_WEEKLY_MODEL_METHOD_2026-05-14.md
  MODEL_EXPERIMENT_HISTORY_DAILY_WEEKLY_LONG_DATA_2026-05-14.md
  OLD_VS_NEW_DATA_RESULTS_2026-05-14.md
ml/
  classification/              # Daily classification workflow
  regression/                  # Legacy regression workflow
scripts/
  step1_load_inspect.py        # Main preprocessing pipeline
  step2_cleaning.py
  step3_integration.py
  step4_transformation.py
  step4b_fix_leakage.py
  step5b_processing.py
  step6_quality_check.py
  ingest_ext_market_regime.py
  build_ext_market_regime_dataset.py
  train_daily_selected_model.py
```

## Setup

Create a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. The final saved artifact was produced on Windows.

## Data Pipeline

Run the maintained preprocessing chain:

```powershell
python scripts\step1_load_inspect.py
python scripts\step2_cleaning.py
python scripts\step3_integration.py
python scripts\step4_transformation.py
python scripts\step4b_fix_leakage.py
python scripts\step5b_processing.py
python scripts\step6_quality_check.py
```

Build the extended market/regime dataset used by the final selected model:

```powershell
python scripts\ingest_ext_market_regime.py
python scripts\build_ext_market_regime_dataset.py
```

Key processed files kept in the repository:

- `data/processed/dataset_final_noleak_processed.csv`
- `data/processed/dataset_step4_noleak.csv`
- `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- `data/processed/dataset_final_noleak_ext_market_regime_v1_features.csv`
- `data/processed/dataset_final_noleak_ext_market_regime_v1_report.json`

## Training The Final Daily Model

Run:

```powershell
python scripts\train_daily_selected_model.py
```

Or use the helper runner:

```powershell
.\scripts\run_daily_training.ps1
```

The training script reads:

- `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- `data/processed/dataset_step4_noleak.csv`
- `ml/classification/results_step5b_v2/step5_selected_features.csv`
- `data/processed/dataset_final_noleak_ext_market_regime_v1_features.csv`

and writes the selected model artifacts to:

```text
ml/classification/results_daily_selected_model_v1/
```

## Reports

Final report files:

- `docs/FINAL_DAILY_WEEKLY_MODEL_METHOD_2026-05-14.md`
- `docs/MODEL_EXPERIMENT_HISTORY_DAILY_WEEKLY_LONG_DATA_2026-05-14.md`
- `docs/OLD_VS_NEW_DATA_RESULTS_2026-05-14.md`
- `docs/summary/CLASSIFICATION_FINAL_PIPELINE.md`
- `docs/summary/PROCESSING_PIPELINE.md`
- `docs/summary/EDA_CLASSIFICATION_PIPELINE.md`

Planning notes, private agent files, cache folders, and temporary experiment outputs are ignored by Git. They can remain in the local workspace for audit/history, but the final deliverable is the compact `final_results/` folder plus the maintained source code and reproducible data files listed above.

## Notes

- This is a weak-signal daily financial classification task. The reported metrics should be interpreted as modest directional edge, not high-confidence price prediction.
- Validation uses chronological splits. Do not use random splits for headline results.
- Same-day and future-derived columns must be checked carefully before modeling to avoid leakage.
