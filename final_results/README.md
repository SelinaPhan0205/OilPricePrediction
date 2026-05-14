# Final Results Bundle

This folder contains the small set of artifacts intended for review/submission.
The broader experiment history is kept in the repository workspace but ignored by Git when it is temporary or not part of the final deliverable.

## Final Model

Use the daily selected model:

- `model/selected_daily_model.joblib`
- `model/selected_daily_model_metrics.csv`
- `model/selected_daily_model_features.csv`
- `model/selected_daily_model_predictions.csv`
- `model/selected_daily_model_summary.md`

Model summary:

- Target: `oil_return_fwd1 > 0`
- Horizon: daily `T -> T+1`
- Model: `LGBMClassifier`
- Feature count: `13`
- Test accuracy: `0.5548`
- Test F1 macro: `0.5442`
- Test ROC-AUC: `0.5697`

## Final Reports

- `reports/FINAL_DAILY_WEEKLY_MODEL_METHOD_2026-05-14.md`
- `reports/MODEL_EXPERIMENT_HISTORY_DAILY_WEEKLY_LONG_DATA_2026-05-14.md`
- `reports/OLD_VS_NEW_DATA_RESULTS_2026-05-14.md`
- `reports/DAILY_TARGET_SPEC_2026-05-12.md`

## Data Metadata

- `data_metadata/dataset_final_noleak_ext_market_regime_v1_features.csv`
- `data_metadata/dataset_final_noleak_ext_market_regime_v1_report.json`
- `data_metadata/old_vs_new_data_result_comparison.csv`

The full datasets remain in `data/processed/` and `data/raw/`. They are referenced by the root `README.md` and the training scripts rather than duplicated here.
