# Selected Daily Model v1

Target: daily `oil_return_fwd1 > 0`.

- Model: `LGBMClassifier`
- Features: `13`
- Added extended features: `ovx_high_regime_lag1, yield_10y_3y_change_lag1_slog1p, oil_high_vol_regime_lag1`
- Test Accuracy: `0.5548`
- Test F1_macro: `0.5442`
- Test AUC: `0.5697`
- Confusion matrix: TP=169, FP=125, TN=297, FN=249

Weekly is not part of this model.
