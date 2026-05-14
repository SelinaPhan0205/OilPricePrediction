# Phương pháp cuối cùng cho Daily và Weekly Oil Direction Model

Ngày chốt: 2026-05-14

Repo: `E:\Project\cs313-oil-prediction`

## 1. Chốt kết quả cuối cùng

| Target | Model chốt | Vì sao chốt | Metric chính |
|---|---|---|---|
| Daily `T -> T+1` | Compact LightGBM trên data 2015-2026 | Daily long-history kém hơn; compact feature set ít nhiễu hơn | Accuracy `0.5548`, AUC `0.5697` |
| Weekly `T -> T+1 week` | Rolling 8-year Logistic Regression trên long-history 2007-2026 no-ACLED | Tận dụng data mới nhưng tránh regime drift; AUC tốt nhất và 11 folds | Accuracy `0.5628`, AUC `0.5685` |

Ghi chú quan trọng:

- Weekly old compact có mean accuracy cao hơn một chút: `0.5686`.
- Nhưng model weekly chốt sau crawl thêm data là rolling 8-year vì dùng long-history hợp lý hơn, có AUC cao hơn, và kiểm định dài hơn.
- Không claim rằng thêm data làm accuracy tốt hơn. Claim đúng là thêm data giúp coverage và giúp thử rolling-window robust hơn.

## 2. Daily final model

### 2.1. Target

Daily target:

```text
target = oil_return_fwd1
target_class = 1 nếu oil_return_fwd1 > 0, ngược lại 0
```

Ý nghĩa:

- Mỗi dòng tại ngày `T` dùng thông tin khả dụng tới ngày `T`.
- Label là hướng lợi suất dầu ở ngày target `T+1`.
- Cột target date là `oil_return_fwd1_date`.

### 2.2. Dataset

Dataset chốt:

- File: `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- Date range: `2015-01-07 -> 2026-03-19`
- Target date range: `2015-01-08 -> 2026-03-20`
- Rows: `2922`
- Columns: `57`
- Target up-rate: khoảng `0.5089`

Nguồn dữ liệu:

- Market: dầu, USD index, S&P 500, VIX.
- FRED/macro: yield spread và các biến vĩ mô đã qua xử lý.
- EIA: inventory, production, imports.
- GDELT: geopolitical/news tone, event count, volume.
- Extended market/regime: OVX, MOVE, gold, DGS3, DGS10, 10Y-3Y spread, oil momentum/volatility regime.

### 2.3. Cột bị loại và leakage control

Các nhóm cột bị loại khỏi model:

- Cột định danh/thời gian: `date`, `oil_return_fwd1_date`
- Target/future return: `oil_return_fwd1`
- Raw price level không stationarity hoặc có rủi ro same-day leakage: `oil_close`, `usd_close`, `sp500_close`, `vix_close`, `wti_fred`
- Redundant GDELT/stress features: `stress_tone`, `stress_goldstein`, `stress_volume`, `gdelt_volume`
- Near-zero variance: `gdelt_data_imputed`

Trong long-history branch còn loại thêm các cột có rủi ro release timing hoặc global preprocessing:

- `cpi_lag`
- `unemployment_lag`
- `fed_funds_rate_lag`
- `cpi_yoy`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `geopolitical_stress_index`
- `oil_volatility_7d`

Daily final dùng nhánh cũ compact, nhưng nguyên tắc leakage được giữ khi chọn feature:

- Dùng target date split, không random split.
- Feature selection chỉ dựa trên train period.
- Imputation nằm trong sklearn pipeline và fit trên train.
- Các feature regime mở rộng đều là lagged/EOD T hoặc rolling quá khứ.

### 2.4. Feature final của daily model

Daily final dùng 13 features:

| Feature | Nhóm | Mô tả |
|---|---|---|
| `vix_return_slog1p` | Market risk | Signed log transform của VIX return, đại diện risk sentiment |
| `oil_return` | Oil price | Lợi suất dầu hiện tại tại ngày `T` |
| `sp500_return_lag1` | Equity market | Lợi suất S&P 500 đã lag |
| `ret_mean_5` | Technical | Trung bình rolling 5 ngày của oil return |
| `momentum_10` | Technical | Momentum 10 ngày của dầu |
| `macd_signal` | Technical | MACD signal từ chuỗi giá dầu |
| `gdelt_tone_lag1` | GDELT | Tone tin tức/geopolitical đã lag |
| `gdelt_volume_lag1_log1p` | GDELT | Event/news volume đã lag và log transform |
| `yield_spread` | Macro/yield | Spread lãi suất/yield curve |
| `net_imports_change_pct_slog1p` | EIA/supply | Biến động nhập khẩu ròng, signed log transform |
| `ovx_high_regime_lag1` | Oil volatility regime | Cờ regime OVX cao, lagged |
| `yield_10y_3y_change_lag1_slog1p` | Yield curve | Biến động spread 10Y-3Y, lagged signed log |
| `oil_high_vol_regime_lag1` | Oil regime | Cờ oil realized volatility cao, lagged |

Ba feature extended được chọn bằng mutual information trong train period:

- `ovx_high_regime_lag1`
- `yield_10y_3y_change_lag1_slog1p`
- `oil_high_vol_regime_lag1`

### 2.5. Train/test split

Split chốt:

```text
Train: oil_return_fwd1_date < 2023-01-01
Test:  oil_return_fwd1_date >= 2023-01-01
```

Số dòng:

- Train rows: `2032`
- Test rows: `840`

Lý do dùng target date để split:

- Vì label là forward return.
- Nếu split theo `date` thường có thể làm một số dòng gần boundary dùng target ở tương lai vượt qua boundary.
- Split theo `oil_return_fwd1_date` rõ hơn cho bài toán T -> T+1.

### 2.6. Model và hyperparameters

Model:

```python
Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            n_jobs=4,
        )),
    ]
)
```

Fine-tune đã làm:

- Không dùng grid search lớn ở final step.
- Dựa trên branch compact tốt nhất trước đó.
- Chọn thêm top `3` extended features bằng mutual information trên train period.
- So với full extended feature set, compact + top regime features tốt hơn.

### 2.7. Daily final result

Artifact:

- Script: `scripts/train_daily_selected_model.py`
- Result folder: `ml/classification/results_daily_selected_model_v1`
- Model file: `ml/classification/results_daily_selected_model_v1/selected_daily_model.joblib`
- Metrics file: `ml/classification/results_daily_selected_model_v1/selected_daily_model_metrics.csv`
- Feature file: `ml/classification/results_daily_selected_model_v1/selected_daily_model_features.csv`

Fixed test result:

| Metric | Value |
|---|---:|
| Accuracy | `0.5548` |
| AUC | `0.5697` |
| F1 macro | `0.5442` |
| Predicted up-rate | `0.3500` |
| TP | `169` |
| FP | `125` |
| TN | `297` |
| FN | `249` |

Walk-forward robustness:

- Result folder: `ml/classification/results_daily_selected_plan_v1`
- Folds: `6`
- Mean accuracy: `0.5467`
- Median accuracy: `0.5509`
- Mean AUC: `0.5492`
- Mean F1 macro: `0.5303`
- Majority baseline: `0.5115`
- Previous-day persistence baseline: `0.4942`

Chốt daily:

> Daily final dùng compact LightGBM 13 features trên dataset 2015-2026. Long-history daily bị loại vì làm giảm accuracy và AUC.

## 3. Weekly final model

### 3.1. Target

Weekly target:

```text
target = weekly_oil_return_fwd1
target_class = 1 nếu weekly_oil_return_fwd1 > 0, ngược lại 0
```

Ý nghĩa:

- Mỗi dòng weekly tổng hợp thông tin tới cuối tuần hiện tại.
- Label là hướng lợi suất dầu của tuần kế tiếp.
- Đây là T -> T+1 week, không phải same-week classification.

### 3.2. Dataset

Dataset chốt cho weekly post-crawl:

- Daily source: `data/processed/dataset_final_noleak_ext_market_regime_2007_2026_no_acled.csv`
- Price source: `data/processed/dataset_step4_noleak_2007_2026_no_acled.csv`
- Weekly dataset output: `ml/classification/results_weekly_rolling_selected_model_v1/weekly_training_dataset.csv`
- Daily date range: `2007-08-02 -> 2026-03-19`
- Weekly rows total: `962`
- Final train window: `2018-03-20 -> 2026-03-20`
- Final train rows: `418`
- Train target up-rate: `0.5478`

Nguồn dữ liệu:

- Market/finance: dầu, USD, S&P 500, VIX, OVX, MOVE, gold.
- Macro/yield: yield spread, DGS3, DGS10, 10Y-3Y spread.
- Supply/EIA: inventory, inventory z-score, production, net imports.
- GDELT: event volume, tone, Goldstein, weekly aggregations.
- Calendar: một số biến thời gian nếu có trong candidate pool.

ACLED không dùng trong final weekly vì raw ACLED CSV không có trong checkout long-history và các ablation trước đó không cho thấy ACLED giúp daily tốt hơn.

### 3.3. Weekly aggregation

Weekly dataset được build từ daily no-leak dataset:

- Tạo weekly oil return.
- Tổng hợp weekly mean/std/sum/min/max/change tùy từng feature group.
- Tạo weekly target `weekly_oil_return_fwd1`.
- Gắn target date weekly để walk-forward theo năm.

Ví dụ nhóm feature weekly:

- Market: `weekly_oil_return`, `weekly_oil_return_std`, `weekly_usd_return`, `weekly_sp500_return`, `weekly_vix_return_slog1p_sum`
- Macro: `weekly_yield_spread_mean`, `weekly_dgs10_lag1_mean`, `weekly_dgs10_lag1_change`
- Supply: `weekly_inventory_zscore_mean`, `weekly_inventory_zscore_sum`
- GDELT: `weekly_gdelt_volume_lag1_log1p_sum`, `weekly_gdelt_tone_7d_change`

### 3.4. Feature selection

Final weekly dùng feature selection theo nhóm:

```text
Chọn top 3 mutual-information features trong mỗi group:
- market_finance
- macro_yield
- supply_eia
- gdelt
```

Feature selection chỉ fit trên train window của từng fold để tránh leakage.

Candidate chốt:

```text
roll8y__all_group_mi_top3_each__logreg_c1
```

Ý nghĩa candidate:

- `roll8y`: train bằng trailing 8-year window.
- `all_group_mi_top3_each`: chọn top 3 MI features từ mỗi nhóm feature.
- `logreg_c1`: Logistic Regression với `C=1.0`.

### 3.5. Feature final của weekly model

Weekly final dùng 12 features:

| Feature | Nhóm | Mô tả |
|---|---|---|
| `macd_cross` | Market/technical | Tín hiệu giao cắt MACD của dầu |
| `ma_50` | Market/technical | Moving average 50 ngày |
| `weekly_gold_return_slog1p_std` | Market/finance | Độ biến động weekly của gold return |
| `weekly_dgs10_lag1_change` | Macro/yield | Thay đổi weekly của DGS10 lagged |
| `dgs3_lag1` | Macro/yield | Lãi suất/yield 3 năm đã lag |
| `weekly_dgs10_lag1_mean` | Macro/yield | Trung bình weekly của DGS10 lagged |
| `weekly_inventory_zscore_mean` | Supply/EIA | Trung bình weekly inventory z-score |
| `weekly_inventory_zscore_sum` | Supply/EIA | Tổng weekly inventory z-score |
| `inventory_change_pct` | Supply/EIA | Phần trăm thay đổi inventory |
| `weekly_gdelt_volume_lag1_log1p_sum` | GDELT | Tổng weekly GDELT volume lagged log1p |
| `gdelt_volume_lag1_log1p` | GDELT | GDELT volume daily lagged log1p tại weekly row |
| `weekly_gdelt_tone_7d_change` | GDELT | Thay đổi weekly của GDELT 7-day tone |

Nhận xét:

- Feature set có đủ 4 nhóm: market/technical, macro/yield, supply/EIA, GDELT.
- Không dùng full 100+ feature matrix.
- Không dùng ACLED.
- Không dùng expanding window từ 2007 vì bị regime drift.

### 3.6. Train/evaluation split

Evaluation chính:

- Script: `scripts/evaluate_weekly_rolling_window.py`
- Result folder: `ml/classification/results_weekly_rolling_window_v2_tuned_saved`
- Walk-forward folds: `11`
- Fold starts: `2015-01-01 -> 2025-01-01`
- Mỗi fold validate một năm kế tiếp.
- Train window: trailing `8` years trước fold start.
- Minimum train rows: theo script rolling evaluator.

Final saved model:

- Script: `scripts/train_weekly_rolling_selected_model.py`
- Result folder: `ml/classification/results_weekly_rolling_selected_model_v1`
- Final train window: `2018-03-20 -> 2026-03-20`
- Train rows: `418`
- Weekly rows total: `962`

Lý do dùng rolling window:

- Expanding từ 2007 làm model học quá nhiều regime cũ.
- Oil market trước/sau 2014, 2020, 2022 có behavior khác nhau.
- Rolling 8 năm giữ đủ data nhưng bớt ảnh hưởng lịch sử quá xa.

### 3.7. Model và hyperparameters

Model:

```python
Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            C=1.0,
            l1_ratio=0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
        )),
    ]
)
```

Fine-tune đã làm:

- So sánh LightGBM và Logistic Regression.
- So sánh expanding và rolling windows.
- So sánh nhiều feature policies:
  - `domain_compact`
  - `domain_mi_top4_each`
  - `domain_mi_top6_each`
  - `all_group_mi_top3_each`
  - `all_group_mi_top5_each`
  - `gdelt_domain_only`
  - `market_domain_only`
  - `supply_domain_only`
- Thử nhiều rolling windows ở search trước đó, sau đó chốt `8` năm.
- Chốt `C=1.0` cho Logistic Regression trong selected branch.

### 3.8. Weekly final result

Walk-forward result:

| Metric | Value |
|---|---:|
| Folds | `11` |
| Mean accuracy | `0.5628` |
| Median accuracy | `0.5577` |
| Mean AUC | `0.5685` |
| Mean F1 macro | `0.4962` |
| Mean majority baseline | `0.5066` |
| Mean persistence baseline | `0.5172` |
| Mean train rows per fold | `413.7` |
| Mean features | `12` |

Per-fold result:

| Fold | Accuracy | AUC | F1 macro |
|---|---:|---:|---:|
| `2015-01-01` | `0.5577` | `0.6027` | `0.3580` |
| `2016-01-01` | `0.4151` | `0.6540` | `0.2933` |
| `2017-01-01` | `0.5385` | `0.6119` | `0.5024` |
| `2018-01-01` | `0.5577` | `0.4696` | `0.5048` |
| `2019-01-01` | `0.6346` | `0.5933` | `0.6103` |
| `2020-01-01` | `0.6154` | `0.6030` | `0.5486` |
| `2021-01-01` | `0.6415` | `0.5833` | `0.5490` |
| `2022-01-01` | `0.5577` | `0.4723` | `0.4501` |
| `2023-01-01` | `0.6538` | `0.6281` | `0.6406` |
| `2024-01-01` | `0.4423` | `0.4711` | `0.4404` |
| `2025-01-01` | `0.5769` | `0.5644` | `0.5607` |

Saved artifact:

- Model file: `ml/classification/results_weekly_rolling_selected_model_v1/weekly_rolling_selected_model.joblib`
- Feature file: `ml/classification/results_weekly_rolling_selected_model_v1/weekly_selected_features.csv`
- Summary JSON: `ml/classification/results_weekly_rolling_selected_model_v1/weekly_rolling_selected_model_summary.json`
- Training dataset: `ml/classification/results_weekly_rolling_selected_model_v1/weekly_training_dataset.csv`

In-sample final training result:

- Train accuracy: `0.5837`
- Train target up-rate: `0.5478`
- Train predicted up-rate: `0.5000`

Chốt weekly:

> Weekly final dùng rolling 8-year Logistic Regression với 12 MI-selected grouped features. Đây là setup tốt nhất sau khi có long-history data vì AUC cao nhất, kiểm định 11 folds, và tránh nhét toàn bộ lịch sử 2007-2026 vào expanding train.

## 4. So sánh model chốt với các nhánh bị loại

### 4.1. Daily

| Branch | Accuracy | AUC | F1 macro | Kết luận |
|---|---:|---:|---:|---|
| Daily compact fixed split | `0.5548` | `0.5697` | `0.5442` | Chốt daily |
| Daily compact walk-forward | `0.5467` | `0.5492` | `0.5303` | Robustness check tốt hơn baseline |
| Daily long fixed split | `0.5310` | `0.5332` | `0.4929` | Loại |
| Daily long walk-forward | `0.5241` | `0.5395` | `0.4917` | Loại |
| Daily regime expert best | `0.5467` | `0.5492` | `0.5303` | Không hơn global selected |

### 4.2. Weekly

| Branch | Folds | Accuracy | AUC | F1 macro | Kết luận |
|---|---:|---:|---:|---:|---|
| Old weekly compact LGBM | `6` | `0.5686` | `0.5429` | `0.5134` | Highest accuracy, nhưng data ngắn hơn |
| Weekly v2 GDELT/no-ACLED | `6` | `0.5620` | `0.5401` | `0.5210` | Tốt nhưng chưa robust bằng rolling |
| Weekly long expanding | `14` | `0.5345` | `0.5422` | `0.4627` | Loại do regime drift |
| Weekly long compact expanding | `14` | `0.5345` | `0.5274` | `0.4826` | Loại |
| Weekly rolling 8-year | `11` | `0.5628` | `0.5685` | `0.4962` | Chốt weekly post-crawl |

## 5. Cách trình bày trong báo cáo

Nên viết kết luận như sau:

> Với daily target, mô hình tốt nhất là compact LightGBM 13 features trên dữ liệu 2015-2026, đạt accuracy 0.5548 và AUC 0.5697 trên holdout từ 2023. Khi mở rộng dữ liệu GDELT về 2007, daily performance giảm xuống accuracy 0.5310, cho thấy long-history không tự động cải thiện bài toán daily do regime drift và label noise.

> Với weekly target, mô hình tốt nhất sau khi tận dụng long-history data là rolling-window Logistic Regression dùng trailing 8-year training window và top 3 mutual-information features theo từng nhóm market, macro, supply, GDELT. Mô hình đạt mean accuracy 0.5628 và mean AUC 0.5685 trên 11 yearly walk-forward folds. Dù old weekly compact có accuracy cao hơn nhẹ, rolling weekly là phương pháp final hợp lý hơn cho long-history vì kiểm soát regime drift và có AUC tốt hơn.

## 6. Lệnh tái chạy chính

Daily selected model:

```powershell
py -3.13 scripts\train_daily_selected_model.py
```

Daily walk-forward check:

```powershell
py -3.13 scripts\evaluate_daily_selected_plan.py
```

Build long-history no-ACLED dataset:

```powershell
py -3.13 scripts\build_daily_long_no_acled_pipeline.py
```

Weekly rolling-window evaluation:

```powershell
py -3.13 scripts\evaluate_weekly_rolling_window.py --out-dir ml\classification\results_weekly_rolling_window_v2_tuned_saved --rolling-years 8 --logreg-c-values 1.0 --save-prediction-candidate roll8y__all_group_mi_top3_each__logreg_c1
```

Train selected weekly model artifact:

```powershell
py -3.13 scripts\train_weekly_rolling_selected_model.py
```

## 7. Final artifacts

Daily:

- `ml/classification/results_daily_selected_model_v1/selected_daily_model.joblib`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_metrics.csv`
- `ml/classification/results_daily_selected_model_v1/selected_daily_model_features.csv`

Weekly:

- `ml/classification/results_weekly_rolling_selected_model_v1/weekly_rolling_selected_model.joblib`
- `ml/classification/results_weekly_rolling_selected_model_v1/weekly_selected_features.csv`
- `ml/classification/results_weekly_rolling_selected_model_v1/weekly_rolling_selected_model_summary.json`

Report comparison:

- `docs/old_vs_new_data_result_comparison.csv`
- `docs/OLD_VS_NEW_DATA_RESULTS_2026-05-14.md`
- `docs/MODEL_EXPERIMENT_HISTORY_DAILY_WEEKLY_LONG_DATA_2026-05-14.md`
- `docs/FINAL_DAILY_WEEKLY_MODEL_METHOD_2026-05-14.md`

