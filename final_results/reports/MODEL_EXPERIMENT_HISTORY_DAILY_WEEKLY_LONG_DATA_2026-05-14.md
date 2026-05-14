# Báo cáo toàn bộ thử nghiệm Daily, Weekly và Long-History Data

Ngày tổng kết: 2026-05-14

Repo: `E:\Project\cs313-oil-prediction`

## 1. Mục tiêu bài toán

Bài toán chính là dự đoán hướng biến động giá dầu thô, dưới dạng phân loại nhị phân:

- Daily target: dự đoán giá dầu ngày kế tiếp tăng hay giảm.
- Weekly target: dự đoán lợi suất dầu tuần kế tiếp tăng hay giảm.

Hai target đều được xây theo nguyên tắc forward return:

- Daily: `oil_return_fwd1 > 0`
- Weekly: `weekly_oil_return_fwd1 > 0`

Trong toàn bộ thử nghiệm, không dùng random split. Các mô hình được đánh giá bằng split theo thời gian, fixed chronological holdout, expanding walk-forward hoặc rolling-window walk-forward để tránh leakage thời gian.

## 2. Dữ liệu đã dùng

### 2.1. Dữ liệu cũ 2015-2026

Nhánh cũ dùng dataset đã xử lý sẵn:

- Daily source: `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- Date range: `2015-01-07 -> 2026-03-19`
- Target date range: `2015-01-08 -> 2026-03-20`
- Rows: `2922`
- Columns: `57`
- Target up-rate: khoảng `0.5089`

Nguồn dữ liệu chính:

- Market: giá dầu, USD index, S&P 500, VIX.
- FRED/macro: lãi suất, yield spread, một số biến vĩ mô.
- EIA: tồn kho, sản lượng, nhập khẩu ròng.
- GDELT: tone, Goldstein, event volume liên quan geopolitical/conflict/news.
- Extended market regime: OVX, MOVE, gold, yield 3Y/10Y, oil momentum/volatility regime.

### 2.2. Dữ liệu GDELT long-history 2007-2026

File GDELT mới:

- Old file: `data/raw/gdelt_data.csv`
- New file: `data/raw/gdelt_data_2007_2026.csv`

So sánh coverage:

| File | Date range | Rows | Non-empty GDELT days | Missing days | Cell missing |
|---|---|---:|---:|---:|---:|
| `data/raw/gdelt_data.csv` | `2015-01-01 -> 2026-03-20` | `2927` | `2913` | `14` | `0.253%` |
| `data/raw/gdelt_data_2007_2026.csv` | `2007-01-01 -> 2026-03-20` | `5015` | `4999` | `16` | `0.164%` |

Kết luận về crawl thêm data:

- Crawl thêm data có hiệu quả về coverage.
- GDELT mới dài hơn và missing ít hơn theo tỷ lệ.
- Nhưng thêm lịch sử dài không tự động làm model tốt hơn, vì dầu có regime drift rất mạnh.

### 2.3. Long-history no-ACLED dataset

Do raw ACLED CSV không có trong checkout hiện tại, nhánh long-history được build theo hướng no-ACLED:

- Script: `scripts/build_daily_long_no_acled_pipeline.py`
- Final dataset: `data/processed/dataset_final_noleak_ext_market_regime_2007_2026_no_acled.csv`
- Date range: `2007-08-02 -> 2026-03-19`
- Target date range: `2007-08-03 -> 2026-03-20`
- Rows: `4861`
- Columns: `53`
- Missing values: `0`
- Target up-rate: `0.4921`

Input raw files:

- `data/raw/market_data_2007_2026.csv`
- `data/raw/fred_data_2007_2026.csv`
- `data/raw/eia_data_2007_2026.csv`
- `data/raw/gdelt_data_2007_2026.csv`
- `data/raw/market_ext_regime_2007_2026.csv`
- `data/raw/fred_ext_regime_2007_2026.csv`

Các cột bị loại vì rủi ro leakage:

- `cpi_lag`
- `unemployment_lag`
- `fed_funds_rate_lag`
- `cpi_yoy`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `stress_tone`
- `stress_volume`
- `stress_goldstein`
- `geopolitical_stress_index`
- `oil_volatility_7d`

Lý do chính:

- Một số macro monthly series có rủi ro xuất hiện quá sớm so với ngày công bố thật.
- Một số stress feature được scale bằng phân phối train+validation cũ.
- Một số biến rolling/winsorized có nguy cơ dùng thống kê toàn chuỗi.

## 3. Feature engineering đã thử

### 3.1. Base daily feature pipeline

Pipeline preprocessing chính:

- Step 2: clean từng nguồn dữ liệu.
- Step 3: reindex theo business day và forward-fill có kiểm soát.
- Step 4: tạo return, rolling, lag, macro, EIA, GDELT, calendar và target forward.
- Step 4b: drop leakage columns.
- Step 5b: transform/log/scale một số biến để model-ready.

Nhóm feature chính:

- Market/price: oil return, USD return, S&P 500 return, VIX return.
- Technical: rolling mean, momentum, MACD, moving average, volatility.
- Macro/yield: yield spread, DGS3, DGS10, 10Y-3Y spread/change.
- EIA/supply: inventory change, inventory z-score, production/import changes.
- GDELT: tone, tone rolling, Goldstein score, event volume.
- Regime/cross-asset: OVX high regime, oil high-vol regime, MOVE, gold, risk-off flags.

### 3.2. Extended market/regime features

Extended features được thêm từ market_ext và FRED ext:

- OVX: `ovx_return_slog1p`, `ovx_return_lag1_slog1p`, `ovx_level_lag1_log1p`, `ovx_high_regime_lag1`, `ovx_extreme_regime_lag1`
- MOVE: `move_return_slog1p`, `move_return_lag1_slog1p`, `move_level_lag1_log1p`, `move_high_regime_lag1`, `move_extreme_regime_lag1`
- Gold: `gold_return_slog1p`, `gold_return_lag1_slog1p`, `gold_level_lag1_log1p`
- Yield curve: `dgs3_lag1`, `dgs3_change_lag1_slog1p`, `dgs10_lag1`, `dgs10_change_lag1_slog1p`, `yield_10y_3y_lag1`, `yield_10y_3y_change_lag1_slog1p`
- Oil regime: `oil_momentum_5d_lag1`, `oil_momentum_20d_lag1`, `oil_realized_vol_5d_lag1`, `oil_realized_vol_20d_lag1`, `oil_uptrend_5d_lag1`, `oil_uptrend_20d_lag1`, `oil_high_vol_regime_lag1`
- Cross-asset: `risk_off_cross_asset_lag1`

Kết quả thực tế: thêm full feature matrix không tốt bằng chọn ít feature có kiểm soát. Các branch tốt nhất đều dùng feature set nhỏ.

## 4. Daily experiments

### 4.1. Daily selected compact model, fixed test split

Đây là daily model tốt nhất hiện tại.

- Script: `scripts/train_daily_selected_model.py`
- Dataset: `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- Result folder: `ml/classification/results_daily_selected_model_v1`
- Target: `oil_return_fwd1 > 0`
- Split: target date `< 2023-01-01` train, `>= 2023-01-01` test
- Train rows: `2032`
- Test rows: `840`
- Model: `LGBMClassifier`
- Features: `13`
- Test accuracy: `0.5548`
- Test AUC: `0.5697`
- Test F1 macro: `0.5442`
- Confusion matrix: TP=`169`, FP=`125`, TN=`297`, FN=`249`

Selected features:

- `vix_return_slog1p`
- `oil_return`
- `sp500_return_lag1`
- `ret_mean_5`
- `momentum_10`
- `macd_signal`
- `gdelt_tone_lag1`
- `gdelt_volume_lag1_log1p`
- `yield_spread`
- `net_imports_change_pct_slog1p`
- `ovx_high_regime_lag1`
- `yield_10y_3y_change_lag1_slog1p`
- `oil_high_vol_regime_lag1`

### 4.2. Daily selected compact, yearly walk-forward

- Script: `scripts/evaluate_daily_selected_plan.py`
- Result folder: `ml/classification/results_daily_selected_plan_v1`
- Folds: `6`
- Fold range: `2020-01-01 -> 2025-01-01`
- Mean accuracy: `0.5467`
- Median accuracy: `0.5509`
- Mean AUC: `0.5492`
- Mean F1 macro: `0.5303`
- Majority baseline: `0.5115`
- Previous-day persistence baseline: `0.4942`

Kết luận:

- Model tốt hơn baseline nhưng chưa đạt mức `0.60`.
- Daily target vẫn rất nhiễu.
- Đây vẫn là daily branch tốt nhất.

### 4.3. Daily selective prediction

Thử dùng confidence threshold để chỉ dự đoán khi model tự tin.

Kết quả trên daily selected compact:

- Threshold tốt nhất nếu yêu cầu coverage `>= 70%`: `0.56`
- Mean coverage: `0.799`
- Mean accuracy: `0.5584`

Kết luận:

- Có cải thiện nhẹ so với full coverage.
- Nhưng chưa đủ để claim model lên `0.60`.
- Đây chỉ là diagnostic, không phải final model chính.

### 4.4. Daily regime slice và regime expert

Quan sát regime slice:

- `ovx_high_regime_lag1=high`: mean accuracy `0.5805`, mean AUC `0.5725`
- `oil_high_vol_regime_lag1=high`: mean accuracy `0.5531`, mean AUC `0.5463`

Sau đó thử regime-aware candidates:

- `global_selected`
- `global_oil_high_interactions`
- `global_ovx_interactions`
- `global_dual_regime_interactions`
- `expert_by_oil_high_vol_regime_lag1`
- `expert_by_ovx_high_regime_lag1`

Kết quả:

| Candidate | Folds | Mean Acc | Mean AUC | Mean F1 |
|---|---:|---:|---:|---:|
| `global_selected` | `6` | `0.5467` | `0.5492` | `0.5303` |
| `global_oil_high_interactions` | `6` | `0.5397` | `0.5387` | `0.5258` |
| `global_dual_regime_interactions` | `6` | `0.5351` | `0.5339` | `0.5206` |
| `global_ovx_interactions` | `6` | `0.5269` | `0.5364` | `0.5106` |
| `expert_by_oil_high_vol_regime_lag1` | `6` | `0.5256` | `0.5225` | `0.5137` |
| `expert_by_ovx_high_regime_lag1` | `6` | `0.5192` | `0.5246` | `0.5073` |

Kết luận:

- Regime slice có tín hiệu, nhưng model expert/interactions không thắng global selected.
- Không chốt nhánh regime expert.

### 4.5. Daily ACLED/no-ACLED ablation

Thử xem ACLED-like features có giúp daily không:

| Candidate | Folds | Mean Acc | Mean AUC | Mean F1 |
|---|---:|---:|---:|---:|
| `selected_no_acled` | `6` | `0.5467` | `0.5492` | `0.5303` |
| `selected_plus_acled` | `6` | `0.5326` | `0.5397` | `0.5188` |
| `mi_top20_no_acled` | `6` | `0.5179` | `0.5311` | `0.4868` |
| `mi_top20_with_acled` | `6` | `0.5115` | `0.5266` | `0.4985` |
| `mi_top13_no_acled` | `6` | `0.5089` | `0.5256` | `0.4957` |
| `mi_top13_with_acled` | `6` | `0.5019` | `0.5294` | `0.4980` |

Kết luận:

- Daily selected không cần ACLED để đạt kết quả tốt nhất trong nhánh này.
- Thêm ACLED-like features làm kém hơn.

### 4.6. Daily long-history no-ACLED, fixed test split

Sau khi sửa GDELT và build long dataset, thử lại daily model trên data 2007-2026:

- Script: `scripts/train_daily_selected_model.py` với dataset long no-ACLED
- Dataset: `data/processed/dataset_final_noleak_ext_market_regime_2007_2026_no_acled.csv`
- Result folder: `ml/classification/results_daily_long_no_acled_selected_model_v1`
- Split: target date `< 2023-01-01` train, `>= 2023-01-01` test
- Train rows: `3971`
- Test rows: `840`
- Model: `LGBMClassifier`
- Features: `13`
- Test accuracy: `0.5310`
- Test AUC: `0.5332`
- Test F1 macro: `0.4929`
- Confusion matrix: TP=`108`, FP=`83`, TN=`338`, FN=`311`

Selected long-history features:

- `vix_return_slog1p`
- `oil_return`
- `sp500_return_lag1`
- `ret_mean_5`
- `momentum_10`
- `macd_signal`
- `gdelt_tone_lag1`
- `gdelt_volume_lag1_log1p`
- `yield_spread`
- `net_imports_change_pct_slog1p`
- `oil_uptrend_20d_lag1`
- `move_level_lag1_log1p`
- `dgs3_lag1`

Kết luận:

- Daily long-history kém hơn daily cũ.
- Việc thêm 2007-2014 làm model học thêm các regime không còn ổn định cho giai đoạn test 2023-2026.

### 4.7. Daily long-history walk-forward và MI ablation

Walk-forward long-history:

- Result folder: `ml/classification/results_daily_long_no_acled_selected_plan_v1`
- Folds: `11`
- Mean accuracy: `0.5241`
- Median accuracy: `0.5211`
- Mean AUC: `0.5395`
- Mean F1 macro: `0.4917`
- Majority baseline: `0.4937`
- Previous-day persistence baseline: `0.4878`

MI ablation trên long-history:

| Candidate | Folds | Mean Acc | Mean AUC | Mean F1 |
|---|---:|---:|---:|---:|
| `selected_no_acled` | `11` | `0.5241` | `0.5395` | `0.4917` |
| `selected_plus_acled` | `11` | `0.5241` | `0.5395` | `0.4917` |
| `mi_top13_no_acled` | `11` | `0.5139` | `0.5138` | `0.4992` |
| `mi_top20_no_acled` | `11` | `0.5010` | `0.5110` | `0.4764` |

Kết luận:

- MI reselection không cứu được daily long-history.
- Daily final vẫn chọn nhánh cũ compact 2015-2026.

## 5. Weekly experiments

Weekly được thử vì daily T -> T+1 quá nhiễu. Weekly target giảm nhiễu hơn và cho kết quả ổn định hơn trong một số setup.

### 5.1. Weekly v2 no-ACLED GDELT

- Result folder: `ml/classification/results_weekly_multifactor_no_acled_gdelt_v2`
- Data: 2015-2026
- Folds: `6`
- Target: `weekly_oil_return_fwd1 > 0`

Top candidates:

| Candidate | Folds | Mean Acc | Median Acc | Mean AUC | Mean F1 |
|---|---:|---:|---:|---:|---:|
| `all_multifactor_no_acled` | `6` | `0.5620` | `0.5577` | `0.5326` | `0.5167` |
| `gdelt_only_threshold_f1_macro` | `6` | `0.5620` | `0.5481` | `0.5401` | `0.5210` |
| `gdelt_only_threshold_accuracy` | `6` | `0.5556` | `0.5385` | `0.5401` | `0.5023` |
| `balanced_group_mi_top3_each` | `6` | `0.5493` | `0.5288` | `0.5600` | `0.5005` |

Kết luận:

- Weekly bắt đầu có tín hiệu tốt hơn daily.
- GDELT có ích trong weekly hơn daily.
- Nhưng full multi-factor vẫn có nguy cơ noisy vì weekly sample ít.

### 5.2. Weekly compact v3, old 2015-2026 data

Đây là weekly result có mean accuracy cao nhất trong toàn bộ thử nghiệm.

- Script: `scripts/evaluate_weekly_multifactor_compact_v3.py`
- Result folder: `ml/classification/results_weekly_multifactor_compact_v3`
- Daily source: `data/processed/dataset_final_noleak_ext_market_regime_v1.csv`
- Weekly rows: `574`
- Week date range: `2015-03-20 -> 2026-03-13`
- Target week range: `2015-03-27 -> 2026-03-20`
- Target up-rate: `0.5418`
- ACLED dropped, GDELT kept.

Feature groups available:

- `market_finance`: `82`
- `macro_yield`: `18`
- `supply_eia`: `16`
- `gdelt`: `24`
- `calendar`: `4`
- `domain_compact`: `50`

Best candidate:

- Candidate: `domain_mi_top4_each__lgbm`
- Folds: `6`
- Mean accuracy: `0.5686`
- Median accuracy: `0.5673`
- Mean AUC: `0.5429`
- Mean F1 macro: `0.5134`
- Majority baseline: `0.5365`
- Persistence baseline: `0.5205`

Kết luận:

- Nếu chỉ tối ưu mean accuracy, đây là weekly result cao nhất.
- Nhưng nó chỉ dùng 6 folds trong giai đoạn 2020-2025 và data 2015-2026.
- AUC không cao bằng weekly rolling sau khi thêm long-history.

### 5.3. Weekly long-history expanding

Sau khi sửa GDELT và build long-history no-ACLED dataset, thử weekly expanding từ 2007:

- Result folder: `ml/classification/results_weekly_long_multifactor_no_acled_gdelt_v2_fixed_gdelt`
- Data: `2007-2026`
- Evaluation: expanding weekly walk-forward
- Folds: `14`
- Best candidate: `stable_mi_top5_each__logreg`
- Mean accuracy: `0.5345`
- Median accuracy: `0.5577`
- Mean AUC: `0.5422`
- Mean F1 macro: `0.4627`
- Majority baseline: `0.5148`
- Persistence baseline: `0.5053`

Kết luận:

- Expanding training từ 2007 làm performance giảm.
- Long-history có regime drift, không nên train kiểu cứ cộng hết dữ liệu cũ vào.

### 5.4. Weekly long-history compact expanding

Thử compact feature set trên long-history:

- Result folder: `ml/classification/results_weekly_long_no_acled_compact_v3_fixed_gdelt`
- Best candidate: `stable_mi_top5_each__logreg`
- Folds: `14`
- Mean accuracy: `0.5345`
- Median accuracy: `0.5385`
- Mean AUC: `0.5274`
- Mean F1 macro: `0.4826`

Kết luận:

- Compact long-history expanding vẫn không tốt.
- Vấn đề chính không phải chỉ do quá nhiều feature, mà do training window quá dài qua nhiều regime.

### 5.5. Weekly rolling-window long-history

Sau khi expanding long-history không tốt, thử rolling-window để chỉ train trên lịch sử gần hơn.

- Script: `scripts/evaluate_weekly_rolling_window.py`
- Result folder: `ml/classification/results_weekly_rolling_window_v2_tuned_saved`
- Dataset: `data/processed/dataset_final_noleak_ext_market_regime_2007_2026_no_acled.csv`
- Weekly rows: `962`
- Target: `weekly_oil_return_fwd1 > 0`
- Fold range: `2015-01-01 -> 2025-01-01`
- Rolling window: trailing `8` years
- Feature selection: top `3` mutual-information features per group
- Model: `LogisticRegression(C=1.0, class_weight='balanced')`
- Candidate: `roll8y__all_group_mi_top3_each__logreg_c1`

Kết quả:

- Folds: `11`
- Mean accuracy: `0.5628`
- Median accuracy: `0.5577`
- Mean AUC: `0.5685`
- Mean F1 macro: `0.4962`
- Majority baseline: `0.5066`
- Persistence baseline: `0.5172`

Per-fold:

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

Kết luận:

- Rolling-window giải quyết regime drift tốt hơn expanding.
- Accuracy chưa vượt old weekly compact `0.5686`, nhưng AUC cao hơn rõ `0.5685` so với `0.5429`.
- Đây là weekly branch tốt nhất sau khi dùng long-history data một cách có kiểm soát.

## 6. So sánh old vs new data

| Run | Horizon | Data scope | Candidate | Folds | Accuracy | AUC | F1 macro |
|---|---|---|---|---:|---:|---:|---:|
| Old daily selected compact | Daily | 2015-2026 fixed split | `selected` | - | `0.5548` | `0.5697` | `0.5442` |
| New daily long selected | Daily | 2007-2026 fixed split | `selected` | - | `0.5310` | `0.5332` | `0.4929` |
| Old daily selected WF | Daily | 2015-2026 yearly WF | `selected_lgbm` | `6` | `0.5467` | `0.5492` | `0.5303` |
| New daily long selected WF | Daily | 2007-2026 yearly WF | `selected_lgbm` | `11` | `0.5241` | `0.5395` | `0.4917` |
| Old weekly compact best | Weekly | 2015-2026 compact WF | `domain_mi_top4_each__lgbm` | `6` | `0.5686` | `0.5429` | `0.5134` |
| New weekly long expanding | Weekly | 2007-2026 expanding WF | `stable_mi_top5_each__logreg` | `14` | `0.5345` | `0.5422` | `0.4627` |
| New weekly rolling best | Weekly | 2007-2026 rolling 8y WF | `roll8y__all_group_mi_top3_each__logreg_c1` | `11` | `0.5628` | `0.5685` | `0.4962` |

## 7. Kết luận tổng hợp

### Daily

Daily long-history không hiệu quả. Chốt daily vẫn là model compact trên dataset cũ 2015-2026:

- Final daily folder: `ml/classification/results_daily_selected_model_v1`
- Accuracy: `0.5548`
- AUC: `0.5697`
- F1 macro: `0.5442`

### Weekly

Weekly có hai kết luận cần tách rõ:

- Nếu chỉ nhìn highest mean accuracy: old weekly compact tốt nhất với accuracy `0.5686`.
- Nếu chọn branch cuối cùng sau khi crawl thêm long-history và kiểm soát regime drift: weekly rolling 8-year là branch đáng dùng nhất, với accuracy `0.5628` và AUC `0.5685`.

Chốt cho báo cáo final:

- Daily final: old compact daily LGBM.
- Weekly final: rolling 8-year Logistic Regression trên long-history no-ACLED, vì đây là nhánh tận dụng data mới hợp lý nhất và có AUC tốt nhất.

Không nên claim rằng crawl thêm data làm accuracy tốt hơn. Claim đúng hơn là:

> Crawl thêm GDELT giúp dữ liệu đầy đủ và cho phép kiểm định long-history. Tuy nhiên, với oil direction prediction, more data is not automatically better. Long-history chỉ hữu ích khi dùng rolling-window để giảm regime drift.

