# Old vs New Data Results - 2026-05-14

## Kết luận ngắn

Thêm/crawl GDELT 2007-2026 **có hiệu quả về mặt dữ liệu**, nhưng **không tạo cải thiện trực tiếp về accuracy model** nếu chỉ nhét thêm lịch sử vào cùng một công thức train.

Kết luận cuối cùng:

- Daily: **không cải thiện**. Bản cũ 2015-2026 vẫn tốt hơn bản long-history mới.
- Weekly expanding từ 2007: **không cải thiện**. Lịch sử quá xa làm nhiễu do regime drift.
- Weekly rolling 8 năm: **nhánh mới tốt nhất sau crawl thêm data**, nhưng accuracy vẫn **hơi thấp hơn best weekly cũ** nếu so đúng mean accuracy.
- Best metric cũ theo weekly accuracy: `0.5686`.
- Best metric mới sau crawl thêm data: `0.5628` accuracy, nhưng AUC tốt hơn (`0.5685`) và được kiểm trên 11 folds thay vì 6 folds.

Nói thẳng: nếu KPI chính là **mean accuracy**, crawl thêm data **chưa chứng minh là tốt hơn**. Nếu KPI là **robustness/AUC với lịch sử dài hơn**, weekly rolling là branch đáng giữ.

## Data Coverage

| File | Date range | Rows | Non-empty GDELT days | Missing days | Cell missing |
|---|---|---:|---:|---:|---:|
| `data/raw/gdelt_data.csv` | 2015-01-01 -> 2026-03-20 | 2927 | 2913 | 14 | 0.253% |
| `data/raw/gdelt_data_2007_2026.csv` | 2007-01-01 -> 2026-03-20 | 5015 | 4999 | 16 | 0.164% |

Dataset model-ready:

| Dataset | Date range | Rows | Cols | Missing | Target up-rate |
|---|---|---:|---:|---:|---:|
| Old daily processed ext dataset | 2015-01-07 -> 2026-03-19 | 2922 | 57 | 0 | 0.5089 |
| New long no-ACLED ext dataset | 2007-08-02 -> 2026-03-19 | 4861 | 53 | 0 | 0.4921 |

Data đã sạch hơn và dài hơn thật. Vấn đề là model không tự động tốt hơn vì phân phối 2007-2014 khác giai đoạn gần đây.

## Daily Comparison

| Run | Data | Evaluation | Accuracy | AUC | F1 macro | Baseline note |
|---|---|---|---:|---:|---:|---|
| Old daily selected compact | 2015-2026 | Fixed test target date >= 2023-01-01 | 0.5548 | 0.5697 | 0.5442 | Current best daily compact branch |
| New daily long selected | 2007-2026 no-ACLED | Fixed test target date >= 2023-01-01 | 0.5310 | 0.5332 | 0.4929 | Worse after adding long history |
| Old daily selected walk-forward | 2015-2026 | 6 yearly folds | 0.5467 | 0.5492 | 0.5303 | Majority 0.5115, persistence 0.4942 |
| New daily long selected walk-forward | 2007-2026 no-ACLED | 11 yearly folds | 0.5241 | 0.5395 | 0.4917 | Majority 0.4937, persistence 0.4878 |

Daily verdict: **không nên dùng long-history daily branch làm final**. Best daily vẫn là old compact branch:

- Folder: `ml/classification/results_daily_selected_model_v1`
- Accuracy: `0.5548`
- AUC: `0.5697`

## Weekly Comparison

| Run | Data | Evaluation | Best candidate | Folds | Accuracy | Median Acc | AUC | F1 macro | Baselines |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Old weekly compact best | 2015-2026 | Weekly walk-forward | `domain_mi_top4_each__lgbm` | 6 | 0.5686 | 0.5673 | 0.5429 | 0.5134 | Majority 0.5365, persistence 0.5205 |
| Old weekly v2 best | 2015-2026 | Weekly walk-forward | `gdelt_only_threshold_f1_macro` | 6 | 0.5620 | 0.5481 | 0.5401 | 0.5210 | Majority 0.5365, persistence 0.5205 |
| New weekly long expanding best | 2007-2026 | Expanding weekly WF | `stable_mi_top5_each__logreg` | 14 | 0.5345 | 0.5577 | 0.5422 | 0.4627 | Majority 0.5148, persistence 0.5053 |
| New weekly long compact best | 2007-2026 | Expanding weekly WF | `stable_mi_top5_each__logreg` | 14 | 0.5345 | 0.5385 | 0.5274 | 0.4826 | Majority 0.5148, persistence 0.5053 |
| New weekly rolling best | 2007-2026 | Rolling 8-year weekly WF | `roll8y__all_group_mi_top3_each__logreg_c1` | 11 | 0.5628 | 0.5577 | 0.5685 | 0.4962 | Majority 0.5066, persistence 0.5172 |

Weekly verdict:

- Nếu chọn theo **highest mean accuracy**, old weekly compact vẫn thắng: `0.5686 > 0.5628`.
- Nếu chọn theo **AUC + kiểm chứng dài hơn**, new weekly rolling đáng giữ: `AUC 0.5685`, 11 folds, dùng long data có kiểm soát drift.
- Nếu chọn theo **final practical branch sau khi crawl thêm data**, chọn weekly rolling 8-year, không chọn expanding from 2007.

## Why More Data Did Not Automatically Help

Thêm data giúp giảm missing và mở rộng lịch sử, nhưng model dầu daily/weekly bị các vấn đề sau:

1. **Regime drift**: 2007-2014 không giống 2020-2026. Expanding training từ quá xa làm model học quan hệ cũ.
2. **Daily target quá nhiễu**: daily T -> T+1 gần random hơn weekly; thêm dữ liệu không sửa được label noise.
3. **Feature relation không ổn định**: GDELT/conflict/market volatility có tác dụng khác nhau theo từng giai đoạn.
4. **No-ACLED long branch khác setup cũ**: new long dataset không có ACLED raw file trong checkout, nên so daily cũ vs mới không phải chỉ khác GDELT, mà còn khác data scope.

Vì vậy, hiệu quả thật sự của crawl thêm data là:

- Tốt cho data coverage.
- Tốt để test long-history/rolling-window.
- Không tốt nếu dùng expanding train toàn bộ 2007-2026.
- Chưa cải thiện best accuracy so với old weekly compact.

## Final Result To Use

### Final daily result

Use old compact daily:

- Folder: `ml/classification/results_daily_selected_model_v1`
- Model: `LGBMClassifier`
- Features: 13
- Accuracy: `0.5548`
- AUC: `0.5697`
- F1 macro: `0.5442`

Daily long-history is rejected for now.

### Final weekly result

Use new weekly rolling as the final post-crawl branch:

- Search folder: `ml/classification/results_weekly_rolling_window_v2_tuned_saved`
- Saved model folder: `ml/classification/results_weekly_rolling_selected_model_v1`
- Model file: `weekly_rolling_selected_model.joblib`
- Candidate: `roll8y__all_group_mi_top3_each__logreg_c1`
- Model: `LogisticRegression(C=1.0, class_weight='balanced')`
- Rolling window: trailing 8 years
- Feature policy: top 3 MI features per group from market, macro/yield, supply, and GDELT
- Accuracy: `0.5628`
- AUC: `0.5685`
- F1 macro: `0.4962`

This is not the highest historical weekly accuracy, but it is the best result after adding long data and controlling regime drift.

## Final Recommendation

For the project report:

1. Report **daily compact** as the final daily model.
2. Report **weekly rolling 8-year** as the final weekly experiment after long-data crawl.
3. Be honest that adding GDELT history improved data completeness but did **not** beat the old best accuracy.
4. Say the best lesson is not "more data is always better"; it is "long data needs rolling-window training because oil regimes drift."

## Files Created

- `docs/old_vs_new_data_result_comparison.csv`
- `docs/OLD_VS_NEW_DATA_RESULTS_2026-05-14.md`
- `docs/WEEKLY_ROLLING_IMPROVEMENT_2026-05-14.md`
- `ml/classification/results_weekly_rolling_window_v2_tuned_saved/`
- `ml/classification/results_weekly_rolling_selected_model_v1/`
