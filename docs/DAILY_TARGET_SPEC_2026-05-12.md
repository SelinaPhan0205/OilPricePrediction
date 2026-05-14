# Daily Target Specification - 2026-05-12

## Decision

Giữ bài toán chính là **daily next-trading-day direction**.

Weekly chỉ là nhánh thử nghiệm phụ nếu daily không còn cải thiện được sau khi đã kiểm tra feature/regime/validation. Không chuyển pipeline chính sang weekly ở thời điểm này.

## Target Chính

Mỗi dòng tại ngày giao dịch `T` dùng feature đã biết tại `T` để dự đoán hướng return của ngày giao dịch kế tiếp `T+1`.

Target hiện tại:

```text
target_return = oil_return_fwd1 = oil_return.shift(-1)
target_date   = oil_return_fwd1_date = date.shift(-1)
target_class  = 1 nếu oil_return_fwd1 > 0, ngược lại 0
```

Diễn giải:

- `date`: ngày feature được quan sát.
- `oil_return_fwd1_date`: ngày mà target xảy ra.
- `oil_return_fwd1`: daily return của dầu tại ngày target.
- `target_class=1`: giá dầu tăng ở ngày giao dịch kế tiếp.
- `target_class=0`: giá dầu không tăng hoặc giảm ở ngày giao dịch kế tiếp.

## Artifact Đang Dùng

Dataset daily chính cho nhánh no-leak:

- `data/processed/dataset_final_noleak_processed.csv`
- số dòng: `2922`
- số cột: `30`
- target date range: `2015-01-08 -> 2026-03-20`
- class balance:
  - `UP=1487`
  - `DOWN=1435`
  - `UP rate=0.5089`

Dataset step4 gốc cũng khớp target:

- `data/processed/dataset_step4_transformed.csv`
- số dòng: `2922`
- số cột: `56`
- target date range: `2015-01-08 -> 2026-03-20`
- class balance:
  - `UP=1487`
  - `DOWN=1435`
  - `UP rate=0.5089`

## Config Hiện Tại

`ml/config.py` đang trỏ đúng daily target:

```python
TARGET = "oil_return_fwd1"
TARGET_DATE_COL = "oil_return_fwd1_date"
```

Split cũng dùng `oil_return_fwd1_date` khi có cột này, nên train/test được chia theo ngày target thay vì ngày feature:

- train/validation trước `2023-01-01`
- final test từ `2023-01-01` trở đi

## Quy Tắc Khi Train Tiếp

1. Không đổi target sang weekly trong nhánh chính.
2. Không dùng `oil_return_fwd1` hoặc `oil_return_fwd1_date` làm feature.
3. Nếu dùng feature cùng ngày `T` như `oil_return`, phải hiểu đây là setup end-of-day `T -> next trading day T+1`.
4. Các rolling/regime feature nên dùng lag hoặc past-only window để tránh future leakage.
5. Nếu thử weekly sau này, tạo output và result folder riêng, không ghi đè daily artifacts.

## Lệnh Train Daily Tiếp Theo

Ví dụ chạy nhánh daily extended market/regime đang có:

```powershell
$env:CLASSIFICATION_DATA_PATH = "data\processed\dataset_final_noleak_ext_market_regime_v1.csv"
$env:CLASSIFICATION_OUT_DIR = "ml\classification\results_ext_market_regime_v1"
& "C:\Users\SelinaPhan\AppData\Local\Programs\Python\Python313\python.exe" ml\classification\step1_train_baseline.py
& "C:\Users\SelinaPhan\AppData\Local\Programs\Python\Python313\python.exe" ml\classification\step2_finetune_ensemble.py
& "C:\Users\SelinaPhan\AppData\Local\Programs\Python\Python313\python.exe" ml\classification\step5_smart_selection.py
```

Nếu muốn quay về baseline no-leak daily:

```powershell
$env:CLASSIFICATION_DATA_PATH = "data\processed\dataset_final_noleak_processed.csv"
$env:CLASSIFICATION_OUT_DIR = "ml\classification\results_daily_noleak_baseline"
```

