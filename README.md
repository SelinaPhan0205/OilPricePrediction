# OIL PRICE PREDICTION PROJECT
**Dự đoán biến động giá dầu dựa trên các yếu tố kinh tế, tài chính và địa chính trị**

---

## 📋 Tổng quan

Dự án này xây dựng mô hình Machine Learning nhằm **dự đoán biến động giá dầu** trên thị trường quốc tế, dựa trên 7 nguồn dữ liệu từ các tổ chức uy tín: ACLED, GDELT, FRED, EIA, và Yahoo Finance.

Giá dầu thô trên thị trường quốc tế chịu ảnh hưởng phức tạp từ ba nhóm yếu tố chính:
- **Địa chính trị**: Xung đột quân sự, bất ổn chính trị ở Trung Đông (khu vực sản xuất 30% dầu thô toàn cầu)
- **Kinh tế vĩ mô**: Lãi suất Fed, lạm phát, thất nghiệp, mong đợi tăng trưởng kinh tế
- **Thị trường tài chính & Cung/Cầu thực tế**: USD Index, S&P 500, VIX, tồn kho dầu, sản lượng khai thác

---

## 🎯 Mục tiêu

1. **Tích hợp đa nguồn dữ liệu**: Thu thập, làm sạch, và kết hợp dữ liệu từ 7 nguồn khác nhau
2. **Feature Engineering**: Tạo các biến thông tin có ý nghĩa từ dữ liệu thô
3. **Xây dựng mô hình dự báo**: Sử dụng Machine Learning để dự đoán chiều biến động hoặc giá tuyệt đối
4. **Phân tích yếu tố**: Xác định độ ảnh hưởng tương đối của từng nhóm yếu tố

---

## 📊 Nguồn dữ liệu

| Nhóm yếu tố | Nguồn | Series/Metrics | Giai đoạn | Tần suất |
|---|---|---|---|---|
| **Địa chính trị (thực tế)** | ACLED | Số sự kiện xung đột, Thương vong, Cường độ 7 ngày | 2015–2026/02 | Daily |
| **Địa chính trị (truyền thông)** | GDELT | Sentiment tone, Goldstein scale, Volume tin tức | 2015–2026/02 | Daily |
| **Giá dầu & Biến động** | Yahoo Finance | Brent/WTI Close, Return | 2015–2026/02 | Daily |
| **Thị trường tài chính** | Yahoo Finance | USD Index, S&P 500, VIX | 2015–2026/02 | Daily |
| **Kinh tế vĩ mô** | FRED | Fed Funds Rate, CPI, Unemployment, Yield Spread | 2015–2026/02 | Monthly/Daily |
| **Cung/Cầu dầu** | EIA | Tồn kho dầu, Sản lượng khai thác, Nhập khẩu ròng | 2015–2026/02 | Weekly |

**Quy mô dataset:**
- **Số observations**: ~2.871 business days (loại bỏ thứ Bảy, Chủ Nhật, ngày lễ thị trường)
- **Số features dự kiến**: ~37 (sau feature engineering)
- **Giai đoạn train/test**: Training 2015–2023; Testing 2024–2026/02

---

## 📁 Cấu trúc dự án

```
OilPriceProject/
│
├── README.md                          # Tệp này
├── requirements.txt                   # Thư viện Python cần thiết
│
├── data/
│   ├── raw/                           # Dữ liệu thô từ các nguồn
│   │   ├── eia_data.csv              # EIA API crawl
│   │   ├── fred_data.csv             # FRED API crawl
│   │   ├── gdelt_data.csv            # GDELT file download
│   │   ├── market_data.csv           # Yahoo Finance (Oil, USD, S&P500, VIX)
│   │   └── acled_data.csv            # ACLED manual export (khi có)
│   │
│   └── processed/
│       ├── dataset_preprocessed.csv  # Sau cleaning & handling missing values
│       ├── dataset_final.csv         # Sau merge & aligned dates
│       └── dataset_final_full.csv    # Với tất cả features sau feature engineering
│
├── scripts/
│   ├── crawl_macro_supply.py         # EIA & FRED API crawling
│   ├── crawl_gdelt.py                # GDELT file downloading & parsing
│   ├── ingest_data.py                # Load raw data & basic validation
│   ├── preprocess_data.py            # Cleaning, handle missing values, align dates
│   ├── feature_engineering.py        # Create lag, rolling, time-based features
│   └── visualize_data.py             # EDA & correlation plots
│
└── notebooks/
    └── (sẽ thêm Jupyter notebooks cho analysis & modeling)
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Chuẩn bị môi trường

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu

#### 2.1 ACLED (thủ công)
1. Truy cập [acleddata.com/data-export-tool](https://acleddata.com/conflict-data/data-export-tool)
2. Filter: Khu vực = Trung Đông (Iraq, Iran, Saudi Arabia, Syria, Yemen, Israel, Palestine, Lebanon, Kuwait, UAE, Qatar, Bahrain, Oman, Jordan)
3. Giai đoạn: 2015-01-01 đến 2026-02-27
4. Export as CSV → lưu vào `data/raw/acled_data.csv`

#### 2.2 FRED & EIA (API)
```bash
# Cần API keys:
# - FRED: https://fred.stlouisfed.org/docs/api/ (miễn phí)
# - EIA: https://www.eia.gov/opendata/ (miễn phí)

python scripts/crawl_macro_supply.py --fred-key YOUR_FRED_KEY --eia-key YOUR_EIA_KEY
```

#### 2.3 Yahoo Finance & GDELT (tự động trong script)
```bash
python scripts/ingest_data.py
```

### 3. Xử lý dữ liệu

```bash
# Cleaning & alignment
python scripts/preprocess_data.py

# Feature engineering
python scripts/feature_engineering.py

# EDA & Visualization
python scripts/visualize_data.py
```

---

## 📈 Dòng xử lý dữ liệu (Pipeline)

```
Raw Data (7 sources)
    ↓
[ingest_data.py] Load & validate
    ↓
[preprocess_data.py] Clean, handle missing values, align dates
    ↓
Preprocessed Dataset (2.871 rows × 28 base features)
    ↓
[feature_engineering.py] Create lag, rolling, seasonal features
    ↓
Final Dataset (2.871 rows × ~37 features)
    ↓
[Modeling] (sẽ thêm trong giai đoạn tiếp theo)
```

### Các xử lý quan trọng:

1. **Xử lý ngày cuối tuần**: Sự kiện xung đột xảy ra cuối tuần được gộp vào thứ Hai (ngày giao dịch tiếp theo)
2. **Forward-fill cho dữ liệu thưa**: EIA (weekly) và FRED monthly được forward-fill, không nội suy
3. **Handle missing values**:
   - GDELT: 14 ngày có missing → forward-fill hoặc interpolation
   - Các source khác: Không có missing values sau forward-fill
4. **Standardization**: Các feature sẽ được chuẩn hóa (Z-score) trước khi đưa vào model

---

## 📊 Danh sách Features (cuối cùng dự kiến)

### Features gốc từ các nguồn (~28 features)

**ACLED (3 features)**
- `conflict_event_count`: Số sự kiện xung đột trong ngày
- `fatalities`: Số thương vong
- `conflict_intensity_7day`: Cường độ 7 ngày (rolling sum)

**GDELT (9 features)**
- `gdelt_tone`: Sentiment trung bình
- `gdelt_goldstein`: Goldstein scale trung bình
- `gdelt_volume`, `gdelt_events`: Khối lượng tin tức
- `gdelt_tone_7d`, `gdelt_goldstein_7d`: Rolling 7-day average
- `gdelt_tone_30d`, `gdelt_tone_spike`: Rolling 30-day + flag

**Yahoo Finance - Oil (2 features)**
- `oil_close_price`: Giá dầu đóng cửa
- `oil_return`: Tỷ lệ thay đổi giá

**Yahoo Finance - Markets (6 features)**
- `usd_index_close`, `usd_return`
- `sp500_close`, `sp500_return`
- `vix_close`, `vix_return`

**FRED (5 features)**
- `fed_funds_rate`, `cpi`, `unemployment`: Macro indicators
- `yield_spread`: Chênh lệch lãi suất 10Y–2Y
- `wti_fred`: WTI giá từ FRED (cross-check)

**EIA (4 features)**
- `crude_inventory_weekly`: Tồn kho dầu
- `crude_production_weekly`: Sản lượng khai thác
- `net_imports_weekly`: Nhập khẩu ròng
- `inventory_change_pct`: Thay đổi tồn kho (%)

### Features Engineering (~8 features)

**Lag features (2)**
- `oil_return_lag1`, `oil_return_lag2`

**Rolling features (3)**
- `conflict_7d`: Tổng sự kiện 7 ngày
- `fatalities_7d`: Tổng thương vong 7 ngày
- `oil_volatility_7d`: Biến động 7 ngày

**Temporal features (2)**
- `day_of_week`: Thứ trong tuần
- `month`: Tháng trong năm

---

## ⚠️ Ghi chú quan trọng

1. **Data Leakage phòng chống**: FRED monthly data được shift 1 tháng trước khi merge, tránh model thấy trước thông tin chưa công bố
2. **Outlier thực tế**: `wti_fred` có giá trị -36.98 vào 20/4/2020 — đây là sự kiện thực có thật (lần đầu tiên giá dầu âm trong lịch sử), không phải lỗi dữ liệu, sẽ giữ nguyên
3. **Multicollinearity**: `wti_fred` có correlation > 0.99 với `oil_close_price`, sẽ xem xét bỏ bớt hoặc giữ để cross-check
4. **Seasonality**: Nhu cầu dầu tăng vào mùa đông và mùa lái xe hè — `month` feature sẽ giúp nắm bắt điều này

---

## 📝 Giai đoạn dự kiến

- [x] Thu thập & làm sạch dữ liệu
- [x] Preprocessing & merge datasets
- [ ] Feature engineering & EDA chi tiết
- [ ] Train-test split & model selection
- [ ] Xây dựng & tuning mô hình
- [ ] Evaluating & Feature importance analysis
- [ ] Visualize kết quả & viết báo cáo

---

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**
- **pandas**: Data manipulation & preprocessing
- **numpy**: Numerical computing
- **scikit-learn**: Machine Learning models
- **matplotlib & seaborn**: Data visualization
- **requests & yfinance**: API calls & web scraping
- **jupyter**: Interactive notebooks (sắp tới)

---

## 📚 Tài liệu tham khảo

- [ACLED - Armed Conflict Location & Event Data Project](https://acleddata.com/)
- [GDELT - Global Database of Events, Language, and Tone](https://www.gdeltproject.org/)
- [FRED - Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- [EIA - U.S. Energy Information Administration](https://www.eia.gov/)
- [Yahoo Finance](https://finance.yahoo.com/)

---

## 👤 Tác giả

**Team OilPriceProject**  
Khóa KTDLUD, Đại học [Tên ĐH]

---

## 📄 License

MIT License - Sử dụng tự do cho mục đích học tập và nghiên cứu

---

## 📧 Liên hệ & Hỗ trợ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ hoặc mở issue trong repository.

**Last Updated**: 2026-03-23
