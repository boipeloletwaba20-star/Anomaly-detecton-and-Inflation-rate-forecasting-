# Anomaly Detection in Electricity Meter Readings & Forecasting South African Inflation

**Two problems, one theme: build models that are validated honestly.**


---

## Problem
This repo answers two separate questions:
- **Part A, anomaly detection:** given a year of hourly electricity meter readings from 200 labelled buildings, flag anomalous readings in 206 *different*, unseen buildings.
- **Part B, inflation forecasting:** forecast South Africa's monthly headline CPI inflation (year-on-year, %) for January to December 2026, comparing a statistical model with a machine learning model.

## Data
| Part | Source | What | Size |
|---|---|---|---|
| A | Module-supplied dataset (Google Drive) | `train.csv`: building_id, timestamp, meter_reading, anomaly label<br>`test.csv`: row_id, building_id, timestamp, meter_reading | 1,749,494 training rows (200 buildings)<br>1,800,567 test rows (206 buildings) |
| B | [Statistics South Africa, CPI (P0141)](https://www.statssa.gov.za/?page_id=1859) | Headline CPI year-on-year inflation, monthly | 60 months (Jan 2021 – Dec 2025); Jan–Jun 2026 held out |

The Part A data was supplied by the module and is not redistributed here. The Part B CPI data is in `part-b-inflation-forecasting/data/`.

## Tools
Python · Google Colab · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `xgboost` · `catboost` · `statsmodels` (ADF, ARIMA, Holt) · `matplotlib` · `seaborn`

## Method

### Part A: Anomaly detection
1. **Cleaning:** timestamps parsed, no duplicates found, and missing readings (about 6%) filled *within each building* by forward then backward fill. Buildings differ in scale by orders of magnitude, so a global mean would have created false anomalies.
2. **Class imbalance:** only **2.13%** of readings are anomalous, so accuracy is meaningless. F1 and PR-AUC were used as the primary metrics.
3. **Feature engineering:** 32 features built within each building's own time series: lags (1, 24, 48 and 168 hours), rolling mean, standard deviation, minimum and maximum (6, 24 and 168 hours), residuals and z-scores, percentage change, cyclical hour, day and month encodings, and an IQR outlier flag.
4. **Leakage-safe validation:** a **building-grouped** split (70% of buildings for training, 30% for validation), checked with 5-fold `GroupKFold`, because the test set contains entirely new buildings.
5. **Models:** XGBoost (`scale_pos_weight`), Balanced Random Forest and CatBoost, each with the decision threshold tuned for F1 on the validation set only.

### Part B: Inflation forecasting
1. **Target:** the year-on-year inflation rate rather than the CPI index. The rate is what the Reserve Bank targets, it's directly interpretable, and it removes the index's upward drift.
2. **Exploratory analysis:** additive decomposition, an ADF test (non-stationary in levels, stationary after one difference), and ACF and PACF plots.
3. **Backtest:** an expanding-window, one-step-ahead backtest over January to December 2025. All tuning used pre-2025 data only.
4. **Models:** ARIMA(0,1,1), selected by AIC search; XGBoost with lag, rolling and calendar features, tuned with an 18-combination grid; and a damped-trend Holt model as a benchmark.
5. **Final forecast:** all models refitted on the full 2021–2025 data to forecast 2026, with 95% intervals.

## Key findings

### Part A
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.851 | 0.655 | 0.740 | 0.933 | 0.773 |
| **Balanced Random Forest** | **0.969** | **0.727** | **0.830** | 0.937 | **0.824** |
| CatBoost | 0.835 | 0.522 | 0.643 | 0.942 | 0.690 |

- The **Balanced Random Forest** was selected (threshold 0.947). When it flags a reading, it is correct about 97% of the time.
- **ROC-AUC is misleading under heavy imbalance.** CatBoost had the best ROC-AUC but missed almost half of the real anomalies.
- On the unseen test buildings, the model flagged **19,006 of 1,800,567 readings (1.06%)**.
- The anomalies are **contextual**: spikes against a building's own baseline, flat stuck readings and sudden hour-on-hour jumps, not simple global outliers.

### Part B
| Model | MAE (pp) | RMSE (pp) | MAPE | MASE |
|---|---|---|---|---|
| ARIMA(0,1,1) | 0.252 | 0.278 | 8.09% | 0.740 |
| **XGBoost** | **0.204** | 0.268 | **6.35%** | **0.598** |
| Holt damped trend | 0.218 | **0.266** | 6.88% | 0.639 |

- **XGBoost** had the lowest MAE and forecast about **3.5% inflation for every month of 2026**, inside the new 2–4% target band.
- The margins between the models are small (hundredths of a percentage point over 12 forecasts), so the three forecasts are best read as a range.
- **Out-of-sample check:** actual inflation rose to **5.0% by June 2026**. All three models missed this, because they use only past inflation and cannot see shocks such as fuel prices.



## How to run
1. Open the notebooks in Google Colab: [Part A]([COLAB_LINK_PART_A](https://colab.research.google.com/drive/1MDvWSNSp2ID4EM7hPbGnUUqaZX0D_L94?usp=sharing)) · [Part B]([COLAB_LINK_PART_B](https://colab.research.google.com/drive/1LejUVj8y6XXFrCiyzsSVcBNlG7ZlxZ4b?usp=sharing))
2. **Part A:** place `train.csv`, `test.csv` and `sample_submission.csv` in the Colab working folder (or your Google Drive), then **Runtime → Run all**. The first cell installs the required packages. The notebook writes `submission.csv` (row_id, anomaly).
3. **Part B:** upload `CPI_monthly_2020-2026.xlsx` from `part-b-inflation-forecasting/data/`, then **Runtime → Run all**. The notebook writes `CPI_2026_Forecast.xlsx`.

## Outputs
- **Part A test predictions:** [submission.csv]:(https://drive.google.com/file/d/1WXlAQc0IaoRqvr6ZIxT4wvJTiEKJQXi_/view?usp=sharing)) (1,800,567 rows, too large for GitHub, so it's stored on Google Drive)
- **Part B 2026 forecasts:** `part-b-inflation-forecasting/outputs/CPI_2026_Forecast.xlsx`[https://docs.google.com/spreadsheets/d/1rqdPiuzrZaLLg4c4VUa6oGNp2KOKmBrl/edit?usp=sharing&ouid=103415627662826488641&rtpof=true&sd=true]

## Repository structure
```
anomaly-detection-and-inflation-forecasting/
├── README.md
├── part-a-anomaly-detection/
│   └── anomaly_detection_meter_readings.ipynb
├── part-b-inflation-forecasting/
│   ├── cpi_forecasting_2026.ipynb
│   ├── data/
│   │   └── CPI_monthly_2020-2026.xlsx
│   └── outputs/
     └── CPI_2026_Forecast.xlsx

```

