# HVAC Fault Detection & Energy Optimization Using Machine Learning and Deep Learning

> Capstone Project | CS-326: Applied AI and Machine Learning | Spring 2026

---

## Team Members

| Name | ID |
|---|---|
| M Zain Ul Abideen | ME-1932 |
| M Hashir | ME-1960 |
| Aun Raza | ME-1968 |
| Talha Yasir | ME-1952 |

---

## Project Overview

This project presents a complete 3-component HVAC Health Monitoring System that uses Machine Learning and Deep Learning to detect chiller faults and optimize building energy consumption. The system transforms HVAC management from reactive emergency maintenance to proactive, predictive health management.

**The two core problems addressed:**
- **Energy Waste** — HVAC consumes 40% of total commercial building energy worldwide, with inefficient scheduling wasting $300 billion annually
- **Undetected Faults** — Commercial chillers operate with hidden faults for months, increasing energy draw by 20–40%, while traditional rule-based systems catch only ~70% of faults

---

## Repository Structure

```
├── eda_energy_optimization.ipynb       # EDA for energy dataset (ASHRAE GEPIII)
├── ml_models_energy_optimization.ipynb # ML models: RF, LightGBM, XGBoost for energy
├── dl_model_energy_optimization.ipynb  # Deep Learning (LSTM) for energy prediction
├── fdd_eda.ipynb                       # EDA for fault detection dataset (RP-1043)
├── fdd_ml.ipynb                        # ML models for hierarchical FDD
├── fdd_dl.ipynb                        # Neural Network for FDD
└── integration.ipynb                   # Automated trigger pipeline integration
```

---

## Datasets

### Dataset 1 — ASHRAE Great Energy Predictor III (Energy Optimization)
- **Size:** 4.3 million filtered rows (from 20.2M) across 226 buildings, full year 2016
- **Target:** Meter readings across 4 types — Electricity, Chilled Water, Steam, Hot Water
- **Features:** Building metadata, weather data (air temp, dew temp, wind speed, CDD, HDD), time features

### Dataset 2 — ASHRAE RP-1043 Chiller Fault Dataset (Fault Detection)
- **Size:** 11,000 rows of controlled laboratory sensor data
- **Target:** 8 classes — 1 Normal + 7 fault types (Refrigerant Leak, Refrigerant Overcharge, Reduced Condenser Flow, Reduced Evaporator Flow, Condenser Fouling, Excess Oil, Non-condensables)
- **Features:** 16 physical sensor readings (TCI/TCO, subcooling, approach temp, etc.)

---

## Methodology

### Energy Optimization (Regression)
- Separate model trained per meter type
- Log transformation applied to target variable
- Lag features engineered (lag_1h, lag_24h, lag_168h, rolling means)
- Stratified train/test split by month to prevent seasonal bias
- Models: **Random Forest (baseline) → LightGBM → XGBoost → LSTM**

### Fault Detection & Diagnosis — FDD (Classification)

- **2-Level Hierarchical Architecture:**
  - **Level 1:** Binary classification — Normal vs. Faulty
  - **Level 2:** Multi-class diagnosis — Which of the 7 fault types?
- Models: **Random Forest (baseline) → LightGBM → XGBoost → Neural Network**
- Class imbalance handled via `class_weight = 'balanced'`

---

## Results

### Energy Optimization — Best R² per Meter

| Meter | Best Model | R² Score |
|---|---|---|
| Electricity | LightGBM | 0.9846 |
| Chilled Water | XGBoost | 0.9651 |
| Steam | Random Forest | 0.9552 |
| Hot Water | Random Forest | 0.8708 |

> Lag features alone improved Electricity R² from 0.914 → 0.985 (+7.7%)

### FDD — Hierarchical Classification Performance

| Model | Level 1 Accuracy | Level 2 Accuracy |
|---|---|---|
| Random Forest | 0.9900 | 0.9971 |
| LightGBM | 0.9968 | 0.9964 |
| XGBoost | 0.9955 | 0.9971 |
| Neural Network | 0.9855 | 0.9957 |

### LSTM — Energy Optimization (Deep Learning)

| Meter | R² |
|---|---|
| Electricity | 0.9321 |
| Chilled Water | 0.9253 |
| Steam | 0.9407 |
| Hot Water | 0.8183 |

---

## Key Engineering Decisions & Constraints

- Dataset filtered from 20.2M → 4.3M rows (2 of 16 sites) due to Google Colab 12GB RAM limit
- Data types reduced from float64 → float32 to manage memory
- Parquet format used instead of CSV (137MB vs 1.77GB for the same data)
- LSTM trained on 80 of 226 buildings due to memory constraints
- Hard outlier caps applied per meter to prevent Colab session resets
- Highly missing features (`floor_count` 82%, `year_built` 60%) dropped

---

## Ethical Considerations

- **Bias & Fairness:** Dataset spans 16 building types; stratified sampling ensures no seasonal or class bias
- **Privacy:** ASHRAE dataset is fully anonymized — no real addresses or personally identifiable information
- **Interpretability:** Feature importance plots and confusion matrices generated for all models; lag features provide physically meaningful explanations

---

## Future Roadmap

- Expand FDD to boiler systems using LBNL datasets
- Real-time Building Management System (BMS) integration
- Fault severity forecasting
- Anomaly detection for novel/unlabeled fault types
- SHAP-based explainability dashboard for field technicians
- Federated learning across buildings for privacy-preserving deployment

---

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- LightGBM, XGBoost
- TensorFlow / Keras (LSTM & Neural Network)
- matplotlib, seaborn
- pyarrow (Parquet support)

---

## Acknowledgements

- ASHRAE for the Great Energy Predictor III and RP-1043 datasets
- CS-326: Applied AI and Machine Learning course, Spring 2026
