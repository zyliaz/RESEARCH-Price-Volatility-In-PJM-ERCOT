# PJM-ERCOT Price Volatility Research: Development Guide

## Project Overview

This project researches the relationship between electricity price volatility and load/generation in ERCOT and PJM markets, focusing on regions with significant data center growth:
- **PJM**: DOM (Dominion) Zone - Northern Virginia
- **ERCOT**: West Zone - West Texas

## Research Questions

1. How does price volatility differ between energy-only (ERCOT) and capacity (PJM) markets?
2. What is the relationship between load growth (especially data center load) and price volatility?
3. How does renewable penetration affect price volatility in each market?
4. Can we predict volatility spikes using load and generation features?

---

## Data Sources

### ERCOT Data

| Data Type | Product ID | Resolution | Description |
|-----------|------------|------------|-------------|
| RTM Settlement Point Prices | NP6-905-CD | 15-min | Real-time market prices by zone/hub |
| DAM Settlement Point Prices | NP4-190-CD | Hourly | Day-ahead market prices |
| Hourly Load by Zone | NP6-345-CD | Hourly | Actual load by forecast zone |
| Generation by Fuel Type | Fuel Mix Report | 15-min | Generation breakdown by fuel |
| Wind/Solar Generation | NP4-732-CD | 5-min/Hourly | Renewable generation data |

**ERCOT Load Zones**: North, South, West, Houston
**Focus Zone**: West (data center growth, renewable integration, Permian Basin)

### PJM Data

| Data Type | Endpoint | Resolution | Description |
|-----------|----------|------------|-------------|
| RT Hourly LMPs | rt_hrl_lmps | Hourly | Real-time locational marginal prices |
| DA Hourly LMPs | da_hrl_lmps | Hourly | Day-ahead LMPs |
| 5-min LMPs | rt_fivemin_lmps | 5-min | High-frequency price data |
| Hourly Load | hrl_load_metered | Hourly | Metered load by zone |
| Generation by Fuel | gen_by_fuel | Hourly | Generation by fuel type |

**PJM Transmission Zones**: DOM, PECO, PEPCO, BGE, AEP, etc.
**Focus Zone**: DOM (Northern Virginia - world's largest data center cluster)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Extraction Layer                        │
│   src/extraction/ercot_extractor.py | pjm_extractor.py          │
├─────────────────────────────────────────────────────────────────┤
│                     Raw Data Storage                             │
│   data/raw/ercot/*.parquet | data/raw/pjm/*.parquet             │
├─────────────────────────────────────────────────────────────────┤
│                     Data Transformation Layer                    │
│   src/transformation/                                            │
│   - temporal_alignment.py (resample to hourly)                   │
│   - geographic_mapping.py (node-to-zone aggregation)             │
│   - data_cleaning.py (missing data, outliers)                    │
├─────────────────────────────────────────────────────────────────┤
│                     Processed Data                               │
│   data/processed/combined_dataset.parquet                        │
├─────────────────────────────────────────────────────────────────┤
│                     Feature Engineering Layer                    │
│   src/features/                                                  │
│   - volatility_features.py                                       │
│   - load_features.py                                             │
│   - generation_features.py                                       │
├─────────────────────────────────────────────────────────────────┤
│                     ML Modeling Layer                            │
│   src/models/                                                    │
│   - garch_models.py | ml_models.py | evaluation.py              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Extraction | gridstatus |
| Data Processing | pandas, numpy |
| Storage | Parquet (pyarrow) |
| Time Series | statsmodels, arch |
| ML Models | scikit-learn, xgboost |
| Visualization | matplotlib, seaborn, plotly |
| Notebooks | Jupyter |

---

## Configuration

### Confirmed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ERCOT Region | West Zone | High data center growth, renewable integration |
| PJM Region | DOM Zone | Northern Virginia data center cluster |
| Date Range | 2022-2025 | Post-COVID, captures data center boom |
| Weather Data | Temperature only | Correlates with load and price spikes |

---

## Feature Engineering

### Volatility Metrics
- `realized_vol_hourly`: Std dev of sub-hourly returns
- `realized_vol_daily`: Std dev of hourly returns within day
- `price_range`: Max - Min price in period
- `price_spike_flag`: 1 if price > 3x rolling mean

### Load Features
- `load_mw`: Actual load (MW)
- `load_ramp_rate`: Hour-over-hour load change
- `peak_hour_indicator`: 1 if in peak hours (HE7-22)
- `load_pct_of_peak`: Current load / historical peak

### Generation Features
- `renewable_penetration`: (Wind + Solar) / Total generation
- `gas_generation_pct`: Natural gas share
- `net_load`: Load - Wind - Solar
- `net_load_ramp`: Hour-over-hour net load change

### Market Structure Features
- `market_type`: ERCOT vs PJM indicator
- `congestion_component`: LMP congestion price
- `loss_component`: LMP marginal loss

---

## ML Modeling Approach

### Baseline Models
1. GARCH(1,1) - Traditional volatility modeling
2. EGARCH - Asymmetric volatility effects

### Machine Learning Models
1. XGBoost Regressor - Feature importance, non-linearity
2. Random Forest - Robust, interpretable
3. Hybrid GARCH-ML - GARCH residuals as ML features

### Target Variables
- `volatility_next_hour`: 1-hour ahead volatility forecast
- `volatility_next_day`: 24-hour ahead volatility forecast
- `spike_probability`: Probability of price spike

---

## Usage

### Data Extraction

```python
# Extract ERCOT data
python3 src/extraction/ercot_extractor.py

# Extract PJM data
python3 src/extraction/pjm_extractor.py
```

### Run EDA Notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## File Structure

```
PJM-ERCOT-Price-Volatility-Research/
├── docs/
│   └── DEVELOPMENT.md          # This file
├── src/
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration settings
│   │   ├── ercot_extractor.py  # ERCOT data extraction
│   │   └── pjm_extractor.py    # PJM data extraction
│   ├── transformation/
│   │   ├── __init__.py
│   │   ├── temporal_alignment.py
│   │   ├── geographic_mapping.py
│   │   └── data_cleaning.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── volatility_features.py
│   │   ├── load_features.py
│   │   └── generation_features.py
│   └── models/
│       ├── __init__.py
│       ├── garch_models.py
│       ├── ml_models.py
│       └── evaluation.py
├── data/
│   ├── raw/                    # Raw extracted data
│   │   ├── ercot/
│   │   └── pjm/
│   └── processed/              # Cleaned, transformed data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_volatility_modeling.ipynb
│   └── 04_comparative_analysis.ipynb
└── README.md
```

---

## References

- [gridstatus documentation](https://opensource.gridstatus.io/)
- [PJM Data Miner 2](https://dataminer2.pjm.com/)
- [ERCOT Market Information](https://www.ercot.com/mktinfo)
- [GARCH Models for Volatility](https://arch.readthedocs.io/)
