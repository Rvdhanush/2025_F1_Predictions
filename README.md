# F1 Spanish GP 2025 Predictions

Predicting the results of the 2025 Spanish Grand Prix using machine learning and historical data analysis.

## Models

1. **XGBoost Predictor** (`xgboost_predictor.py`)
   - Gradient Boosting model using XGBoost
   - Features: practice session times, improvements, consistency metrics
   - Output: `spanish_gp_2025_xgboost_predictions.csv`

2. **FastF1 Predictor** (`fastf1_predictor.py`)
   - Uses FastF1 API for real timing data
   - Features: lap times, sector times, consistency metrics
   - Output: `spanish_gp_2025_fastf1_predictions.csv`

3. **Logic-based Predictor** (`logic_predictor.py`)
   - Rule-based approach using historical performance
   - Features: practice session performance, historical data
   - Output: `spanish_gp_2025_logic_predictions.csv`

## Project Structure

```
.
├── xgboost_predictor.py         # XGBoost-based prediction model
├── fastf1_predictor.py          # FastF1-based prediction model
├── logic_predictor.py           # Logic-based prediction model
├── spanish_gp_realdata.csv      # Raw F1 data
├── fetch_spanish_gp_data.py     # Data fetching script
├── data_fetcher.py             # Data processing utilities
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run any of the prediction models:
```bash
python xgboost_predictor.py
python fastf1_predictor.py
python logic_predictor.py
```

## Requirements

```
fastf1
pandas
numpy
scikit-learn
xgboost
matplotlib
```

## Data Sources

- FastF1 API for historical race data
- Practice session data (FP1-FP3)
- Driver and team statistics 