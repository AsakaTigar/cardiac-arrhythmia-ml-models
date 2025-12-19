
# Handover Models Package
This directory contains trained models for the cardiac arrhythmia classification task.

## Dataset Mode
- **Mode**: `small_refined`
- **Labels**: AF/AFL, AVNRT, AVRT (SVT subtypes separate), but AT/VT/VF/PVCs excluded from training rows.

## Loading Models
Use `joblib` to load sklearn/xgboost models:
```python
import joblib
model = joblib.load('models/XGBoost.joblib')
mlb = joblib.load('models/mlb.joblib')
imputer = joblib.load('models/imputer.joblib')
```

## Deep Learning Models
DL models (TabNet, etc.) may require their specific libraries (pytorch-tabular, pytorch-tabnet) to load.
