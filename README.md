
# Deep Learning & ML Model Handover

This repository contains the complete set of 13 Machine Learning and Deep Learning models for cardiac arrhythmia classification, trained on the 'small_refined' dataset.

## 📂 Repository Structure

```
.
├── models/                  # Trained model artifacts (.joblib, .zip, .ckpt)
│   ├── XGBoost.joblib
│   ├── TabNet.zip           # (Example DL model)
│   ├── mlb.joblib           # Label binarizer
│   ├── imputer.joblib       # Data imputer
│   └── features.csv         # Feature list
├── app.py                   # Streamlit demo application (HF Space ready)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Demo App
```bash
streamlit run app.py
```

## 🧠 Models Included
1.  **Ensemble/Boosting**: XGBoost, LightGBM, RandomForest, Voting, Stacking
2.  **Traditional**: LogisticRegression, SVM, KNN, GaussianNB
3.  **Deep Learning**: TabNet, FT-Transformer, NODE, MLP

## 🔧 Maintenance
- **Re-training**: Use the provided `save_all_models.py` (parent directory) to re-train traditional models.
- **Data**: Models expect input features as defined in `models/features.csv`.
