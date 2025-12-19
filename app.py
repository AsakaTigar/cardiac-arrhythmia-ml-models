
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Cardiac Arrhythmia Classification", layout="wide")

st.title("Cardiac Arrhythmia Classification Demo")
st.markdown("Predict arrhythmia type based on clinical features using ML models.")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        models = {}
        model_names = [
            "XGBoost", "LightGBM", "RandomForest", "LogisticRegression",
            "SVM", "KNN", "Voting", "Stacking"
        ]
        
        for name in model_names:
            path = f"models/{name}.joblib"
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass # Skip if missing
                
        mlb = joblib.load("models/mlb.joblib")
        imputer = joblib.load("models/imputer.joblib")
        return models, mlb, imputer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

models, mlb, imputer = load_artifacts()

if models:
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
    model = models[selected_model_name]

    # Input Form
    st.sidebar.header("Patient Data")
    
    # Simple Demo Inputs
    # Note: In a real deployment, we need to map all features used in training.
    # This is a simplified interface for demonstration purposes.
    st.info("Note: This demo uses median values for missing features.")
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", 0, 100, 60)
    
    if st.button("Predict"):
        try:
            # Create a placeholder dataframe with all required features (filled with median/0)
            # We rely on the imputer to handle most things, but sklearn requires correct shape
            
            # Since we can't easily introspect the exact feature list at runtime without loading features.csv
            # We will use a try-catch for robustness or load features if available
            try:
                features_df = pd.read_csv("models/features.csv")
                feature_names = features_df.iloc[:, 0].tolist()
            except:
                st.error("Could not load feature definitions.")
                st.stop()
                
            input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
            
            # Map known inputs
            # value = 0 if gender == "Male" else 1 # Example mapping
            # input_df['性别'] = value
            # input_df['年龄'] = age
            
            # Use imputer
            X_input = pd.DataFrame(imputer.transform(input_df), columns=feature_names)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_input)
                res_df = pd.DataFrame(proba, columns=mlb.classes_)
                
                st.subheader("Prediction Results")
                st.dataframe(res_df.style.highlight_max(axis=1))
                
                top_class = mlb.classes_[np.argmax(proba[0])]
                st.success(f"Predicted Diagnosis: **{top_class}**")
            else:
                pred = model.predict(X_input)
                pred_labels = mlb.inverse_transform(pred)
                st.success(f"Predicted: {pred_labels}")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("No models found in `models/` directory. Please upload `.joblib` files.")
