# ============================================================
# 🏥 HEART DISEASE RISK API - UNIVERSAL PRODUCTION BUILD
# ============================================================
import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")
# ------------------------------------------------------------
# 1. THE "NUCLEAR FIX" FOR JOBLIB/PICKLE
# ------------------------------------------------------------
# This forces the unpickler to look at THIS file when it asks for "__main__"
# This solves the "AttributeError: module '__main__' has no attribute..."
# This must run before classes are defined.
if __name__ != '__main__':
    sys.modules['__main__'] = sys.modules[__name__]

# ------------------------------------------------------------
# 2. DEFINE CUSTOM CLASSES (Must exactly match training code)
# ------------------------------------------------------------
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        super().__init__()
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=np.number)
            for col in X_numeric.columns:
                Q1 = X_numeric[col].quantile(0.25)
                Q3 = X_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds[col] = Q1 - self.factor * IQR
                self.upper_bounds[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X_copy = pd.DataFrame(X_copy, columns=self.feature_names_in_)
            else:
                return X_copy 

        for col in self.lower_bounds:
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_) if self.feature_names_in_ is not None else input_features


class DataAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, add_age_groups=True, add_noise=False, noise_factor=0.01):
        super().__init__()
        self.add_age_groups = add_age_groups
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        self.age_bins = [0, 30, 45, 60, 100]
        self.age_labels = ['Young Adult', 'Middle-aged', 'Senior', 'Elderly']
        self.feature_names_in_ = None
        self.numerical_cols_for_noise = None

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not isinstance(X_transformed, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names_in_)
            else:
                return X_transformed

        if self.add_age_groups and 'age_years' in X_transformed.columns:
            X_transformed['age_years'] = pd.to_numeric(X_transformed['age_years'], errors='coerce')
            X_transformed['age_group'] = pd.cut(
                X_transformed['age_years'], 
                bins=self.age_bins, 
                labels=self.age_labels, 
                right=False, 
                include_lowest=True
            )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_) if self.feature_names_in_ is not None else list(input_features)
        if self.add_age_groups and 'age_group' not in names:
            names.append('age_group')
        return np.array(names)


# ------------------------------------------------------------
# 3. INITIALIZE APP & LOAD MODELS
# ------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DYNAMIC PATH FINDING (Works on Local AND Render)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_PATH, "models")

print(f"📂 Model Directory: {MODELS_PATH}")

# Global variables
preprocessing_pipeline_A = None
preprocessing_pipeline_B = None
preprocessing_pipeline_C = None
final_model = None
ensemble_imputer = None
global_target_features = None

def load_models():
    global preprocessing_pipeline_A, preprocessing_pipeline_B, preprocessing_pipeline_C
    global final_model, ensemble_imputer, global_target_features
    
    try:
        preprocessing_pipeline_A = joblib.load(os.path.join(MODELS_PATH, 'preprocessing_pipeline_A.joblib'))
        preprocessing_pipeline_B = joblib.load(os.path.join(MODELS_PATH, 'preprocessing_pipeline_B.joblib'))
        preprocessing_pipeline_C = joblib.load(os.path.join(MODELS_PATH, 'preprocessing_pipeline_C.joblib'))
        ensemble_imputer = joblib.load(os.path.join(MODELS_PATH, 'Ensemble_Imputer.joblib'))
        global_target_features = joblib.load(os.path.join(MODELS_PATH, 'global_features.joblib'))
        
        # Try loading VotingClassifier first, if not found, try StackingClassifier
        try:
            final_model = joblib.load(os.path.join(MODELS_PATH, 'VotingClassifier_ensemble.joblib'))
            print("✅ Loaded VotingClassifier")
        except:
            print("⚠️ VotingClassifier not found, trying StackingClassifier...")
            final_model = joblib.load(os.path.join(MODELS_PATH, 'StackingClassifier_ensemble.joblib'))
            print("✅ Loaded StackingClassifier")
            
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        
    except Exception as e:
        print("\n❌ FATAL ERROR LOADING MODELS")
        print(f"Error: {e}")
        # We do NOT exit here so the server can at least start and show the error log
        # But predictions will fail safely.

# Load models on startup
load_models()


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------
def create_patient_dfs(data: dict):
    col_names_A = ['age_years', 'gender', 'smoke', 'alcohol', 'physical_activity']
    col_names_B = ['age_years', 'bmi', 'cholesterol', 'dbp', 'glucose', 'sbp', 'gender', 'heart_disease', 'hypertension', 'smoke', 'smoking_status']
    col_names_C = ['age_years', 'gender', 'cp', 'sbp', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'thalach', 'exercise_angina', 'oldpeak', 'st_slope', 'num_major_vessels', 'thalassemia']

    base_df = pd.DataFrame([data])
    
    all_cols = set(col_names_A + col_names_B + col_names_C)
    for col in all_cols:
        if col not in base_df.columns:
            base_df[col] = np.nan

    df_A = base_df[col_names_A].copy()
    df_B = base_df[col_names_B].copy()
    df_C = base_df[col_names_C].copy()

    if 'gender' in df_C.columns: df_C['gender'] = pd.to_numeric(df_C['gender'], errors='coerce')
    if 'fasting_blood_sugar' in df_C.columns: df_C['fasting_blood_sugar'] = df_C['fasting_blood_sugar'].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0})
    if 'exercise_angina' in df_C.columns: df_C['exercise_angina'] = df_C['exercise_angina'].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0})

    return df_A, df_B, df_C

def transform_to_df(pipeline, df_raw):
    arr = pipeline.transform(df_raw)
    if isinstance(arr, pd.DataFrame):
        return arr
    try:
        cols = pipeline.get_feature_names_out() if hasattr(pipeline, 'get_feature_names_out') else pipeline.steps[-1][1].get_feature_names_out()
    except:
        cols = [f"feat_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)

def align_dataframe_to_features(df, features):
    aligned_df = pd.DataFrame(index=df.index)
    for feature in features:
        aligned_df[feature] = df[feature] if feature in df.columns else np.nan
    return aligned_df


# ------------------------------------------------------------
# 5. API ENDPOINTS
# ------------------------------------------------------------
@app.get("/")
def health_check():
    if final_model is None:
        return {"status": "error", "message": "Models failed to load. Check server logs."}
    return {"status": "active", "message": "Heart Disease API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}

class PatientData(BaseModel):
    age_years: float
    gender: int
    smoke: int = 0
    alcohol: int = 0
    physical_activity: int = 1
    bmi: float
    cholesterol: float
    dbp: float
    glucose: float
    sbp: float
    cp: str
    fasting_blood_sugar: bool
    rest_ecg: str
    thalach: float = 150
    exercise_angina: bool
    oldpeak: float = 0.0
    st_slope: str = 'flat'
    num_major_vessels: float = 0.0
    thalassemia: str = 'normal'

@app.post("/predict")
def predict(patient: PatientData):
    if final_model is None:
        return {"error": "Models are not loaded. Please check server logs for errors during startup."}

    data = patient.model_dump()
    print(f"\n📥 Input: {data}")

    try:
        df_A_raw, df_B_raw, df_C_raw = create_patient_dfs(data)

        processed_A = transform_to_df(preprocessing_pipeline_A, df_A_raw)
        processed_B = transform_to_df(preprocessing_pipeline_B, df_B_raw)
        processed_C = transform_to_df(preprocessing_pipeline_C, df_C_raw)

        aligned_A = align_dataframe_to_features(processed_A, global_target_features)
        aligned_B = align_dataframe_to_features(processed_B, global_target_features)
        aligned_C = align_dataframe_to_features(processed_C, global_target_features)

        X_ensemble = pd.concat([aligned_A, aligned_B, aligned_C], axis=1)
        X_ensemble = X_ensemble.loc[:, ~X_ensemble.columns.duplicated()]

        X_final = pd.DataFrame(ensemble_imputer.transform(X_ensemble), columns=X_ensemble.columns)
        
        # Predict
        proba = final_model.predict_proba(X_final)[0, 1]

        risk = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.4 else "LOW")
        final_prediction = "Heart Disease" if proba > 0.5 else "Healthy"

        result = {
            "probability": float(proba),
            "risk_category": risk,
            "final_prediction": final_prediction
        }
        print(f"📤 Prediction: {result}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------
# 6. LOCAL EXECUTION (Not used by Render)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)