"""
Configuration file for ED-AI Triage System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Model file paths
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
LGB_MODEL_PATH = MODELS_DIR / "lgb_model.joblib"
SCALER_PATH = MODELS_DIR / "advanced_scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "structured_features.joblib"
BERT_TOKENIZER_PATH = MODELS_DIR / "bert_tokenizer.joblib"
BERT_MODEL_PATH = MODELS_DIR / "bert_model.pth"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

# Data file paths
TRIAGE_DATA_PATH = DATA_DIR / "triage.csv"
VITALS_DATA_PATH = DATA_DIR / "vitalsign.csv"
DIAGNOSIS_DATA_PATH = DATA_DIR / "diagnosis.csv"
EDSTAYS_DATA_PATH = DATA_DIR / "edstays.csv"
MEDRECON_DATA_PATH = DATA_DIR / "medrecon.csv"
PYXIS_DATA_PATH = DATA_DIR / "pyxis.csv"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8080
STREAMLIT_PORT = 8501

# Model hyperparameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Feature engineering parameters
STRUCTURED_FEATURES = [
    'age', 'temperature', 'heart_rate', 'respiratory_rate', 'oxygen_saturation',
    'blood_pressure_systolic', 'blood_pressure_diastolic', 'pain_score',
    'shock_index', 'mean_arterial_pressure', 'pulse_pressure',
    'fever', 'hypotension', 'tachycardia', 'tachypnea', 'hypoxia', 'severe_pain',
    'arrival_mode_encoded', 'consciousness_encoded', 'gender_encoded'
]

# Clinical thresholds
FEVER_THRESHOLD = 38.0
HYPOTENSION_THRESHOLD = 90
TACHYCARDIA_THRESHOLD = 100
TACHYPNEA_THRESHOLD = 20
HYPOXIA_THRESHOLD = 95
SEVERE_PAIN_THRESHOLD = 7

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
KNN_IMPUTE_NEIGHBORS = 5

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "ED-AI-Triage-Advanced"
MLFLOW_TRACKING_URI = None  # Use default

# Docker configuration
DOCKER_IMAGE_NAME = "ed-ai-triage"
DOCKER_TAG = "latest"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

# Call ensure directories on import
ensure_directories()
