from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import torch
import uvicorn

from src.fairness_module import FairnessAssessor, generate_fairness_report
from src.interpretability_module import InterpretabilityEngine, generate_counterfactual_explanation
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
from fastapi import Body

app = FastAPI(
    title="ED-AI Triage System API",
    description="Async FastAPI server for ED-AI Triage System with ClinicalBERT, ensemble models, interpretability, fairness, and counterfactuals",
    version="1.0.0"
)

# Load models and resources at startup
models = {}
interpret_engine = None
fairness_assessor = None
tft_model = None
tft_dataset = None

class PatientFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120)
    temperature: float = Field(..., ge=30.0, le=45.0)
    heart_rate: int = Field(..., ge=0, le=250)
    respiratory_rate: int = Field(..., ge=0, le=100)
    oxygen_saturation: int = Field(..., ge=0, le=100)
    blood_pressure_systolic: int = Field(..., ge=0, le=300)
    blood_pressure_diastolic: int = Field(..., ge=0, le=200)
    pain_score: int = Field(..., ge=0, le=10)
    arrival_encoded: int = Field(..., ge=0, le=1)
    consciousness_encoded: int = Field(..., ge=0, le=2)
    gender_encoded: int = Field(..., ge=0, le=2)

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_probs: Dict[str, float]

class ExplanationRequest(BaseModel):
    features: List[float]

class ExplanationResponse(BaseModel):
    shap_values: Optional[List[float]]
    lime_explanation: Optional[List[Any]]
    integrated_gradients: Optional[List[float]]
    explanation_report: Optional[str]

class FairnessRequest(BaseModel):
    predictions: List[int]
    true_labels: List[int]
    protected_attributes: Dict[str, List[Any]]

class FairnessResponse(BaseModel):
    fairness_results: Dict[str, Any]
    fairness_report: str

class CounterfactualRequest(BaseModel):
    features: List[float]
    target_class: int = Field(1, ge=0, le=1)
    max_changes: int = Field(3, ge=1, le=10)

class CounterfactualResponse(BaseModel):
    success: bool
    changes: Optional[List[Dict[str, Any]]]
    original_prediction: List[float]
    counterfactual_prediction: List[float]
    message: Optional[str]

class TemporalData(BaseModel):
    data: List[Dict[str, Any]]

class TemporalPredictionResponse(BaseModel):
    predictions: List[float]

class DebiasingRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    protected_attr: List[Any]
    method: str = Field("reweighting", description="Debiasing method: 'reweighting' or 'adversarial'")
    target_fairness: float = Field(0.8, ge=0.0, le=1.0)

class DebiasingResponse(BaseModel):
    debiased_X: Optional[List[List[float]]]
    sample_weights: Optional[List[float]]
    message: str

@app.on_event("startup")
def load_resources():
    global models, interpret_engine, fairness_assessor, tft_model, tft_dataset
    try:
        models['xgb_model'] = joblib.load('models/xgb_model.joblib')
        models['lgb_model'] = models['xgb_model']  # Use XGBoost as placeholder for LightGBM
        models['scaler'] = joblib.load('models/advanced_scaler.joblib')
        models['feature_names'] = joblib.load('models/structured_features.joblib')
        # Load ClinicalBERT components if available
        try:
            models['bert_tokenizer'] = joblib.load('models/bert_tokenizer.joblib')
            models['bert_model'] = torch.load('models/bert_model.pth')
        except:
            models['bert_tokenizer'] = None
            models['bert_model'] = None
        interpret_engine = InterpretabilityEngine(
            models['xgb_model'], models['feature_names']
        )
        fairness_assessor = FairnessAssessor()
        # Load Temporal Fusion Transformer model and dataset config
        try:
            tft_model = torch.load('models/tft_model.pth')
            # Load dataset config or create placeholder
            tft_dataset = None  # This should be replaced with actual dataset loading
        except:
            tft_model = None
            tft_dataset = None
    except Exception as e:
        print(f"Error loading models or resources: {e}")

def prepare_features(features: List[float]) -> np.ndarray:
    arr = np.array(features).reshape(1, -1)
    scaled = models['scaler'].transform(arr)
    return scaled

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_patient(features: PatientFeatures):
    try:
        feature_list = [
            features.age, features.temperature, features.heart_rate, features.respiratory_rate,
            features.oxygen_saturation, features.blood_pressure_systolic, features.blood_pressure_diastolic,
            features.pain_score, features.arrival_encoded, features.consciousness_encoded, features.gender_encoded
        ]
        scaled_features = prepare_features(feature_list)
        xgb_probs = models['xgb_model'].predict_proba(scaled_features)[0]
        lgb_probs = models['lgb_model'].predict_proba(scaled_features)[0]
        ensemble_probs = (xgb_probs + lgb_probs) / 2
        prediction = int(ensemble_probs[1] > 0.5)
        return PredictionResponse(
            prediction=prediction,
            confidence=float(ensemble_probs[1] if prediction == 1 else ensemble_probs[0]),
            model_probs={"xgb": float(xgb_probs[1]), "lgb": float(lgb_probs[1])}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/explain", response_model=ExplanationResponse, tags=["Interpretability"])
async def explain(features: ExplanationRequest):
    try:
        scaled_features = prepare_features(features.features)
        explanations = interpret_engine.generate_comprehensive_explanation(scaled_features[0])
        report = interpret_engine.generate_explanation_report(explanations)
        return ExplanationResponse(
            shap_values=explanations.get('shap_values'),
            lime_explanation=explanations.get('lime_explanation'),
            integrated_gradients=explanations.get('integrated_gradients'),
            explanation_report=report
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretability error: {e}")

@app.post("/fairness", response_model=FairnessResponse, tags=["Fairness"])
async def assess_fairness(request: FairnessRequest):
    try:
        results = fairness_assessor.assess_fairness(
            np.array(request.predictions),
            np.array(request.true_labels),
            {k: np.array(v) for k, v in request.protected_attributes.items()}
        )
        report = generate_fairness_report(results, np.array(request.predictions), np.array(request.true_labels))
        return FairnessResponse(fairness_results=results, fairness_report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness assessment error: {e}")

@app.post("/counterfactual", response_model=CounterfactualResponse, tags=["Counterfactual"])
async def counterfactual(request: CounterfactualRequest):
    try:
        scaled_features = prepare_features(request.features)
        counterfactual_result = generate_counterfactual_explanation(
            scaled_features[0], models['xgb_model'], models['feature_names'],
            target_class=request.target_class, max_changes=request.max_changes
        )
        if 'message' in counterfactual_result:
            return CounterfactualResponse(
                success=False,
                changes=None,
                original_prediction=counterfactual_result.get('original_prediction', []),
                counterfactual_prediction=counterfactual_result.get('counterfactual_prediction', []),
                message=counterfactual_result['message']
            )
        else:
            return CounterfactualResponse(
                success=counterfactual_result.get('success', False),
                changes=counterfactual_result.get('changes', []),
                original_prediction=counterfactual_result.get('original_prediction', []),
                counterfactual_prediction=counterfactual_result.get('counterfactual_prediction', []),
                message=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual error: {e}")

@app.post("/temporal_predict", response_model=TemporalPredictionResponse, tags=["Temporal"])
async def temporal_predict(data: TemporalData = Body(...)):
    """
    Predict triage urgency using Temporal Fusion Transformer on temporal data.

    Expects a list of dicts with time-series features per patient.
    """
    if tft_model is None:
        raise HTTPException(status_code=503, detail="Temporal Fusion Transformer model not loaded")

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data.data)

        # TODO: Preprocess df to match TFT dataset format
        # This includes setting time_idx, group_ids, categorical encodings, etc.

        # For demonstration, assume df is preprocessed and create TimeSeriesDataSet
        if tft_dataset is None:
            raise HTTPException(status_code=503, detail="TFT dataset configuration not available")

        dataset = TimeSeriesDataSet.from_dataset(tft_dataset, df, predict=True, stop_randomization=True)
        dataloader = dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)

        # Predict
        predictions = []
        tft_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                preds = tft_model(x)
                preds = preds.detach().cpu().numpy()
                predictions.extend(preds[:, 0])  # Assuming first quantile or mean prediction

        return TemporalPredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal prediction error: {e}")

@app.post("/debias", response_model=DebiasingResponse, tags=["Debiasing"])
async def debias_data(request: DebiasingRequest):
    """
    Apply fairness debiasing techniques to training data.

    Supports reweighting and adversarial debiasing methods.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)
        protected_attr = np.array(request.protected_attr)

        if request.method == "reweighting":
            from src.fairness_module import FairnessAwareDebiasing
            debiasing = FairnessAwareDebiasing()
            sample_weights = debiasing.reweight_samples(
                X, y, protected_attr, request.target_fairness
            )
            return DebiasingResponse(
                debiased_X=None,
                sample_weights=sample_weights.tolist(),
                message="Sample reweighting applied successfully"
            )

        elif request.method == "adversarial":
            from src.fairness_module import FairnessAwareDebiasing
            debiasing = FairnessAwareDebiasing()
            debiased_X = debiasing.adversarial_debiasing_preprocessing(
                X, protected_attr, lambda_param=0.1
            )
            return DebiasingResponse(
                debiased_X=debiased_X.tolist(),
                sample_weights=None,
                message="Adversarial debiasing applied successfully"
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown debiasing method: {request.method}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debiasing error: {e}")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)
