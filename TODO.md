# ED-AI Triage System TODO

## Step 1: Set up Python virtual environment
- [x] Create virtual environment in project root
- [x] Activate venv

## Step 2: Install dependencies
- [x] Install required packages: streamlit, scikit-learn, pandas, numpy, transformers, shap, lime, jupyter
- [x] Create requirements.txt with FastAPI, uvicorn

## Step 3: Create project structure
- [x] Create directories: src/, data/, models/, notebooks/

## Step 4: Create Jupyter notebook for ML/DL code
- [x] notebooks/triage_model.ipynb: synthetic data generation, model training, explainability, save model
- [x] notebooks/advanced_triage_model.ipynb: advanced models with ClinicalBERT, ensemble methods

## Step 5: Create Streamlit app
- [x] src/app.py: basic data entry form, prediction, visualization
- [x] src/advanced_app.py: advanced features with ClinicalBERT, interpretability, fairness

## Step 6: Test end-to-end functionality
- [x] Run notebook to generate data and train model
- [x] Run Streamlit app locally
- [x] Test predictions and explainability
- [x] Validate simulation scenarios

## Step 7: Dockerize Application
- [x] Create Dockerfile for containerizing the app
- [x] Create docker-compose.yml for multi-service setup
- [x] Create build and run scripts for Docker
- [ ] Test Docker build and run locally

## Step 8: Create FastAPI Web API Server
- [x] Create src/api.py with FastAPI application structure
- [x] Implement async endpoints:
  - [x] POST /predict: Patient triage prediction with ensemble model
  - [x] POST /explain: SHAP/LIME/Integrated Gradients explanations
  - [x] POST /fairness: Fairness assessment and debiasing
  - [x] POST /counterfactual: Counterfactual explanations
  - [x] GET /health: API health check
- [ ] Add Temporal Fusion Transformer implementation for temporal predictions
- [ ] Implement advanced debiasing strategies (reweighting, adversarial)
- [ ] Add comprehensive error handling and Pydantic validation
- [ ] Test all endpoints and ensure proper async performance
- [ ] Update Docker configuration for FastAPI server
- [ ] Run server on port 8080 with uvicorn
- [ ] Add API documentation with OpenAPI/Swagger
