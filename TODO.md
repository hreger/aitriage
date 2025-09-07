# ED-AI Triage System Prototype TODO

## Step 1: Set up Python virtual environment
- [x] Create virtual environment in project root
- [x] Activate venv

## Step 2: Install dependencies
- [ ] Install required packages: streamlit, scikit-learn, pandas, numpy, transformers, shap, lime, jupyter
- [ ] Create requirements.txt

## Step 3: Create project structure
- [x] Create directories: src/, data/, models/, notebooks/

## Step 4: Create Jupyter notebook for ML/DL code
- [x] notebooks/triage_model.ipynb: synthetic data generation, model training, explainability, save model

## Step 5: Create Streamlit app
- [x] src/app.py: data entry form, prediction, visualization, simulation dashboard

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
