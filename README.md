# 🏥 ED-AI Triage System

An intelligent Emergency Department triage system powered by machine learning and deep learning models to assist healthcare professionals in prioritizing patient care based on clinical data and vital signs.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- **Patient Triage Prediction**: Real-time assessment of patient urgency using ensemble ML models
- **Interactive Dashboard**: Visual analytics for ED operations and patient flow
- **Triage Simulation**: Predictive modeling for ED capacity planning
- **Multi-modal Data Integration**: Combines structured clinical data with text-based chief complaints

### Advanced Features
- **Model Interpretability**: SHAP, LIME, and Integrated Gradients explanations
- **Fairness Assessment**: Bias detection and mitigation strategies
- **Counterfactual Analysis**: "What-if" scenario exploration
- **Temporal Predictions**: Time-series forecasting for patient outcomes

### Technical Features
- **Ensemble Models**: XGBoost, LightGBM, and ClinicalBERT integration
- **RESTful API**: FastAPI-based endpoints for seamless integration
- **Containerized Deployment**: Docker and docker-compose support
- **Experiment Tracking**: MLflow integration for model versioning

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Docker (optional, for containerized deployment)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ed-ai-triage.git
   cd ed-ai-triage
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required data**
   Place the following CSV files in the `data/` directory:
   - `triage.csv`
   - `vitalsign.csv`
   - `diagnosis.csv`
   - `edstays.csv`
   - `medrecon.csv`
   - `pyxis.csv`

## 💻 Usage

### Running the Streamlit Application

```bash
streamlit run src/app.py
```

Access the application at `http://localhost:8501`

### Running the FastAPI Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload
```

Access the API documentation at `http://localhost:8080/docs`

### Training Models

Run the Jupyter notebooks to train and evaluate models:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `triage_model.ipynb`: Basic ML model training
- `advanced_triage_model.ipynb`: Advanced models with ClinicalBERT
- `xgb_triage_model.ipynb`: XGBoost-specific implementation

## 📁 Project Structure

```
ed-ai-triage/
├── data/                    # Clinical datasets
│   ├── triage.csv
│   ├── vitalsign.csv
│   ├── diagnosis.csv
│   ├── edstays.csv
│   ├── medrecon.csv
│   └── pyxis.csv
├── models/                  # Trained model artifacts
│   ├── xgb_model.joblib
│   ├── advanced_scaler.joblib
│   └── structured_features.joblib
├── notebooks/               # Jupyter notebooks for model development
│   ├── triage_model.ipynb
│   ├── advanced_triage_model.ipynb
│   └── xgb_triage_model.ipynb
├── src/                     # Source code
│   ├── app.py              # Streamlit application
│   ├── advanced_app.py     # Advanced Streamlit features
│   ├── api.py              # FastAPI application
│   ├── fairness_module.py  # Fairness assessment utilities
│   └── interpretability_module.py
├── tests/                   # Unit and integration tests
│   └── test_api.py
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Multi-service Docker setup
├── TODO.md               # Development roadmap
└── README.md             # Project documentation
```

## 🤖 Model Details

### Supported Models
- **XGBoost**: Gradient boosting for structured clinical features
- **LightGBM**: Efficient gradient boosting with GPU support
- **ClinicalBERT**: Transformer-based model for text analysis
- **Ensemble Methods**: Combined predictions for improved accuracy

### Model Files
- `models/xgb_model.joblib`: XGBoost classifier
- `models/advanced_scaler.joblib`: Feature scaler
- `models/structured_features.joblib`: Feature engineering pipeline

### Training Parameters
- Cross-validation with 5-fold splits
- Hyperparameter optimization using grid search
- Feature engineering with clinical thresholds
- Model evaluation using AUC, precision, recall metrics

## 🔌 API Endpoints

### Core Endpoints

#### POST `/predict`
Predict patient triage urgency
```json
{
  "age": 45,
  "temperature": 38.5,
  "heart_rate": 110,
  "respiratory_rate": 22,
  "oxygen_saturation": 92,
  "blood_pressure_systolic": 85,
  "blood_pressure_diastolic": 60,
  "pain_score": 8,
  "arrival_mode": "Ambulance",
  "consciousness": "Alert",
  "chief_complaint": "Severe chest pain"
}
```

#### POST `/explain`
Generate model explanations using SHAP/LIME
```json
{
  "patient_data": {...},
  "explanation_method": "shap"
}
```

#### POST `/fairness`
Assess model fairness across demographic groups
```json
{
  "predictions": [...],
  "sensitive_attributes": [...]
}
```

#### POST `/counterfactual`
Generate counterfactual explanations
```json
{
  "patient_data": {...},
  "target_outcome": "non-urgent"
}
```

#### GET `/health`
API health check endpoint

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest --cov=src tests/
```

## 🐳 Docker

### Build and Run

```bash
# Build the Docker image
docker build -t ed-ai-triage .

# Run the container
docker run -p 8080:8080 -p 8501:8501 ed-ai-triage
```

### Using Docker Compose

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Contact the development team

---

**Disclaimer**: This system is designed to assist healthcare professionals and should not replace clinical judgment. Always consult with qualified medical personnel for patient care decisions.
