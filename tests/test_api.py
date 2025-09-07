import pytest
from fastapi.testclient import TestClient
import numpy as np
from src.api import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_predict_endpoint(self):
        """Test patient prediction endpoint"""
        test_data = {
            "age": 45,
            "temperature": 98.6,
            "heart_rate": 80,
            "respiratory_rate": 16,
            "oxygen_saturation": 98,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "pain_score": 3,
            "arrival_encoded": 0,
            "consciousness_encoded": 0,
            "gender_encoded": 1
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_probs" in data
        assert isinstance(data["prediction"], int)
        assert 0 <= data["confidence"] <= 1

    def test_predict_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "temperature": 98.6,
            "heart_rate": 80,
            "respiratory_rate": 16,
            "oxygen_saturation": 98,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "pain_score": 3,
            "arrival_encoded": 0,
            "consciousness_encoded": 0,
            "gender_encoded": 1
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_explain_endpoint(self):
        """Test interpretability endpoint"""
        test_data = {
            "features": [45.0, 98.6, 80.0, 16.0, 98.0, 120.0, 80.0, 3.0, 0.0, 0.0, 1.0]
        }

        response = client.post("/explain", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "shap_values" in data
        assert "lime_explanation" in data
        assert "integrated_gradients" in data
        assert "explanation_report" in data

    def test_fairness_endpoint(self):
        """Test fairness assessment endpoint"""
        test_data = {
            "predictions": [0, 1, 0, 1, 0],
            "true_labels": [0, 1, 0, 0, 1],
            "protected_attributes": {
                "gender": [0, 1, 0, 1, 0],
                "age_group": [0, 1, 1, 0, 1]
            }
        }

        response = client.post("/fairness", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "fairness_results" in data
        assert "fairness_report" in data
        assert isinstance(data["fairness_results"], dict)
        assert isinstance(data["fairness_report"], str)

    def test_counterfactual_endpoint(self):
        """Test counterfactual explanation endpoint"""
        test_data = {
            "features": [45.0, 98.6, 80.0, 16.0, 98.0, 120.0, 80.0, 3.0, 0.0, 0.0, 1.0],
            "target_class": 1,
            "max_changes": 3
        }

        response = client.post("/counterfactual", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "changes" in data
        assert "original_prediction" in data
        assert "counterfactual_prediction" in data
        assert "message" in data

    def test_temporal_predict_endpoint(self):
        """Test temporal prediction endpoint"""
        test_data = {
            "data": [
                {
                    "time_idx": 0,
                    "group_id": "patient_1",
                    "age": 45.0,
                    "temperature": 98.6,
                    "heart_rate": 80.0,
                    "respiratory_rate": 16.0,
                    "oxygen_saturation": 98.0,
                    "blood_pressure_systolic": 120.0,
                    "blood_pressure_diastolic": 80.0,
                    "pain_score": 3.0,
                    "arrival_encoded": 0.0,
                    "consciousness_encoded": 0.0,
                    "gender_encoded": 1.0
                }
            ]
        }

        response = client.post("/temporal_predict", json=test_data)
        # This might return 503 if TFT model is not loaded, which is expected
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert isinstance(data["predictions"], list)

    def test_debias_endpoint_reweighting(self):
        """Test debiasing endpoint with reweighting method"""
        test_data = {
            "X": [
                [45.0, 98.6, 80.0, 16.0, 98.0, 120.0, 80.0, 3.0, 0.0, 0.0, 1.0],
                [30.0, 99.0, 90.0, 18.0, 96.0, 130.0, 85.0, 5.0, 1.0, 0.0, 0.0]
            ],
            "y": [0, 1],
            "protected_attr": [0, 1],
            "method": "reweighting",
            "target_fairness": 0.8
        }

        response = client.post("/debias", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "debiased_X" in data
        assert "sample_weights" in data
        assert "message" in data
        assert data["sample_weights"] is not None
        assert data["message"] == "Sample reweighting applied successfully"

    def test_debias_endpoint_adversarial(self):
        """Test debiasing endpoint with adversarial method"""
        test_data = {
            "X": [
                [45.0, 98.6, 80.0, 16.0, 98.0, 120.0, 80.0, 3.0, 0.0, 0.0, 1.0],
                [30.0, 99.0, 90.0, 18.0, 96.0, 130.0, 85.0, 5.0, 1.0, 0.0, 0.0]
            ],
            "y": [0, 1],
            "protected_attr": [0, 1],
            "method": "adversarial",
            "target_fairness": 0.8
        }

        response = client.post("/debias", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "debiased_X" in data
        assert "sample_weights" in data
        assert "message" in data
        assert data["debiased_X"] is not None
        assert data["message"] == "Adversarial debiasing applied successfully"

    def test_debias_invalid_method(self):
        """Test debiasing endpoint with invalid method"""
        test_data = {
            "X": [[45.0, 98.6, 80.0, 16.0, 98.0, 120.0, 80.0, 3.0, 0.0, 0.0, 1.0]],
            "y": [0],
            "protected_attr": [0],
            "method": "invalid_method",
            "target_fairness": 0.8
        }

        response = client.post("/debias", json=test_data)
        assert response.status_code == 400

    def test_performance_benchmark(self):
        """Benchmark API performance"""
        import time

        test_data = {
            "age": 45,
            "temperature": 98.6,
            "heart_rate": 80,
            "respiratory_rate": 16,
            "oxygen_saturation": 98,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "pain_score": 3,
            "arrival_encoded": 0,
            "consciousness_encoded": 0,
            "gender_encoded": 1
        }

        # Test response time for 10 requests
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = client.post("/predict", json=test_data)
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)

        # Assert reasonable performance (adjust thresholds as needed)
        assert avg_response_time < 2.0  # Average response time < 2 seconds
        assert max_response_time < 5.0  # Max response time < 5 seconds

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading

        test_data = {
            "age": 45,
            "temperature": 98.6,
            "heart_rate": 80,
            "respiratory_rate": 16,
            "oxygen_saturation": 98,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "pain_score": 3,
            "arrival_encoded": 0,
            "consciousness_encoded": 0,
            "gender_encoded": 1
        }

        results = []

        def make_request():
            response = client.post("/predict", json=test_data)
            results.append(response.status_code)
            return response

        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            concurrent.futures.wait(futures)

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 20

if __name__ == "__main__":
    pytest.main([__file__])
