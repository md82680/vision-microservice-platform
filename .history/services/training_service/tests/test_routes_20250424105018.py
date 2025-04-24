"""
Tests for training service API routes
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

@patch("src.app.run_training_job")
def test_start_training_endpoint(mock_run_training):
    """Test training job creation endpoint"""
    # Set up mock
    mock_run_training.return_value = None
    
    # Test request
    payload = {
        "model_type": "resnet18",
        "dataset": "cifar10",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "description": "Test training job"
    }
    
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["model_type"] == "resnet18"

def test_list_jobs_endpoint():
    """Test listing jobs endpoint"""
    response = client.get("/jobs")
    assert response.status_code == 200
    
    # Response should be a list
    data = response.json()
    assert isinstance(data, list)

@patch("src.app.get_model_details")
def test_get_model_endpoint(mock_get_model):
    """Test getting model details endpoint"""
    # Mock the model details
    mock_model = {
        "model_id": "resnet18_cifar10_20230501120000",
        "model_type": "resnet18",
        "dataset": "cifar10",
        "status": "available",
        "created_at": "2023-05-01T12:00:00",
        "metrics": {"best_val_acc": 0.85}
    }
    mock_get_model.return_value = mock_model
    
    response = client.get("/models/resnet18_cifar10_20230501120000")
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_id"] == "resnet18_cifar10_20230501120000"
    assert data["model_type"] == "resnet18"

@patch("src.app.get_model_details")
def test_get_nonexistent_model(mock_get_model):
    """Test getting a model that doesn't exist"""
    mock_get_model.return_value = None
    
    response = client.get("/models/nonexistent_model")
    assert response.status_code == 404 