"""
Tests for health check endpoint
"""
import pytest
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint returns 200 and correct service info"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "training-service"
    assert "version" in data 