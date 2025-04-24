from fastapi.testclient import TestClient
try:
    from src.app import app
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.app import app

client = TestClient(app)

def test_health_ok():
    """Test that the health endpoint returns OK status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "api-gateway"
    assert "version" in data 