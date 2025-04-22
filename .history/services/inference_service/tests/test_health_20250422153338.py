from fastapi.testclient import TestClient
try:
    from src.app import app
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.app import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    response = r.json()
    assert response["status"] == "ok"
    assert "service" in response
    assert "version" in response
    assert "model_type" in response
    assert response["model_type"] == "resnet18"
