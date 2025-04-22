import io, random
from fastapi.testclient import TestClient
from PIL import Image

try:
    from src.app import app, CLASSES
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.app import app, CLASSES

client = TestClient(app)

def random_image():
    arr = [random.randint(0, 255) for _ in range(32*32*3)]
    img = Image.frombytes("RGB", (32, 32), bytes(arr))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_predict_returns_valid_class():
    buf = random_image()
    files = {"file": ("dummy.png", buf, "image/png")}
    r = client.post("/predict", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] in CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
