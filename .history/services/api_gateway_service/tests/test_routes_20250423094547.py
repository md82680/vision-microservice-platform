import io
import unittest.mock
from fastapi.testclient import TestClient
import httpx

try:
    from src.app import app
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.app import app

client = TestClient(app)

# Mock for httpx client responses
class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Error", request=None, response=self)

def test_predict_proxy_success():
    """Test that the predict endpoint correctly proxies to the inference service"""
    # Create a test image
    test_img = io.BytesIO(b"fake image content")
    
    # Mock the httpx client post method
    with unittest.mock.patch("httpx.AsyncClient.post") as mock_post:
        # Setup the mock to return a successful response
        mock_response = MockResponse(200, {"predicted_class": "dog", "confidence": 0.95})
        mock_post.return_value = mock_response
        
        # Make request to our API
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_img, "image/jpeg")}
        )
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == {"predicted_class": "dog", "confidence": 0.95}
        
        # Verify the call was made correctly to the inference service
        mock_post.assert_called_once()
        
def test_predict_proxy_error():
    """Test that errors from the inference service are properly handled"""
    # Create a test image
    test_img = io.BytesIO(b"fake image content")
    
    # Mock the httpx client post method
    with unittest.mock.patch("httpx.AsyncClient.post") as mock_post:
        # Setup the mock to return an error response
        mock_response = MockResponse(400, {"detail": "Bad request"})
        mock_post.return_value = mock_response
        mock_post.side_effect = httpx.HTTPStatusError(
            "Bad request", 
            request=httpx.Request("POST", "http://test"), 
            response=mock_response
        )
        
        # Make request to our API
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_img, "image/jpeg")}
        )
        
        # Assertions
        assert response.status_code == 400
        
def test_not_implemented_endpoint():
    """Test that unimplemented endpoints return 501 Not Implemented"""
    response = client.get("/some-random-endpoint")
    assert response.status_code == 501 