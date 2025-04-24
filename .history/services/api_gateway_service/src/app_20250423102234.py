"""
API Gateway Service
Routes requests to appropriate microservices
"""
import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference-service")

# Prometheus metrics
REQUESTS = Counter('api_gateway_requests_total', 'Total requests to API Gateway', ['path', 'method', 'status'])

# Create FastAPI app
app = FastAPI(title="API Gateway")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for calling other services
@app.on_event("startup")
async def startup_event():
    # Create global HTTP client for reuse
    app.state.http_client = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def shutdown_event():
    # Close HTTP client on shutdown
    await app.state.http_client.aclose()

# Request counting middleware
@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    REQUESTS.labels(
        path=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    return response

# Health check endpoint
@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check for the API gateway"""
    return {
        "status": "ok",
        "service": "api-gateway",
        "version": "1.0.0"
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus to scrape metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Proxy endpoint for inference service
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def call_inference_service(client: httpx.AsyncClient, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the inference service with retry logic"""
    response = await client.post(f"{INFERENCE_SERVICE_URL}{path}", json=data)
    response.raise_for_status()
    return response.json()

@app.post("/predict")
async def predict_proxy(file: UploadFile = File(...)):
    """Proxy endpoint for model prediction"""
    try:
        # Read file content
        content = await file.read()
        
        # Forward to inference service
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, content, file.content_type)}
            response = await client.post(
                f"{INFERENCE_SERVICE_URL}/predict", 
                files=files
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling inference service: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Generic fallback for unimplemented endpoints
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    """Fallback endpoint for future extensibility"""
    return JSONResponse(
        status_code=501,
        content={"error": f"Endpoint /{path} not implemented"}
    ) 