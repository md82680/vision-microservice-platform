"""
API Gateway Service
Routes requests to appropriate microservices
"""
import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

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
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8001")
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8002")

# Prometheus metrics
REQUESTS = Counter('api_gateway_requests_total', 'Total requests to API Gateway', ['path', 'method', 'status'])

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create global HTTP client for reuse on startup
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    yield
    # Close HTTP client on shutdown
    await app.state.http_client.aclose()

# Create FastAPI app
app = FastAPI(title="API Gateway", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Log incoming request details
        logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")
        
        # Read file content
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        
        # Forward to inference service
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create a new file-like object for forwarding
                files = {'file': (file.filename, content, file.content_type)}
                
                # Test connection first
                try:
                    health_response = await client.get(f"{INFERENCE_SERVICE_URL}/health")
                    logger.info(f"Inference service health check: {health_response.status_code}")
                    if health_response.status_code != 200:
                        raise HTTPException(status_code=502, detail="Inference service health check failed")
                except httpx.ConnectError as e:
                    logger.error(f"Could not connect to inference service: {str(e)}")
                    raise HTTPException(status_code=502, detail="Could not connect to inference service")
                
                # Make prediction request
                response = await client.post(
                    f"{INFERENCE_SERVICE_URL}/predict",
                    files=files,
                    headers={'Content-Type': 'multipart/form-data'}
                )
                logger.info(f"Inference service response: {response.status_code}")
                
                if response.status_code >= 400:
                    error_detail = response.json().get('detail', 'Unknown error')
                    logger.error(f"Inference service error: {error_detail}")
                    raise HTTPException(status_code=response.status_code, detail=error_detail)
                    
                return response.json()
                
        except httpx.ConnectError as e:
            logger.error(f"Could not connect to inference service: {str(e)}")
            raise HTTPException(status_code=502, detail="Could not connect to inference service")
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get('detail', str(e)) if e.response.content else str(e)
            logger.error(f"Inference service error: {error_detail}")
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.TimeoutException as e:
            logger.error(f"Timeout waiting for inference service: {str(e)}")
            raise HTTPException(status_code=504, detail="Timeout waiting for inference service")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Training Service Endpoints ────────────────────────────────────────────────────

@app.post("/train")
async def train_model(request: Request):
    """Start a new model training job"""
    try:
        # Forward the training request
        json_data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TRAINING_SERVICE_URL}/train",
                json=json_data
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling training service: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a specific training job"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TRAINING_SERVICE_URL}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling training service: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TRAINING_SERVICE_URL}/jobs")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling training service: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
async def list_models():
    """List all available trained models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TRAINING_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling training service: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a specific model"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{TRAINING_SERVICE_URL}/models/{model_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Error calling training service: {str(e)}")
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