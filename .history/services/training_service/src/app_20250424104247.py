"""
Training Service API
Provides endpoints for training, registering, and managing ML models
"""
import os
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .trainer import train_cifar10_model
from .model_registry import register_model, list_models, get_model_details

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models/trained")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Track active training jobs
active_jobs = {}

# Prometheus metrics
TRAINING_JOBS = Counter('training_jobs_total', 'Total number of training jobs started')
ACTIVE_JOBS = Gauge('active_training_jobs', 'Number of currently running training jobs')
TRAINING_DURATION = Gauge('training_duration_seconds', 'Duration of training jobs in seconds', ['model_type', 'status'])

# Input models for API
class TrainingRequest(BaseModel):
    model_type: str = "resnet18"
    dataset: str = "cifar10"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    description: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    model_type: str
    start_time: str
    end_time: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Training service starting up")
    yield
    # Shutdown
    logger.info("Training service shutting down")

# Create FastAPI app
app = FastAPI(title="ML Training Service", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check for the training service"""
    return {
        "status": "ok",
        "service": "training-service",
        "version": "1.0.0"
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus to scrape metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def run_training_job(job_id: str, params: TrainingRequest):
    """Background task to run a training job"""
    ACTIVE_JOBS.inc()
    
    start_time = datetime.now()
    active_jobs[job_id]["status"] = "running"
    active_jobs[job_id]["start_time"] = start_time.isoformat()
    
    try:
        logger.info(f"Starting training job {job_id} for {params.model_type} on {params.dataset}")
        
        # Run the training process
        model_path, metrics = await train_cifar10_model(
            model_type=params.model_type,
            epochs=params.epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            save_path=f"{MODEL_SAVE_PATH}/{job_id}_{params.model_type}.pth"
        )
        
        # Register the model
        register_model(
            model_path=model_path,
            model_type=params.model_type,
            dataset=params.dataset,
            metrics=metrics,
            job_id=job_id,
            description=params.description
        )
        
        # Update job status
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        TRAINING_DURATION.labels(model_type=params.model_type, status="success").set(duration)
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["end_time"] = end_time.isoformat()
        active_jobs[job_id]["metrics"] = metrics
        
        logger.info(f"Training job {job_id} completed successfully")
    
    except Exception as e:
        # Update job status on error
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        TRAINING_DURATION.labels(model_type=params.model_type, status="failed").set(duration)
        
        error_msg = str(e)
        logger.error(f"Training job {job_id} failed: {error_msg}")
        
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["end_time"] = end_time.isoformat()
        active_jobs[job_id]["error"] = error_msg
    
    finally:
        ACTIVE_JOBS.dec()

@app.post("/train", response_model=JobStatus)
async def start_training(params: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "model_type": params.model_type,
        "start_time": datetime.now().isoformat(),
    }
    
    # Increment metrics
    TRAINING_JOBS.inc()
    
    # Start the training job in the background
    background_tasks.add_task(run_training_job, job_id, params)
    
    return JobStatus(**active_jobs[job_id])

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific training job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatus(**active_jobs[job_id])

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all training jobs"""
    return [JobStatus(**job_info) for job_info in active_jobs.values()]

@app.get("/models")
async def get_models():
    """List all available trained models"""
    return list_models()

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a specific model"""
    model_details = get_model_details(model_id)
    if not model_details:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model_details 