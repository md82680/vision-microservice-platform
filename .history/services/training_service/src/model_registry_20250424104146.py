"""
Model Registry
Handles the registration and tracking of trained models
"""
import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
REGISTRY_PATH = os.getenv("REGISTRY_PATH", "./models/registry")
MODELS_DB_PATH = os.path.join(REGISTRY_PATH, "models_db.json")

# Create registry directory
os.makedirs(REGISTRY_PATH, exist_ok=True)

def _initialize_db():
    """Initialize the models database if it doesn't exist"""
    if not os.path.exists(MODELS_DB_PATH):
        with open(MODELS_DB_PATH, 'w') as f:
            json.dump({"models": []}, f)

def _load_models_db() -> Dict[str, List[Dict[str, Any]]]:
    """Load the models database"""
    _initialize_db()
    with open(MODELS_DB_PATH, 'r') as f:
        return json.load(f)

def _save_models_db(db: Dict[str, List[Dict[str, Any]]]):
    """Save the models database"""
    with open(MODELS_DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)

def register_model(
    model_path: str,
    model_type: str,
    dataset: str,
    metrics: Dict[str, Any],
    job_id: str,
    description: Optional[str] = None
) -> str:
    """
    Register a trained model in the registry
    
    Args:
        model_path: Path to the model file
        model_type: Type of the model (e.g., "resnet18")
        dataset: Dataset used for training
        metrics: Training metrics
        job_id: ID of the training job
        description: Optional description of the model
        
    Returns:
        The model ID
    """
    # Load models database
    db = _load_models_db()
    
    # Generate model ID (using timestamp)
    model_id = f"{model_type}_{dataset}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create model registry entry
    model_entry = {
        "model_id": model_id,
        "model_type": model_type,
        "dataset": dataset,
        "training_job_id": job_id,
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "description": description,
        "status": "available",
        "path": f"{REGISTRY_PATH}/{model_id}.pth"
    }
    
    # Copy model file to registry
    try:
        shutil.copy(model_path, model_entry["path"])
        logger.info(f"Model {model_id} copied to registry at {model_entry['path']}")
    except Exception as e:
        logger.error(f"Failed to copy model to registry: {str(e)}")
        model_entry["status"] = "failed"
        model_entry["error"] = str(e)
    
    # Add model to database
    db["models"].append(model_entry)
    _save_models_db(db)
    
    logger.info(f"Model {model_id} registered in the registry")
    return model_id

def list_models() -> List[Dict[str, Any]]:
    """
    List all models in the registry
    
    Returns:
        List of model metadata
    """
    db = _load_models_db()
    
    # Sort models by creation date (newest first)
    models = sorted(
        db["models"], 
        key=lambda m: m.get("created_at", ""), 
        reverse=True
    )
    
    # Return simplified model info (without full metrics)
    return [
        {
            "model_id": m["model_id"],
            "model_type": m["model_type"],
            "dataset": m["dataset"],
            "created_at": m["created_at"],
            "status": m["status"],
            "accuracy": m.get("metrics", {}).get("best_val_acc", None),
            "description": m.get("description", "")
        }
        for m in models
    ]

def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model
    
    Args:
        model_id: ID of the model to retrieve
        
    Returns:
        Model metadata or None if not found
    """
    db = _load_models_db()
    
    # Find the model with the given ID
    for model in db["models"]:
        if model["model_id"] == model_id:
            return model
    
    return None

def delete_model(model_id: str) -> bool:
    """
    Delete a model from the registry
    
    Args:
        model_id: ID of the model to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    db = _load_models_db()
    
    # Find the model with the given ID
    for i, model in enumerate(db["models"]):
        if model["model_id"] == model_id:
            # Delete model file
            try:
                if os.path.exists(model["path"]):
                    os.remove(model["path"])
            except Exception as e:
                logger.error(f"Failed to delete model file: {str(e)}")
                return False
            
            # Remove from database
            db["models"].pop(i)
            _save_models_db(db)
            logger.info(f"Model {model_id} deleted from registry")
            return True
    
    logger.warning(f"Model {model_id} not found in registry")
    return False 