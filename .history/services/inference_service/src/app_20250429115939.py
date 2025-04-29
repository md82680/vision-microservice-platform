"""
CIFAR-10 Image Classification Service
Provides a REST API for inference with a ResNet18 model
Used as part of the vision-microservice-platform
"""
import io, os
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import logging

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet_cifar10.pth")

# ─── CIFAR‑10 label list ────────────────────────────────────────────────────────
CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ─── Load model once at startup ────────────────────────────────────────────────
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# identical pre‑processing to training
PRE = transforms.Compose([
    transforms.Resize(32),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="CIFAR‑10 ResNet Inference")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok",
            "service": "inference-service",
            "version": "1.0.0",
            "model_type": "resnet18"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request. File: {file.filename}, Content-Type: {file.content_type}")
    
    # Validate content type
    if not file.content_type:
        logger.error("No content type provided")
        raise HTTPException(status_code=400, detail="No content type provided")
    
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        logger.error(f"Unsupported file type: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image."
        )

    try:
        # Read file content
        img_bytes = await file.read()
        if not img_bytes:
            logger.error("Empty file received")
            raise HTTPException(status_code=400, detail="Empty file received")
            
        logger.info(f"Read {len(img_bytes)} bytes from file")
        
        try:
            # Open and convert image
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        logger.info(f"Original image size: {img.size}, Mode: {img.mode}")
        
        # Resize image to 32x32
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        logger.info(f"Resized image size: {img.size}")
        
        # Preprocess image
        batch = PRE(img).unsqueeze(0).to(DEVICE)
        logger.info(f"Preprocessed image shape: {batch.shape}")

        # Make prediction
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[0]
            logger.info(f"Model output shape: {logits.shape}")

        # Get prediction result
        idx = int(probs.argmax())
        result = {
            "predicted_class": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4)
        }
        logger.info(f"Prediction result: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
