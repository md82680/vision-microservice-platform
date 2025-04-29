"""
CIFAR-10 Image Classification Service
Provides a REST API for inference with a ResNet18 model
Used as part of the vision-microservice-platform
"""
import io, os
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import logging

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

logger = logging.getLogger(__name__)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok",
            "service": "inference-service",
            "version": "1.0.0",
            "model_type": "resnet18"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request. File: {file.filename}, Content-Type: {file.content_type}")
    
    if file.content_type not in {"image/png", "image/jpeg"}:
        logger.error(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        img_bytes = await file.read()
        logger.info(f"Read {len(img_bytes)} bytes from file")
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        logger.info(f"Image size: {img.size}, Mode: {img.mode}")
        
        # Validate image dimensions
        if img.size[0] < 32 or img.size[1] < 32:
            raise HTTPException(status_code=422, detail="Image dimensions too small")
        
        batch = PRE(img).unsqueeze(0).to(DEVICE)
        logger.info(f"Preprocessed image shape: {batch.shape}")

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[0]
            logger.info(f"Model output shape: {logits.shape}")

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
