import io, os
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet_cifar10.pth")

# ─── CIFAR‑10 label list ────────────────────────────────────────────────────────
CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ─── Load model once at startup ────────────────────────────────────────────────
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# identical pre‑processing to training
PRE = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="CIFAR‑10 ResNet Inference")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok",
            "service": "inference-service",
            }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    batch = PRE(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]

    idx = int(probs.argmax())
    return {
        "predicted_class": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4)
    }
