@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "inference-service",
        "version": "1.0.0",
        "model_type": "resnet18"
    } 