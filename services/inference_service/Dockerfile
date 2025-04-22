FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# install runtime deps only
RUN pip install --no-cache-dir -r requirements.txt

# copy source & checkpoint
COPY src/ ./src
COPY models/ ./models

EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
