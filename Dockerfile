FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY src ./src
COPY frontend ./frontend
COPY config.yaml .

# Supplied CNN checkpoint (tracked in the repo).
COPY models/amharic_ocr_v2.pth ./models/amharic_ocr_v2.pth

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
