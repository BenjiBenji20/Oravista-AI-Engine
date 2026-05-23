FROM python:3.12-slim

# Prevent Python from writing pyc files to disk and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (Updated for Debian Trixie compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and filter out Windows-specific libraries
COPY requirements.txt .
RUN grep -vE "pywin32|pywinpty|pypiwin32|wmi" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Create the explicit target directory structure matching the service logic
RUN mkdir -p src/weights

# --- DOWNLOAD HEAVY BINARIES FROM GITHUB RELEASES ---
RUN curl -L -o src/weights/oravista_dental_image_diagnosis_v1.onnx \
    "https://github.com/BenjiBenji20/Oravista-AI-Engine/releases/download/v1.0.0-weights/oravista_dental_image_diagnosis_v1.onnx"

RUN curl -L -o src/weights/anchors.npy \
    "https://github.com/BenjiBenji20/Oravista-AI-Engine/releases/download/v1.0.0-weights/anchors.npy"

# Copy the rest of your source code into the container image workspace
COPY . .

ENV PORT=8080
EXPOSE 8080

# Enforces JSON array syntax to handle OS termination signals properly
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT --log-level debug"]