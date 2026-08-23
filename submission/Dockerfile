# Submission container for offline inference. Confirm base image + CUDA version against the
# official submission spec before finalizing (login-gated).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY submission/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Baked, offline model weights (populate before build; do NOT download at runtime).
COPY submission/assets ./assets
COPY src ./src
COPY submission/predict.py .

ENV PYTHONPATH=/app/src
# Runs offline; the grader mounts test audio and reads the output CSV.
ENTRYPOINT ["python3", "predict.py"]
