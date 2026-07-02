# AI MECHANIC ULTIMATE v10.0
# Docker Hub: expertcarcheck/ai-mechanic:latest
# 
# BUILD: docker build -t expertcarcheck/ai-mechanic:latest .
# PUSH:  docker push expertcarcheck/ai-mechanic:latest
#
# Uses RunPod PyTorch base (CUDA 12.1, Ubuntu 22.04)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/cache
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg libsndfile1 imagemagick \
    && rm -rf /var/lib/apt/lists/*

# vLLM FIRST — brings its own compatible torch, avoids version conflict
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    vllm==0.5.5 sentencepiece protobuf

# Remaining ML packages (torchvision/torchaudio auto-match vLLM's torch)
RUN pip install --no-cache-dir \
    torchvision torchaudio \
    transformers==4.44.0 accelerate bitsandbytes \
    scipy numpy pillow opencv-python \
    soundfile librosa \
    trimesh einops timm \
    kokoro onnxruntime-gpu \
    requests beautifulsoup4 \
    runpod

# TripoSR for 3D generation (install from GitHub)
RUN pip install --no-cache-dir git+https://github.com/VAST-AI-Research/TripoSR.git

# Copy handler files
COPY handler.py /workspace/handler.py
COPY obd_database.py /workspace/obd_database.py

# Pre-download small models (speeds up cold start)
RUN python3 -c "
import torch
from transformers import pipeline
print('Preloading Parakeet...')
pipe = pipeline('automatic-speech-recognition', 
    model='nvidia/parakeet-tdt-0.6b-v3',
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16)
print('Parakeet OK')
print('Preloading Kokoro...')
from kokoro import KModel, KPipeline
m = KModel().to('cuda' if torch.cuda.is_available() else 'cpu').eval()
print('Kokoro OK')
" 2>/dev/null || echo "Models will load on first request"

# RunPod serverless entry point
CMD ["python3", "-u", "handler.py"]
