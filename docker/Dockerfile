# syntax=docker/dockerfile:1.4
# Base image: NVIDIA CUDA 12.1 with cuDNN, Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace

# Install Python dependencies first (better layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

# Copy the rest of the project
COPY . /workspace/

# Expose Jupyter port
EXPOSE 8888

# Default: launch JupyterLab without auth (development only)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
