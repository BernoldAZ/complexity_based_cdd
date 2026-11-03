# Base image with CUDA 11.2 and cuDNN 8
FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.9, pip, and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install Jupyter
RUN python3 -m pip install jupyterlab notebook

# Expose notebook port
EXPOSE 8887

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8887", "--allow-root", "--no-browser"]

# docker build -t my-jupyter-image .
#
# PowerShell:
# docker run -it --rm -p 127.0.0.1:8887:8887 -v "${PWD}:/app" --gpus all  my-jupyter-image
#
# bash/Git Bash:
# docker run -it --rm -p 127.0.0.1:8887:8887 -v "$(pwd):/app" --gpus all  my-jupyter-image

# If no gpu is available run (PowerShell):
# docker run -it --rm -p 127.0.0.1:8887:8887 -v "${PWD}:/app" my-jupyter-image
#
# If no gpu is available run (bash/Git Bash):
# docker run -it --rm -p 127.0.0.1:8887:8887 -v "$(pwd):/app" my-jupyter-image
