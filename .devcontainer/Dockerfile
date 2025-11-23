# ARG for install type: "cpu" or "cuda" (default: cpu)
ARG INSTALL_TYPE=cpu

# Base image: Use python:3.10-slim for CPU
FROM python:3.10-slim

# Install system dependencies and Zsh in one layer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libboost-all-dev \
    libcairo2-dev \
    libeigen3-dev \
    libffi-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libx11-dev \
    libxext-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Create venv and install dependencies
RUN python -m venv gflownet-env && \
    . gflownet-env/bin/activate && \
    python -m pip install --upgrade pip --no-cache-dir && \
    if [ "$INSTALL_TYPE" = "cuda" ]; then \
        python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118 && \
        python -m pip install --no-cache-dir --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html && \
        python -m pip install --no-cache-dir "jax[cuda12]" optax jaxtyping equinox; \
    else \
        python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
        python -m pip install --no-cache-dir --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html && \
        python -m pip install --no-cache-dir jax optax jaxtyping equinox; \
    fi && \
    python -m pip install --no-cache-dir -e .[dev,jax]

RUN echo '#!/bin/bash\nsource gflownet-env/bin/activate\nexec "$@"' > /workspace/start.sh && chmod +x /workspace/start.sh

CMD ["/workspace/start.sh", "bash"]