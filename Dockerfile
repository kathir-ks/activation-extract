# Dockerfile for TPU v5e-64 Activation Extraction
#
# This Docker image contains all dependencies for running activation
# extraction on Google Cloud TPU v5e-64
#

FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=tpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python dependencies
# Note: JAX TPU version is specific to TPU v5e
# Install JAX, jaxlib, and flax for TPU (libtpu will be loaded from host)
RUN pip install --no-cache-dir \
    'jax[tpu]' \
    flax \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    fsspec \
    gcsfs \
    tqdm \
    pillow \
    matplotlib \
    termcolor

# Copy application code
COPY kvcache_utils.py /workspace/
COPY qwen2_jax.py /workspace/
COPY qwen2_jax_with_hooks.py /workspace/
COPY extract_activations_arc_v5e64.py /workspace/
COPY extract_activations_fineweb_multihost.py /workspace/
COPY convert_hf_to_arc_format.py /workspace/
COPY launch_v5e64.sh /workspace/

# Copy ARC utilities (if available)
COPY arc24/ /workspace/arc24/

# Create cache directory for HuggingFace models
# This should be mounted to a volume with sufficient space
RUN mkdir -p /cache/huggingface
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Make launch script executable
RUN chmod +x /workspace/launch_v5e64.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
