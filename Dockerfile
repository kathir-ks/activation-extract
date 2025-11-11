# Dockerfile for TPU v5e-64 Activation Extraction
#
# This Docker image contains all dependencies for running activation
# extraction on Google Cloud TPU v5e-64
#

FROM python:3.10-slim

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
RUN pip install --no-cache-dir \
    "jax[tpu]>=0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    fsspec \
    gcsfs \
    tqdm \
    numpy \
    && pip install --no-cache-dir \
    # Additional dependencies for ARC
    pillow \
    matplotlib

# Copy application code
COPY qwen2_jax.py /workspace/
COPY qwen2_jax_with_hooks.py /workspace/
COPY extract_activations_arc_v5e64.py /workspace/
COPY extract_activations_fineweb_multihost.py /workspace/
COPY convert_hf_to_arc_format.py /workspace/
COPY launch_v5e64.sh /workspace/

# Copy ARC utilities (if available)
COPY arc24/ /workspace/arc24/

# Make launch script executable
RUN chmod +x /workspace/launch_v5e64.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
