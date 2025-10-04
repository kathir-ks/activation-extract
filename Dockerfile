# Dockerfile for ARC-AGI Distributed Inference with TPU Support
# Optimized for running on Google Cloud TPU VMs (v4, v5e)

FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAX_PLATFORMS=tpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (for cloud storage)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Install JAX with TPU support
RUN pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install additional dependencies for the pipeline
RUN pip install \
    transformers>=4.30.0 \
    datasets>=2.14.0 \
    google-cloud-storage>=2.10.0 \
    tqdm>=4.65.0 \
    numpy>=1.24.0 \
    jinja2>=3.1.0 \
    termcolor>=2.3.0

# Copy the entire pipeline code
COPY . /workspace/

# Make scripts executable
RUN chmod +x /workspace/*.sh /workspace/*.py

# Set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories for outputs
RUN mkdir -p /workspace/outputs /workspace/activations /workspace/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import jax; print(jax.devices())" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "distributed_inference_with_activations.py", "--help"]
