# Dockerfile for TPU v5e-64 Activation Extraction
#
# This Docker image contains all dependencies for running activation
# extraction on Google Cloud TPU v5e-64
#
# =============================================================================
# BUILD AND PUSH TO ARTIFACT REGISTRY
# =============================================================================
#
# 1. Set your project and region variables:
#    export PROJECT_ID="absolute-axis-470415-g6"
#    export AR_REGION="us-central1"
#    export AR_REPO="arc-agi-us-central1"
#
# 2. Create Artifact Registry repository (if not exists):
#    gcloud artifacts repositories create ${AR_REPO} \
#      --repository-format=docker \
#      --location=${AR_REGION} \
#      --project=${PROJECT_ID} \
#      --description="Docker repository for ARC-AGI activation extraction"
#
# 3. Configure Docker authentication for Artifact Registry:
#    gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev
#
# 4. Build the image:
#    docker build -t ${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction .
#
# 5. Push to Artifact Registry:
#    docker push ${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction
#
# NOTE: Do NOT use gcr.io (Google Container Registry) as it is deprecated.
#       Always use Artifact Registry: ${AR_REGION}-docker.pkg.dev/...
#
# =============================================================================
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

# Copy core model code
COPY kvcache_utils.py /workspace/
COPY qwen2_jax.py /workspace/
COPY qwen2_jax_with_hooks.py /workspace/

# Copy extraction scripts (refactored + legacy)
COPY extract_activations.py /workspace/
COPY extract_activations_arc_v5e64.py /workspace/
COPY extract_activations_fineweb_multihost.py /workspace/

# Copy dataset utilities
COPY convert_hf_to_arc_format.py /workspace/
COPY create_sharded_dataset.py /workspace/
COPY shard_manager.py /workspace/

# Copy core utilities (refactored shared code)
COPY core/ /workspace/core/

# Copy launch script
COPY launch_v5e64.sh /workspace/

# Copy ARC utilities
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
