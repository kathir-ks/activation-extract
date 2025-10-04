# Documentation Index

Complete guide to distributed inference with activation extraction on TPU pods.

## 📚 Quick Navigation

### Getting Started
- **[README.md](README.md)** - Main overview and introduction
- **[QUICKSTART.md](QUICKSTART.md)** - Basic usage guide for single-host TPU
- **[QUICK_START_EXTRACTION.md](QUICK_START_EXTRACTION.md)** - Quick guide for activation extraction

### Docker Setup
- **[DOCKER_MULTIHOST.md](DOCKER_MULTIHOST.md)** - 📦 **Complete multi-host TPU Docker guide** ⭐
- **[MULTIHOST_QUICKREF.md](MULTIHOST_QUICKREF.md)** - Quick reference commands for multi-host
- **[Dockerfile](Dockerfile)** - Docker image definition
- **[docker-compose.yml](docker-compose.yml)** - Multi-service orchestration
- **[run-docker.sh](run-docker.sh)** - Helper script for Docker operations

### Activation Extraction
- **[README_ACTIVATION_EXTRACTION.md](README_ACTIVATION_EXTRACTION.md)** - Complete activation extraction guide
- **[ACTIVATION_EXTRACTION_SUMMARY.md](ACTIVATION_EXTRACTION_SUMMARY.md)** - Implementation summary
- **[LAYERS_CONFIG.md](LAYERS_CONFIG.md)** - Layer extraction configuration (10-23)
- **[example_activation_extraction.py](example_activation_extraction.py)** - Usage examples

### Architecture & Implementation
- **[README_DISTRIBUTED_INFERENCE.md](README_DISTRIBUTED_INFERENCE.md)** - Distributed inference architecture
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was built and how
- **[FILES_OVERVIEW.txt](FILES_OVERVIEW.txt)** - Complete file listing

### Testing
- **[README_TESTS.md](README_TESTS.md)** - Testing documentation
- **[TEST_SUMMARY.md](TEST_SUMMARY.md)** - Test coverage summary

---

## 🎯 Use Case Guides

### Single-Host TPU (v5e-64)

**Goal**: Run inference on a single TPU pod

**Documentation**:
1. [QUICKSTART.md](QUICKSTART.md) - Setup and basic usage
2. [QUICK_START_EXTRACTION.md](QUICK_START_EXTRACTION.md) - With activation extraction

**Commands**:
```bash
# Build Docker
./run-docker.sh build

# Transform dataset
./run-docker.sh transform

# Run inference
./run-docker.sh distributed
```

---

### Multi-Host TPU (v5e-256, v4-512, etc.)

**Goal**: Run distributed inference across multiple TPU hosts

**Documentation**:
1. 📦 **[DOCKER_MULTIHOST.md](DOCKER_MULTIHOST.md)** - Complete step-by-step guide ⭐
2. [MULTIHOST_QUICKREF.md](MULTIHOST_QUICKREF.md) - Quick command reference

**Key Steps**:
1. Create multi-host TPU pod
2. Install Docker on all hosts
3. Build image on all hosts
4. Share data via GCS
5. Run synchronized inference

---

### Activation Extraction for SAE Training

**Goal**: Extract layer activations for Sparse Autoencoder training

**Documentation**:
1. [README_ACTIVATION_EXTRACTION.md](README_ACTIVATION_EXTRACTION.md) - Complete guide
2. [LAYERS_CONFIG.md](LAYERS_CONFIG.md) - Layer configuration (extracting 10-23)
3. [example_activation_extraction.py](example_activation_extraction.py) - Code examples

**What Gets Extracted**:
- Layers: 10-23 (14 layers)
- Hidden dim: 896
- Storage: ~2.52 GB for 50k samples
- Format: Pickle files per layer per batch

---

### Dataset Transformation

**Goal**: Convert HuggingFace dataset to ARC-AGI format

**File**: [transform_hf_to_arc.py](transform_hf_to_arc.py)

**Command**:
```bash
python transform_hf_to_arc.py \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_dir ./arc_data \
  --max_samples 50000
```

**Output**:
- `arc_format_train.json` - Tasks in ARC format
- `test_outputs_train.json` - Ground truth outputs

---

### Testing & Validation

**Goal**: Verify implementation correctness

**Documentation**: [README_TESTS.md](README_TESTS.md)

**Run Tests**:
```bash
# All tests
python run_tests.py

# Specific category
python run_tests.py --category inference
python run_tests.py --category pipeline
python run_tests.py --category quick
```

---

## 📁 File Organization

### Core Implementation

| File | Description |
|------|-------------|
| `qwen2_jax.py` | Base Qwen model in JAX |
| `qwen2_jax_with_hooks.py` | Model with activation extraction hooks |
| `distributed_inference_with_activations.py` | Distributed TPU inference |
| `simple_extraction_inference.py` | Sequential inference with extraction |
| `arc_inference_jax.py` | ARC-AGI inference pipeline |
| `transform_hf_to_arc.py` | Dataset transformation |

### ARC-AGI Components

| File | Description |
|------|-------------|
| `arc24/encoders.py` | Grid encoding/decoding |
| `arc24/prompting.py` | Prompt generation |
| `arc24/data_augmentation.py` | Data augmentation |
| `arc24/utils.py` | Utility functions |

### Docker & Deployment

| File | Description |
|------|-------------|
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Multi-service setup |
| `run-docker.sh` | Docker helper script |
| `docker-entrypoint.sh` | Container initialization |
| `.dockerignore` | Build exclusions |

### Testing

| File | Description |
|------|-------------|
| `test_arc_inference.py` | ARC inference tests (20 tests) |
| `test_pipeline.py` | Pipeline tests (16 tests) |
| `test_quick_smoke.py` | Quick smoke tests (5 tests) |
| `run_tests.py` | Test runner |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | Main overview |
| `QUICKSTART.md` | Quick start guide |
| `DOCKER_MULTIHOST.md` | Multi-host Docker guide ⭐ |
| `README_ACTIVATION_EXTRACTION.md` | Activation extraction guide |
| `README_DISTRIBUTED_INFERENCE.md` | Distributed inference details |
| `LAYERS_CONFIG.md` | Layer configuration |
| `MULTIHOST_QUICKREF.md` | Multi-host command reference |
| This file | Documentation index |

### Utilities & Examples

| File | Description |
|------|-------------|
| `example_activation_extraction.py` | Activation extraction examples |
| `example_usage.py` | General usage examples |
| `run_complete_pipeline.sh` | End-to-end automation |

---

## 🚀 Quick Start by Scenario

### Scenario 1: "I have a v5e-64 TPU and want to run inference"

1. Read: [QUICKSTART.md](QUICKSTART.md)
2. Run:
   ```bash
   ./run-docker.sh build
   ./run-docker.sh transform
   ./run-docker.sh distributed
   ```

### Scenario 2: "I have a v5e-256 multi-host TPU"

1. Read: 📦 **[DOCKER_MULTIHOST.md](DOCKER_MULTIHOST.md)** ⭐
2. Use: [MULTIHOST_QUICKREF.md](MULTIHOST_QUICKREF.md) for commands
3. Follow the 10-step guide in DOCKER_MULTIHOST.md

### Scenario 3: "I want to extract activations for SAE training"

1. Read: [README_ACTIVATION_EXTRACTION.md](README_ACTIVATION_EXTRACTION.md)
2. Check config: [LAYERS_CONFIG.md](LAYERS_CONFIG.md)
3. Run examples: `python example_activation_extraction.py`
4. Start extraction: [QUICK_START_EXTRACTION.md](QUICK_START_EXTRACTION.md)

### Scenario 4: "I want to understand the codebase"

1. Read: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Review: [FILES_OVERVIEW.txt](FILES_OVERVIEW.txt)
3. Check architecture: [README_DISTRIBUTED_INFERENCE.md](README_DISTRIBUTED_INFERENCE.md)

### Scenario 5: "I want to modify layer extraction"

1. Read: [LAYERS_CONFIG.md](LAYERS_CONFIG.md)
2. Edit: `distributed_inference_with_activations.py` line 65
3. Update: `layers_to_extract` field
4. Example: Change to `list(range(15, 24))` for layers 15-23

### Scenario 6: "I want to run tests"

1. Read: [README_TESTS.md](README_TESTS.md)
2. Run: `python run_tests.py`
3. Check: [TEST_SUMMARY.md](TEST_SUMMARY.md) for coverage

---

## 🔧 Configuration Reference

### Key Configuration Files

1. **Model Config**: `qwen2_jax.py` line 23-37
   - Number of layers: 24
   - Hidden size: 896
   - Attention heads: 14

2. **Activation Config**: `distributed_inference_with_activations.py` line 63-68
   - Layers to extract: 10-23 (14 layers)
   - Save frequency: every 10 batches
   - Cloud upload: optional

3. **Distributed Config**: `distributed_inference_with_activations.py` line 59-61
   - Mesh shape: (data_parallel, model_parallel)
   - v5e-64: (8, 8)
   - v5e-256: (16, 16)

### Environment Variables

```bash
# TPU Configuration
export JAX_PLATFORMS=tpu
export MESH_SHAPE="8,8"  # Adjust for your TPU size

# Paths
export MODEL_PATH=/path/to/qwen2.5-0.5b
export GCS_BUCKET=gs://your-bucket

# Inference Settings
export BATCH_SIZE=8
export MAX_SAMPLES=50000
```

---

## 📊 Performance Reference

### Storage Requirements (50k samples, layers 10-23)

- Per sample: 50.4 KB
- Total: ~2.52 GB
- Files: ~1,400 pickle files
- Metadata: ~1 MB

### Time Estimates

| TPU Type | Cores | Time (50k samples) |
|----------|-------|-------------------|
| v5e-64   | 64    | 50-100 min        |
| v5e-256  | 256   | 15-20 min         |
| v4-512   | 512   | 8-12 min          |

---

## 🆘 Troubleshooting

See specific documentation:
- Docker issues: [DOCKER_MULTIHOST.md](DOCKER_MULTIHOST.md) section "Troubleshooting"
- Activation issues: [README_ACTIVATION_EXTRACTION.md](README_ACTIVATION_EXTRACTION.md) section "Troubleshooting"
- Test failures: [README_TESTS.md](README_TESTS.md)

---

## 📞 Support

- File issues: GitHub repository
- Review examples: `example_activation_extraction.py`, `example_usage.py`
- Check logs: Docker container logs or `/tmp/outputs/`

---

## ✅ Checklists

### Pre-Flight Checklist (Multi-Host)

- [ ] TPU pod created
- [ ] Docker installed on all hosts
- [ ] Code copied to all hosts
- [ ] Docker image built on all hosts
- [ ] Data uploaded to GCS
- [ ] Model downloaded to all hosts
- [ ] GCS bucket configured
- [ ] Mesh shape set correctly

### Post-Run Checklist

- [ ] Activations saved locally
- [ ] Activations uploaded to GCS
- [ ] Metadata file created
- [ ] Output predictions generated
- [ ] No errors in logs
- [ ] TPU resources cleaned up

---

## 🎓 Learning Path

1. **Beginner**: Start with [QUICKSTART.md](QUICKSTART.md)
2. **Intermediate**: Read [README_DISTRIBUTED_INFERENCE.md](README_DISTRIBUTED_INFERENCE.md)
3. **Advanced**: Study [DOCKER_MULTIHOST.md](DOCKER_MULTIHOST.md)
4. **Expert**: Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Status**: Production Ready ✅
