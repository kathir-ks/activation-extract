# Qwen Activation Extraction for Sparse Autoencoders

Complete pipeline for extracting layer activations from Qwen 2.5 models on TPUs for Sparse Autoencoder (SAE) training and mechanistic interpretability research on ARC-AGI tasks.

## 🎯 What This Does

This system enables:
1. **Extract** layer activations from Qwen 2.5 models (0.5B, 7B)
2. **Run** on massively parallel pre-emptible TPUs (32-64 workers)
3. **Store** activations efficiently with automatic GCS upload
4. **Resume** automatically from checkpoints (fault-tolerant)
5. **Process** ARC-AGI tasks and FineWeb datasets

## ⚡ Quick Start (Parallel Workers - Recommended)

**For massively parallel extraction on pre-emptible TPUs:**

```bash
# Step 1: Create dataset streams (once)
python create_dataset_streams.py \
    --num_streams 32 \
    --output_dir ./data/streams \
    --max_samples 200000

# Step 2: On each TPU (worker 0 to 31)
export TPU_WORKER_ID=0  # Change for each TPU
export GCS_BUCKET=my-bucket
export UPLOAD_TO_GCS=true
./launch_worker.sh
```

See **[README_PARALLEL_WORKERS.md](README_PARALLEL_WORKERS.md)** for complete guide.

## 📚 Documentation

### Getting Started
- **[README_PARALLEL_WORKERS.md](README_PARALLEL_WORKERS.md)** - ⭐ **Start here!** Parallel workers quick start
- **[PARALLEL_WORKERS_GUIDE.md](PARALLEL_WORKERS_GUIDE.md)** - Comprehensive parallel workers guide
- **[QUICK_START.md](QUICK_START.md)** - Traditional single-host quick start

### Technical Documentation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation details & technical overview
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - What was delivered & testing results
- **[GCS_UPLOAD_GUIDE.md](GCS_UPLOAD_GUIDE.md)** - Google Cloud Storage setup
- **[SHARD_FORMAT_SPEC.md](SHARD_FORMAT_SPEC.md)** - Activation data format
- **[DATA_STORAGE_ARCHITECTURE.md](DATA_STORAGE_ARCHITECTURE.md)** - Storage architecture

## 📁 Project Structure

```
qwen/
├── 🚀 Parallel Workers (Recommended)
│   ├── extract_activations.py              # Main extraction script (checkpoint/resume)
│   ├── create_dataset_streams.py           # Split dataset into N streams
│   ├── launch_worker.sh                    # Launch single worker
│   ├── example_parallel_workflow.sh        # Complete workflow example
│   └── test_checkpoint_system.py           # Unit tests
│
├── 🔧 Core Utilities
│   ├── qwen2_jax.py                        # JAX Qwen model implementation
│   ├── qwen2_jax_with_hooks.py             # Model with activation extraction hooks
│   ├── kvcache_utils.py                    # KV cache for efficient generation
│   ├── convert_hf_to_arc_format.py         # HuggingFace → ARC format conversion
│   │
│   ├── core/                               # Shared core utilities
│   │   ├── jax_utils.py                    # JAX/TPU utilities
│   │   ├── dataset_utils.py                # Dataset loading & prompting
│   │   └── activation_storage.py           # GCS upload & storage
│   │
│   └── arc24/                              # ARC-AGI utilities
│       ├── encoders.py                     # Grid encoders (minimal, shape, etc.)
│       ├── prompting.py                    # Prompt generation with templates
│       ├── data_augmentation.py            # Geometric transformations
│       └── utils.py                        # Utility functions
│
├── 📦 Refactored Modular Framework
│   └── refactored/
│       ├── model/                          # Model implementation
│       │   ├── qwen.py                     # Core JAX model
│       │   ├── hooks.py                    # Activation hooks
│       │   └── kv_cache.py                 # KV cache utils
│       ├── arc/                            # ARC integration
│       ├── data/                           # Dataset utilities
│       └── extraction/                     # Extraction pipeline
│
├── 🗂️ Deployment & Scripts
│   ├── deploy/                             # Deployment scripts
│   └── scripts/                            # Helper scripts
│
├── 📚 Documentation
│   ├── README_PARALLEL_WORKERS.md          # Parallel workers quick start
│   ├── PARALLEL_WORKERS_GUIDE.md           # Comprehensive guide
│   ├── IMPLEMENTATION_SUMMARY.md           # Technical details
│   ├── IMPLEMENTATION_COMPLETE.md          # Delivery summary
│   ├── GCS_UPLOAD_GUIDE.md                 # GCS setup
│   └── [other docs...]
│
└── 🗄️ Archived (Old Implementations)
    └── archived_old/
        ├── extraction_scripts/             # Old extraction implementations
        ├── test_files/                     # Old test files
        ├── verification_scripts/           # Old verification scripts
        └── old_docs/                       # Outdated documentation
```

## 🏗️ Architecture

### Parallel Workers Architecture (Current)

```
┌─────────────────────────────────────┐
│  barc0/200k_HEAVY dataset           │
│  Split into N streams               │
└──────────┬──────────────────────────┘
           │
   ┌───────┼───────┬────────┐
   │       │       │        │
Stream0 Stream1 Stream2 ... StreamN
   │       │       │        │
   ▼       ▼       ▼        ▼
┌─────┐┌─────┐┌─────┐  ┌─────┐
│TPU 0││TPU 1││TPU 2│..│TPU N│
└──┬──┘└──┬──┘└──┬──┘  └──┬──┘
   │      │      │        │
   │ Checkpoint + GCS     │
   ▼      ▼      ▼        ▼
gs://bucket/activations/
├── tpu_0/shard_*.pkl.gz
├── tpu_1/shard_*.pkl.gz
└── ...
```

**Key Features:**
- ✅ **Independent Workers** - No coordination needed
- ✅ **Checkpoint/Resume** - Automatic recovery from pre-emption
- ✅ **Per-Worker GCS Folders** - Organized, conflict-free storage
- ✅ **Cost-Effective** - Designed for 70% cheaper pre-emptible TPUs
- ✅ **All Layers** - Extracts all 24 layers by default

## 🔧 Installation

```bash
# Clone repository
git clone <repo-url>
cd qwen

# Install dependencies
pip install -r requirements.txt

# Authenticate with GCS (if using GCS upload)
gcloud auth application-default login
```

### Requirements
- Python 3.10+
- JAX with TPU support
- Transformers (HuggingFace)
- Google Cloud SDK (for GCS upload)

See `requirements.txt` for complete list.

## 📝 Usage Examples

### 1. Parallel Workers (Recommended)

**Best for:** 32-64 pre-emptible TPUs processing 200k samples

```bash
# Create streams once
python create_dataset_streams.py --num_streams 32 --max_samples 200000

# Launch each worker
export TPU_WORKER_ID=0
export GCS_BUCKET=my-bucket
export UPLOAD_TO_GCS=true
./launch_worker.sh
```

**Features:**
- Automatic checkpoint/resume
- Per-worker GCS folders
- Fault-tolerant for pre-emptible TPUs
- 70% cost savings vs on-demand

**Documentation:** [README_PARALLEL_WORKERS.md](README_PARALLEL_WORKERS.md)

### 2. Single-Host Extraction (Traditional)

**Best for:** Testing or small datasets on a single TPU

```bash
python extract_activations.py \
    --dataset_path data/tasks.jsonl \
    --model_path Qwen/Qwen2.5-0.5B \
    --output_dir ./activations \
    --batch_size 4 \
    --upload_to_gcs \
    --gcs_bucket my-bucket
```

**Documentation:** [QUICK_START.md](QUICK_START.md)

### 3. Dataset Conversion

**Convert HuggingFace dataset to ARC format:**

```bash
python convert_hf_to_arc_format.py \
    --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
    --output_file arc_tasks.jsonl \
    --max_tasks 1000 \
    --verbose
```

## 🧪 Testing

```bash
# Test checkpoint system
python test_checkpoint_system.py

# Expected output:
# ✅ ALL TESTS PASSED
```

## 📊 Performance

- **Throughput:** ~3-5 samples/second per TPU
- **Checkpoint Overhead:** <1% of total time
- **Resume Time:** <30 seconds from checkpoint
- **GCS Upload:** Non-blocking background upload

## 💰 Cost Estimate

### Example: 200k Samples on 32 Pre-emptible v4-8 TPUs

- **Pre-emptible:** 32 × $1.35/hour × 3 hours = **$130**
- **On-demand:** 32 × $4.50/hour × 3 hours = **$432**
- **Savings:** **$302 (70% reduction)**

## 🔑 Key Features

### Fault Tolerance
- ✅ Checkpoint after every ~1GB shard (~10 minutes)
- ✅ Automatic resume on restart
- ✅ Maximum data loss: 1 shard
- ✅ No manual intervention needed

### Scalability
- ✅ Independent workers (no coordination)
- ✅ Horizontal scaling (add more workers = faster)
- ✅ No bottlenecks or shared state
- ✅ Failure isolation (one worker failure doesn't affect others)

### Organization
- ✅ Per-worker GCS folders: `gs://bucket/activations/tpu_N/`
- ✅ No conflicts between workers
- ✅ Easy monitoring (check each folder independently)
- ✅ Checkpoint files: `./checkpoints/checkpoint_worker_N.json`

## 🛠️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TPU_WORKER_ID` | 0 | Worker ID (0 to N-1) |
| `GCS_BUCKET` | - | GCS bucket for uploads |
| `UPLOAD_TO_GCS` | false | Enable GCS upload |
| `MODEL_PATH` | Qwen/Qwen2.5-0.5B | Model to use |
| `BATCH_SIZE` | 4 | Batch size per device |
| `SHARD_SIZE_GB` | 1.0 | Shard size in GB |

See [PARALLEL_WORKERS_GUIDE.md](PARALLEL_WORKERS_GUIDE.md) for complete configuration options.

## 📈 Monitoring

```bash
# Check specific worker
cat checkpoints/checkpoint_worker_5.json

# Check GCS uploads
gsutil ls gs://my-bucket/activations/tpu_5/

# Monitor all workers
for i in {0..31}; do
    echo -n "Worker $i: "
    jq -r '.total_samples_processed' checkpoints/checkpoint_worker_$i.json
done
```

## 🐛 Troubleshooting

### Common Issues

**Worker not starting?**
```bash
# Check stream file exists
ls data/streams/stream_$(printf '%03d' $TPU_WORKER_ID).jsonl

# Check GCS authentication
gsutil ls gs://$GCS_BUCKET/
```

**Worker not resuming?**
```bash
# Check checkpoint
cat checkpoints/checkpoint_worker_$TPU_WORKER_ID.json

# Verify TPU_WORKER_ID is set
echo $TPU_WORKER_ID
```

**GCS upload failing?**
```bash
# Re-authenticate
gcloud auth application-default login

# Test write access
echo "test" | gsutil cp - gs://$GCS_BUCKET/test.txt
```

See [PARALLEL_WORKERS_GUIDE.md](PARALLEL_WORKERS_GUIDE.md) for comprehensive troubleshooting.

## 📦 Output Format

Each shard contains activations for all layers:

```python
import pickle
import gzip

# Load a shard
with gzip.open('shard_0001.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# Structure:
# data = {
#     0: [  # Layer 0
#         {'sample_idx': 0, 'activation': np.array(...), 'shape': (seq_len, hidden_size), ...},
#         ...
#     ],
#     1: [...],  # Layer 1
#     ...
#     23: [...]  # Layer 23
# }
```

See [SHARD_FORMAT_SPEC.md](SHARD_FORMAT_SPEC.md) for details.

## 🤝 Contributing

See [refactored/CONTRIBUTING.md](refactored/CONTRIBUTING.md) for development guidelines.

## 📄 License

[Add license information]

## 🗂️ Archived Code

Old implementations have been moved to `archived_old/` for reference. These include:
- Old multi-host coordination code
- Old extraction scripts
- Old test files
- Outdated documentation

See [archived_old/README.md](archived_old/README.md) for details.

## 📞 Support

For issues or questions:
1. Check [PARALLEL_WORKERS_GUIDE.md](PARALLEL_WORKERS_GUIDE.md) troubleshooting section
2. Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. Run `test_checkpoint_system.py` to verify your setup

---

**Status:** ✅ Production-ready with comprehensive documentation and testing

**Last Updated:** January 14, 2026
