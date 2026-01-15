# Archived Files

This folder contains old implementations, tests, and documentation that have been superseded by the new parallel workers system.

## Archive Date
January 14, 2026

## Folder Structure

### `extraction_scripts/`
Old extraction implementations that have been replaced:
- `arc_inference_jax.py` - Old ARC inference
- `extract_activations_arc.py` - Old ARC extraction
- `extract_activations_fineweb.py` - Old FineWeb extraction
- `extract_activations_arc_v5e64.py` - Old multi-host v5e-64
- `extract_activations_fineweb_multihost.py` - Old multi-host FineWeb
- `extract_activations_optimized.py` - Old optimized version
- `launch_distributed_extraction.py` - Old distributed launcher

**Replaced by:** `extract_activations.py` with parallel workers support

### `test_files/`
Old test files:
- `test_activations.py`
- `test_extraction.py`
- `test_extraction_7b.py`
- `test_batching.py`
- `test_timing.py`
- `test_fineweb_extraction.py`
- `test_fineweb_detailed.py`
- `test_arc_vs_fineweb_comparison.py`
- `test_gcs_upload.py`

**Replaced by:** `test_checkpoint_system.py`

### `verification_scripts/`
Old verification and benchmark scripts:
- `verify_model_inference.py`
- `verify_model.py`
- `quick_verification.py`
- `timing_analysis.py`
- `analyze_timing.py`
- `benchmark_jit_improvements.py`
- `performance_profiler.py`
- `count_tokens.py`
- `memory_calculation.py`

**Status:** No longer needed, core functionality verified

### `utilities/`
Old utility scripts:
- `create_sharded_dataset.py` - Old sharding system
- `shard_manager.py` - Old shard management

**Replaced by:** `create_dataset_streams.py`

### `shell_scripts/`
Old shell scripts:
- `run_timing_test.sh`
- `test_performance_real.sh`
- `test_sharding.sh`
- `launch_v5e64.sh`
- `launch_distributed_extraction.sh`

**Replaced by:** `launch_worker.sh` and `example_parallel_workflow.sh`

### `old_docs/`
Outdated documentation:
- `USAGE_MULTIHOST.md` - Old multi-host instructions
- `MULTI_MACHINE_DEPLOYMENT.md` - Old deployment guide
- `V5E64_DEPLOYMENT_GUIDE.md` - Old v5e-64 guide
- `README_V5E64.md` - Old v5e-64 README
- `MULTIHOST_DEPLOYMENT.md` - Old multi-host deployment
- `JIT_PERFORMANCE_ANALYSIS.md` - Old performance docs
- `PERFORMANCE_IMPROVEMENTS_APPLIED.md` - Old performance docs
- `PERFORMANCE_TEST_RESULTS.md` - Old test results
- `DATASET_GUIDE.md` - Old dataset guide
- `SHARDING_SYSTEM.md` - Old sharding docs
- `REFACTORING.md` - Old refactoring notes
- `DEPLOYMENT_INSTRUCTIONS.md` - Old deployment docs

**Replaced by:**
- `README_PARALLEL_WORKERS.md` - New quick start
- `PARALLEL_WORKERS_GUIDE.md` - New comprehensive guide
- `IMPLEMENTATION_SUMMARY.md` - New technical details
- `IMPLEMENTATION_COMPLETE.md` - New completion summary

## Why These Were Archived

The codebase has been refactored to use a **massively parallel, independent single-host worker architecture** with:

1. **Checkpoint/Resume** - Automatic recovery from pre-emption
2. **Per-Worker GCS Folders** - Organized, conflict-free storage
3. **Zero Coordination** - No complex multi-host coordination needed
4. **Simple Deployment** - Easy to launch and monitor
5. **Cost-Effective** - Designed for pre-emptible TPUs

The old multi-host coordination code and related documentation were complex and harder to maintain. The new system achieves better:
- **Simplicity** - No coordinator, no synchronization
- **Fault Tolerance** - Workers restart independently
- **Scalability** - Add more workers without changes
- **Organization** - Per-worker folders in GCS

## Can These Be Deleted?

These files are kept for reference but can be safely deleted if:
1. The new parallel workers system is working correctly
2. You don't need to reference the old implementations
3. You've backed up the repository

**Recommendation:** Keep for 30-60 days, then delete if not needed.

## Current Active Files

For the current codebase structure, see the main README.md in the root directory.
