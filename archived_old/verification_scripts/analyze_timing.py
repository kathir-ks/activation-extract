# Analyze the timing pattern

# From log:
# 3it [00:22,  8.08s/it] - After loop
# 3it [00:50, 16.87s/it] - After finally

batch_size = 2
total_samples = 3

# Batches:
# Batch 1: samples 0, 1 (size=2) - processed in loop
# Batch 2: sample 2 (size=1) - processed in finally

# Timing:
loop_time = 22  # seconds
total_time = 50  # seconds
finally_time = total_time - loop_time

print("="*60)
print("TIMING ANALYSIS")
print("="*60)
print(f"Loop time (batch 1, size=2): {loop_time}s")
print(f"Finally time (batch 2, size=1): {finally_time}s")
print(f"\n⚠️  PROBLEM: Batch of 1 sample takes {finally_time}s!")
print(f"   Expected: ~{loop_time/2}s (half the work)")
print(f"   Actual: {finally_time}s ({finally_time/(loop_time/2):.1f}x slower!)")
print(f"\n💡 ROOT CAUSE: JIT recompilation for different batch size!")
print(f"   - Batch size 2: JIT compiled in loop") 
print(f"   - Batch size 1: RECOMPILES in finally block (slow!)")
print("="*60)
