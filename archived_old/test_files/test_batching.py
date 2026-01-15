# Quick test to verify batching behavior
batch_size = 2
max_samples = 3

batch_sequences = []
sample_idx = 0
batch_count = 0

print(f"Config: batch_size={batch_size}, max_samples={max_samples}\n")

while sample_idx < max_samples:
    # Simulate adding sample
    batch_sequences.append(f"sample_{sample_idx}")
    print(f"Added sample {sample_idx}, batch now has {len(batch_sequences)} items")
    
    sample_idx += 1
    
    # Process batch when full
    if len(batch_sequences) >= batch_size:
        batch_count += 1
        print(f"  → Processing BATCH {batch_count}: {batch_sequences}")
        batch_sequences = []

# Process remaining
if batch_sequences:
    batch_count += 1
    print(f"  → Processing final BATCH {batch_count}: {batch_sequences}")

print(f"\nTotal batches: {batch_count}")
print(f"Expected: 2 batches (batch1=[0,1], batch2=[2])")
