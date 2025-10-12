#!/usr/bin/env python3
"""
Test the HF dataset transformation
"""

from datasets import load_dataset
import json

print("Loading a sample from the HF dataset...")

# Load just 1 sample to inspect
dataset = load_dataset(
    'barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems',
    split='train',
    streaming=True
)

# Get first example
sample = next(iter(dataset))

print("\n" + "="*70)
print("SAMPLE STRUCTURE")
print("="*70)
print(f"Keys: {sample.keys()}")
print(f"\nNumber of examples: {len(sample['examples'])}")

print("\n" + "="*70)
print("EXAMPLES")
print("="*70)
for i, ex in enumerate(sample['examples']):
    print(f"\nExample {i+1}:")
    if isinstance(ex, dict):
        print(f"  Keys: {ex.keys()}")
        if 'input' in ex:
            print(f"  Input shape: {len(ex['input'])} x {len(ex['input'][0]) if ex['input'] else 0}")
        if 'output' in ex:
            print(f"  Output shape: {len(ex['output'])} x {len(ex['output'][0]) if ex['output'] else 0}")
    else:
        print(f"  Type: {type(ex)}")
        print(f"  Content: {str(ex)[:200]}")

print("\n" + "="*70)
print("FULL SAMPLE (first 500 chars)")
print("="*70)
print(json.dumps(sample, indent=2)[:500])

print("\n✓ Sample inspection complete!")
print("\nNow you can run:")
print("python transform_hf_to_arc.py --max_samples 10")
