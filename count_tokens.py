"""Count tokens for ARC prompts without loading full model weights.

Creates prompts from `test_data_small.json` using the existing `arc24` encoders and
prompting utilities, then tokenizes them with a Hugging Face tokenizer. Defaults to
`gpt2` for a quick run; pass `--model_path` to use a different tokenizer (e.g. your Qwen
repo name). If the tokenizer repo doesn't provide `apply_chat_template`, a simple
fallback implementation is used.

Outputs:
 - Printed summary (min/max/mean/median)
 - `token_lengths.csv` in the current directory with token length per prompt

Usage:
    python3 count_tokens.py --model_path gpt2 --dataset_path test_data_small.json
"""

import json
import argparse
import os
import csv
import numpy as np

from arc24.encoders import create_grid_encoder
from arc24.prompting import create_prompts


def make_fallback_apply_chat_template(messages, add_generation_prompt=False):
    # Simple, conservative chat formatting that approximates the project chat template
    out = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        if role == 'system':
            out.append(f"<|system|>\n{content}\n")
        elif role == 'user':
            out.append(f"<|user|>\n{content}\n")
        elif role == 'assistant':
            out.append(f"<|assistant|>\n{content}\n")
        else:
            out.append(f"<{role}>\n{content}\n")
    out.append('<|end|>')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='gpt2', help='HuggingFace model id for tokenizer (default: gpt2)')
    parser.add_argument('--dataset_path', type=str, default='test_data_small.json', help='Path to ARC dataset JSON')
    parser.add_argument('--prompt_version', type=str, default='output-from-examples-v0')
    parser.add_argument('--predictions_per_task', type=int, default=8)
    parser.add_argument('--grid_encoder', type=str, default="GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))")
    parser.add_argument('--max_seq_length', type=int, default=2048)
    args = parser.parse_args()

    # Load dataset
    dataset_path = os.path.abspath(args.dataset_path)
    with open(dataset_path) as f:
        data = json.load(f)

    grid_encoder = create_grid_encoder(args.grid_encoder)

    # Try to import transformers tokenizer lazily
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError('Please install `transformers` and `tokenizers` packages (pip install transformers tokenizers)')

    print(f"Loading tokenizer `{args.model_path}` (may download from HF)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Provide fallback for `apply_chat_template` if missing
    if not hasattr(tokenizer, 'apply_chat_template'):
        def _apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            return make_fallback_apply_chat_template(messages, add_generation_prompt)
        tokenizer.apply_chat_template = _apply_chat_template
        print('Note: tokenizer had no `apply_chat_template`; using fallback formatting.')

    # Create prompts
    prompts_data = create_prompts(
        data, grid_encoder, tokenizer, args.prompt_version,
        args.predictions_per_task, verbose=False
    )

    print(f"Created {len(prompts_data)} prompts. Tokenizing with `{args.model_path}`...")

    token_lengths = []
    for p in prompts_data:
        text = p['prompt']
        enc = tokenizer(text, return_tensors='np', truncation=True, max_length=args.max_seq_length)
        input_ids = enc['input_ids'][0]
        token_lengths.append(int(len(input_ids)))

    token_lengths = np.array(token_lengths, dtype=int)

    # Summary
    print('\nToken length statistics (tokens per prompt):')
    print(f'  Count: {len(token_lengths)}')
    print(f'  Min:   {token_lengths.min()}')
    print(f'  25%:   {np.percentile(token_lengths, 25):.1f}')
    print(f'  Median:{np.median(token_lengths):.1f}')
    print(f'  Mean:  {token_lengths.mean():.1f}')
    print(f'  75%:   {np.percentile(token_lengths, 75):.1f}')
    print(f'  Max:   {token_lengths.max()}')

    # Save CSV
    out_csv = 'token_lengths.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_index', 'task_id', 'token_length'])
        for i, (p, l) in enumerate(zip(prompts_data, token_lengths)):
            writer.writerow([i, p.get('task_id', ''), int(l)])

    print(f'Wrote token lengths to {out_csv}')


if __name__ == '__main__':
    main()
