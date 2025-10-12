"""
ARC-AGI Inference using JAX/TPU implementation adapted from the original solution
Integrates with the existing qwen2_jax.py TPU setup
"""

import jax
import jax.numpy as jnp
from jax import lax
import json
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import argparse
from tqdm.auto import tqdm
from itertools import islice, product

from transformers import AutoTokenizer
from qwen2_jax import QwenModel, QwenConfig, convert_hf_to_jax_weights
from transformers import AutoModelForCausalLM
import torch

from arc24.data_augmentation import (
    apply_data_augmentation, revert_data_augmentation, get_random_color_map, set_random_seed)
from arc24.prompting import parse_grid_from_response, print_smallest_prompt, create_prompts_from_task
from arc24.encoders import create_grid_encoder


@dataclass
class ARCConfig:
    """Configuration for ARC-AGI inference"""
    output_filepath: str = 'submission.json'
    # Model
    model_path: str = "Qwen/Qwen2.5-0.5B"  # Default to HF model
    max_model_len: int = 10240
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    # Dataset
    dataset_path: str = 'arc_data.json'
    n_tasks: Optional[int] = None
    # Inference params
    max_output_tokens: int = 1100
    predictions_per_task: int = 8
    temperature: float = 0.0
    batch_size: int = 8  # Smaller batch size for JAX
    random_seed: Optional[int] = None
    verbose: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="ARC-AGI Inference Configuration")
    parser.add_argument('--model_path', type=str, help="Path to the model")
    parser.add_argument('--max_model_len', type=int, help="Maximum number of tokens in the model")
    parser.add_argument('--grid_encoder', type=str, help="Name of the grid encoder")
    parser.add_argument('--prompt_version', type=str, help="Prompt version")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset to make inference")
    parser.add_argument('--n_tasks', type=int, help="If given only the first n tasks will be evaluated")
    parser.add_argument('--output_filepath', type=str, help="Path to the json file with the predictions")
    parser.add_argument('--predictions_per_task', type=int, help="Number of predictions per task, use a multiple of 8")
    parser.add_argument('--temperature', type=float, help="temperature for sampling, 0.0 for greedy search")
    parser.add_argument('--batch_size', type=int, help="batch size for inference")
    parser.add_argument('--max_output_tokens', type=int, help="Maximum number of tokens to generate")
    parser.add_argument('--random_seed', type=int, help="Random seed for data augmentation")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    return parser.parse_args()


def load_jax_model_and_tokenizer(model_path: str, config: QwenConfig):
    """Load JAX model and tokenizer"""
    print(f"Loading {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load HF model for weight conversion
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # Initialize JAX model
    jax_model = QwenModel(config)

    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)

    # Initialize parameters
    params = jax_model.init(rng, dummy_input)

    # Convert weights
    print("Converting HuggingFace weights to JAX format...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    return jax_model, tokenizer, params


def create_prompts(data: Dict, grid_encoder, tokenizer, prompt_version: str, predictions_per_task: int):
    """Create prompts for all tasks with data augmentation"""
    prompts = []
    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts'):
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        num_augmentations = len(data_augmentation_params)

        # Calculate how many times to repeat each augmentation
        # Ensure at least 1 prompt per augmentation if predictions_per_task >= num_augmentations
        repeats_per_aug = max(1, predictions_per_task // num_augmentations)

        for hflip, n_rot90 in data_augmentation_params:
            for _ in range(repeats_per_aug):
                color_map = get_random_color_map(change_background_probability=0.1)
                data_augmentation_kwargs = dict(hflip=hflip, n_rot90=n_rot90, color_map=color_map)
                augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)
                task_prompts = create_prompts_from_task(
                    augmented_task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                    is_train_prompt=False, prompt_version=prompt_version)
                for idx, prompt in enumerate(task_prompts):
                    prompts.append(dict(task_id=task_id,
                                      data_augmentation_kwargs=data_augmentation_kwargs,
                                      prompt=prompt,
                                      idx=idx))

            # Break early if we only need a subset of augmentations
            if len(prompts) >= predictions_per_task * len(task['test']):
                break

    return prompts


def generate_tokens_jax(model, params, input_ids, max_tokens: int = 50):
    """Generate tokens using JAX model"""
    generated_ids = input_ids

    for _ in range(max_tokens):
        logits = model.apply(params, generated_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)

        # Simple stopping condition (can be improved)
        if generated_ids.shape[1] > input_ids.shape[1] + max_tokens:
            break

    return generated_ids


def generate_outputs_with_batches(model, params, tokenizer, prompts: List[str],
                                batch_size: int = 8, max_output_tokens: int = 1100):
    """Generate outputs using JAX model in batches"""
    outputs = []
    print(f'Generating outputs with batch_size={batch_size}, there are {len(prompts)} prompts')

    for i in tqdm(range(0, len(prompts), batch_size), desc='Generating outputs with batches'):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize batch
        batch_inputs = []
        for prompt in batch_prompts:
            inputs = tokenizer(prompt, return_tensors="np", truncation=True, max_length=2048)
            batch_inputs.append(jnp.array(inputs['input_ids']))

        # Pad sequences to same length
        max_len = max(seq.shape[1] for seq in batch_inputs)
        padded_inputs = []
        for seq in batch_inputs:
            pad_width = max_len - seq.shape[1]
            if pad_width > 0:
                seq = jnp.pad(seq, ((0, 0), (0, pad_width)), constant_values=tokenizer.pad_token_id or 0)
            padded_inputs.append(seq)

        batch_input_ids = jnp.concatenate(padded_inputs, axis=0)

        # Generate outputs
        generated_ids = generate_tokens_jax(model, params, batch_input_ids, max_output_tokens)

        # Decode outputs
        for j, gen_ids in enumerate(generated_ids):
            if j < len(batch_prompts):  # Handle case where batch is not full
                # Extract only the generated part
                input_len = batch_inputs[j].shape[1]
                generated_part = gen_ids[input_len:]

                # Convert to CPU and decode
                generated_text = tokenizer.decode(np.array(generated_part), skip_special_tokens=True)

                # Mock output object to match original interface
                output = type('obj', (object,), {
                    'outputs': [type('obj', (object,), {
                        'text': generated_text,
                        'cumulative_logprob': 0.0,  # Placeholder
                        'token_ids': np.array(generated_part).tolist()
                    })()]
                })()
                outputs.append(output)

    return outputs


def create_tasks_results(outputs, prompts_conf, grid_encoder, prompt_version: str, data: Dict, verbose: bool = False):
    """Create task results from model outputs"""
    task_results = prompts_conf.copy()
    for idx, output in tqdm(enumerate(outputs), total=len(outputs), desc='Parsing outputs'):
        task_id = prompts_conf[idx]['task_id']
        data_augmentation_kwargs = prompts_conf[idx]['data_augmentation_kwargs']
        sample_idx = prompts_conf[idx]['idx']
        response = output.outputs[0].text

        try:
            if prompt_version.startswith('code-from-examples'):
                # Code-based inference would require execution framework
                # For now, fall back to grid parsing
                grid = parse_grid_from_response(response, grid_encoder)
                grid = revert_data_augmentation(grid, **data_augmentation_kwargs)
            else:
                grid = parse_grid_from_response(response, grid_encoder)
                grid = revert_data_augmentation(grid, **data_augmentation_kwargs)
            validate_grid(grid)
        except Exception as e:
            if verbose:
                print(f'Exception when parsing response from {task_id}_{sample_idx}: {e} \n{response}')
            grid = []

        task_results[idx]['grid'] = grid
        task_results[idx]['response'] = response
        task_results[idx]['cumulative_logprob'] = output.outputs[0].cumulative_logprob
        task_results[idx]['n_tokens'] = len(output.outputs[0].token_ids)

    return task_results


def validate_grid(grid):
    """Validate that grid is properly formatted"""
    assert isinstance(grid, list), f'Grid is not a list: {grid}'
    grid = np.array(grid, dtype=np.int8)
    assert grid.ndim == 2, f'Grid has more than 2 dimensions: {grid.ndim}'
    assert grid.shape[0] > 0, f'Grid has 0 rows: {grid.shape}'
    assert grid.shape[1] > 0, f'Grid has 0 columns: {grid.shape}'
    assert grid.min() >= 0, f'Grid has negative values: {grid.min()}'
    assert grid.max() < 10, f'Grid has values greater than 9: {grid.max()}'


def create_solutions(task_results: List[Dict], data: Dict):
    """Create final solutions from task results"""
    solutions = _create_empty_solutions(data)
    for task_result in task_results:
        task_id = task_result['task_id']
        sample_idx = task_result['idx']
        attempt_name = f"attempt_{len(solutions[task_id][sample_idx]) + 1}"
        solutions[task_id][sample_idx][attempt_name] = task_result['grid']
    return solutions


def _create_empty_solutions(data: Dict):
    """Create empty solutions structure"""
    solutions = dict()
    for task_id, task in data.items():
        solutions[task_id] = [dict() for _ in task['test']]
    return solutions


def inference_main():
    """Main inference function adapted for JAX/TPU"""
    # Parse arguments and create config
    args = parse_args()
    cfg = ARCConfig(**{k: v for k, v in vars(args).items() if v is not None})
    print(f'Inference configuration: {asdict(cfg)}')

    # Initialize TPU
    print("Initializing TPU...")
    try:
        jax.devices("tpu")
        print(f"Found {len(jax.devices('tpu'))} TPU cores")
    except:
        print("No TPU found, using default backend:", jax.default_backend())
        print(f"Available devices: {jax.devices()}")

    # Load data
    with open(cfg.dataset_path) as f:
        data = json.load(f)
    if cfg.n_tasks is not None and cfg.n_tasks > 0:
        data = dict(islice(data.items(), cfg.n_tasks))
    print(f'There are {len(data)} tasks to solve in {cfg.dataset_path}')

    # Create model configuration (you may need to adjust based on your fine-tuned model)
    config = QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True
    )

    # Load model and tokenizer
    model, tokenizer, params = load_jax_model_and_tokenizer(cfg.model_path, config)

    # Set random seed
    set_random_seed(cfg.random_seed)

    # Create grid encoder
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    # Create prompts
    prompts_conf = create_prompts(
        data, grid_encoder, tokenizer, cfg.prompt_version, cfg.predictions_per_task)
    prompts = [conf['prompt'] for conf in prompts_conf]

    if cfg.verbose:
        print_smallest_prompt(prompts)

    # Generate outputs
    outputs = generate_outputs_with_batches(
        model, params, tokenizer, prompts,
        batch_size=cfg.batch_size, max_output_tokens=cfg.max_output_tokens)

    # Process results
    task_results = create_tasks_results(
        outputs=outputs, prompts_conf=prompts_conf, grid_encoder=grid_encoder,
        prompt_version=cfg.prompt_version, data=data, verbose=cfg.verbose)

    solutions = create_solutions(task_results, data)

    # Save results
    with open(cfg.output_filepath, 'w') as f:
        json.dump(solutions, f)
    with open(cfg.output_filepath.replace('.json', '_task_results.json'), 'w') as f:
        json.dump(task_results, f)

    print("Inference completed successfully!")
    print(f"Results saved to {cfg.output_filepath}")


if __name__ == '__main__':
    inference_main()