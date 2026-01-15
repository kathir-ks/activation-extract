"""
Prompt Templates for ARC Tasks

Provides prompt generation for various ARC task formats.
Uses Jinja2 templates for flexible prompt construction.
"""

from jinja2 import Template
from typing import Dict, List, Any


def parse_grid_from_response(text: str, grid_encoder) -> List[List[int]]:
    """Parse a grid from model response text."""
    return grid_encoder.to_grid('```grid' + text)


def create_prompts_from_task(
    task: Dict,
    grid_encoder,
    tokenizer,
    is_train_prompt: bool = True,
    prompt_version: str = 'output-from-examples-v0'
) -> List[str]:
    """Create prompts from an ARC task."""
    system_prompt, prompt_template, answer_template = get_prompt_templates(prompt_version)
    train_samples = [
        {key: grid_encoder.to_text(grid) for key, grid in sample.items()} 
        for sample in task['train']
    ]
    
    prompts = []
    for test_sample in task['test']:
        render_kwargs = dict(
            train_samples=train_samples,
            test_input=grid_encoder.to_text(test_sample['input'])
        )
        
        if prompt_version.startswith('select-output-from-examples'):
            render_kwargs['test_output_choices'] = [
                grid_encoder.to_text(grid) for grid in task['test_output_choices']
            ]
        elif prompt_version.startswith('verify-output-from-examples'):
            render_kwargs['test_output'] = grid_encoder.to_text(test_sample['output'])
        elif prompt_version.startswith('output-from-code'):
            render_kwargs['code'] = task['code']

        user_message = prompt_template.render(**render_kwargs)
        
        if is_train_prompt:
            if prompt_version.startswith('output'):
                output = grid_encoder.to_text(test_sample['output'])
            elif prompt_version.startswith('input-from-inputs'):
                output = grid_encoder.to_text(test_sample['input'])
            elif prompt_version.startswith('code-from-examples'):
                output = '```python\n' + task['code'] + '\n```'
            elif prompt_version.startswith('select-output-from-examples'):
                output = task['test_correct_choice_index']
            elif prompt_version.startswith('verify-output-from-examples'):
                output = task['is_test_output_correct']
            else:
                raise ValueError(f'Unknown prompt version {prompt_version}')
        else:
            if prompt_version.startswith('code-from-examples'):
                output = '```python\n'
            elif prompt_version.startswith('verify-output-from-examples') or \
                 prompt_version.startswith('select-output-from-examples'):
                output = ''
            else:
                output = '```grid'
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_template.render(output=output)}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        if not is_train_prompt:
            prompt = remove_assistant_ending(prompt)
            
        prompts.append(prompt)
        
    return prompts


def remove_assistant_ending(text: str) -> str:
    """Remove trailing assistant tokens from prompt."""
    for token in ['<|eot_id|>', '<|im_end|>', '<|end|>']:
        if token in text:
            return token.join(text.split(token)[:-1])
    raise NotImplementedError('Unknown chat template')


def print_smallest_prompt(prompts: List[str]) -> None:
    """Print the smallest prompt."""
    smallest = sorted(prompts, key=len)[0]
    pretty_print_prompt(smallest)


def pretty_print_prompt(text: str, default_color: str = 'black') -> None:
    """Pretty print a prompt."""
    print('-'*80)
    print(text)
    print('-'*80)


def get_prompt_templates(prompt_version: str):
    """Get system prompt, prompt template, and answer template."""
    if prompt_version == 'output-from-examples-v0':
        return SYSTEM_PROMPT_V0, PROMPT_TEMPLATE_V0, ANSWER_TEMPLATE_V0
    elif prompt_version == 'output-from-examples-v1':
        return SYSTEM_PROMPT_V1, PROMPT_TEMPLATE_V1, ANSWER_TEMPLATE_V0
    elif prompt_version == 'input-from-inputs-v0':
        return SYSTEM_PROMPT_V1, PROMPT_TEMPLATE_INPUT_V0, ANSWER_TEMPLATE_INPUT_V0
    elif prompt_version == 'code-from-examples-v0':
        return SYSTEM_PROMPT_V1, PROMPT_TEMPLATE_CODE_V0, ANSWER_TEMPLATE_CODE_V0
    elif prompt_version == 'output-from-code-v0':
        return SYSTEM_PROMPT_V1, PROMPT_TEMPLATE_OUTPUT_CODE_V0, ANSWER_TEMPLATE_V0
    else:
        raise ValueError(f'Unknown prompt version {prompt_version}')


# System prompts
SYSTEM_PROMPT_V0 = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation between the input and the output and apply it to the last input grid given by the user."""

SYSTEM_PROMPT_V1 = "You are a helpful assistant."

# Main prompt template
PROMPT_TEMPLATE_V0 = Template("""Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
## Test case

### Input

{{ test_input }}
""")

PROMPT_TEMPLATE_V1 = Template("""Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and apply it to the test input grid.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
## Test case

### Input

{{ test_input }}
""")

PROMPT_TEMPLATE_INPUT_V0 = Template("""Your task is to create a new grid that follows the same distribution as the input grids from the Abstraction and Reasoning Challenge (ARC).
{% for sample in train_samples %}
## Grid example {{ loop.index }}

{{ sample.input }}
{% endfor %}
""")

PROMPT_TEMPLATE_CODE_V0 = Template("""Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and write a python function that implements the transformation.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
""")

PROMPT_TEMPLATE_OUTPUT_CODE_V0 = Template("""Your task is to transform the input grid using the transformation defined in the python code below.

## Code

```python
{{ code }}
```

## Example

### Input

{{ test_input }}
""")

# Answer templates
ANSWER_TEMPLATE_V0 = Template("""### Output

{{ output }}""")

ANSWER_TEMPLATE_INPUT_V0 = Template("""## New grid

{{ output }}""")

ANSWER_TEMPLATE_CODE_V0 = Template("""{{ output }}""")
