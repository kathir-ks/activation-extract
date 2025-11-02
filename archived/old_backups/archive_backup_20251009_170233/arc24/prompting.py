from jinja2 import Template
from termcolor import colored


def parse_grid_from_response(text, grid_encoder):
    return grid_encoder.to_grid('```grid' + text)


def create_prompts_from_task(task, grid_encoder, tokenizer,
                             is_train_prompt=True, prompt_version='output-from-examples-v0'):
    system_prompt, prompt_template, answer_template = get_prompt_templates(prompt_version)
    train_samples = [{key: grid_encoder.to_text(grid) for key, grid in sample.items()} for sample in task['train']]
    prompts = []
    for test_sample in task['test']:
        render_kwargs = dict(train_samples=train_samples,
                             test_input=grid_encoder.to_text(test_sample['input']))
        if prompt_version.startswith('select-output-from-examples'):
            render_kwargs['test_output_choices'] = [grid_encoder.to_text(grid) for grid in task['test_output_choices']]
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
            elif prompt_version.startswith('verify-output-from-examples') or prompt_version.startswith('select-output-from-examples'):
                output = ''
            else:
                output = '```grid'
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer_template.render(output=output)}]
        prompt = tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=False)
        if not is_train_prompt:
            prompt = remove_assistant_ending(prompt)
        prompts.append(prompt)
    return prompts


def remove_assistant_ending(text):
    """
phi-3

```
<|assistant|>
### Output
```grid
<|end|>
<|endoftext|>
```

llama 3.1

```
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Output
```grid<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
    """
    if '<|eot_id|>' in text: # llama
        split_text = '<|eot_id|>'
    elif '<|im_end|>' in text: # qwen
        split_text = '<|im_end|>'
    elif '<|end|>' in text:
        split_text = '<|end|>' # phi-3
    else:
        raise NotImplementedError('Unknown chat template')
    return split_text.join(text.split(split_text)[:-1])


def print_smallest_prompt(prompts):
    smallest_prompt = sorted(prompts, key=lambda x: len(x))[0]
    print('\n\nSmaller prompt:')
    pretty_print_prompt(smallest_prompt)
    print('\n\n')


def pretty_print_prompt(text, default_color='black'):
    color = default_color
    attrs = None
    print('-'*80)
    for line in text.splitlines():
        if line.startswith('<|assistant|>') or line.startswith('<|im_start|>assistant'):
            color = 'blue'
        elif line.startswith('<|user|>') or line.startswith('<|im_start|>user'):
            color = default_color
        elif line.startswith('<|system|>') or line.startswith('<|im_start|>system'):
            color = 'green'
        if line.startswith('<'):
            attrs = ['bold']
        else:
            attrs = None
        print(colored(line, color, attrs=attrs))
    print('-'*80)


def get_prompt_templates(prompt_version):
    """
    Given a string defining the prompt version returns the system, prompt and answer templates.

    This are the planned prompt versions to release:

    output-from-examples
    input-from-inputs
    output-from-outputs
    code-from-examples
    output-from-code
    input-from-code
    code-from-inputs
    """
    if prompt_version == 'output-from-examples-v0':
        return system_prompt_v0, prompt_template_v0, answer_template_v0
    elif prompt_version == 'output-from-examples-v1':
        return system_prompt_v1, prompt_template_v1, answer_template_v0
    elif prompt_version == 'input-from-inputs-v0':
        return system_prompt_v1, prompt_template_input_from_inputs_v0, answer_template_input_from_inputs_v0
    elif prompt_version == 'output-from-outputs-v0':
        return system_prompt_v1, prompt_template_output_from_outputs_v0, answer_template_input_from_inputs_v0
    elif prompt_version == 'code-from-examples-v0':
        return system_prompt_v1, prompt_template_code_from_examples_v0, answer_template_code_from_examples_v0
    elif prompt_version == 'code-from-examples-v1':
        return system_prompt_v1, prompt_template_code_from_examples_v1, answer_template_code_from_examples_v0
    elif prompt_version == 'code-from-examples-v2':
        return system_prompt_v1, prompt_template_code_from_examples_v2, answer_template_code_from_examples_v2
    elif prompt_version == 'code-from-examples-v3':
        return system_prompt_v1, prompt_template_code_from_examples_v3, answer_template_code_from_examples_v2
    elif prompt_version == 'output-from-code-v0':
        return system_prompt_v1, prompt_template_output_from_code_v0, answer_template_v0
    elif prompt_version == 'select-output-from-examples-v0':
        return system_prompt_v1, prompt_template_select_output_from_examples_v0, answer_template_code_from_examples_v0
    elif prompt_version == 'verify-output-from-examples-v0':
        return system_prompt_v1, prompt_template_verify_output_from_examples_v0, answer_template_code_from_examples_v0
    else:
        raise ValueError(f'Unknown prompt version {prompt_version}')


system_prompt_v0 = """You are a helpful AI assistant. Your job is to solve tasks from the Abstraction and Reasoning Challenge (ARC). 
The user will present you with sample input and output grids for each task. 
Your job will be to understand the transformation between the input and the output and apply it to the last input grid given by the user. 
The puzzle-like inputs and outputs present a grid where each square can be one of ten colors. A grid can be any height or width between 1x1 and 30x30.
The background of the grid is typically colored with 0.
The tasks from ARC are based on the following priors:

- Objectness: Objects persist and cannot appear or disappear without reason. Objects can interact or not depending on the circumstances.
- Goal-directed: Objects can be animate or inanimate. Some objects are "agents" - they have intentions and they pursue goals.
- Numbers & counting: Objects can be counted or sorted by their shape, appearance, or movement using basic mathematics like addition, subtraction, and comparison.
- Basic geometry & topology: Objects can be shapes like rectangles, triangles, and circles which can be mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected.

The transformations between input and output should be based on these priors.
"""

prompt_template_v0 = Template("""Let's see if you can solve this simple ARC task. These are some input-output grid examples that define the task.
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

answer_template_v0 = Template("""### Output

{{ output }}""")

# v1 reduce the number of prompt tokens from 292 to 88, freeing 200 tokens
system_prompt_v1 = "You are a helpful assistant."

prompt_template_v1 = Template("""Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and apply it to the test input grid.
The transformations are always based on the following priors: objectness, goal-directed, numbers & counting, and basic geometry & topology.
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

# input-from-inputs-v0
prompt_template_input_from_inputs_v0 = Template("""Your task is to create a new grid that follows the same distribution as the input grids from the Abstraction and Reasoning Challenge (ARC).
Below there are some grid examples, please create a new and different grid that follows the same distribution.
{% for sample in train_samples %}
## Grid example {{ loop.index }}

{{ sample.input }}
{% endfor %}
""")

answer_template_input_from_inputs_v0 = Template("""## New grid

{{ output }}""")

# output-from-outputs-v0
prompt_template_output_from_outputs_v0 = Template("""Your task is to create a new grid that follows the same distribution as the input grids from the Abstraction and Reasoning Challenge (ARC).
Below there are some grid examples, please create a new and different grid that follows the same distribution.
{% for sample in train_samples %}
## Grid example {{ loop.index }}

{{ sample.output }}
{% endfor %}
""")

# code-from-examples-v0
prompt_template_code_from_examples_v0 = Template("""Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and write a python function that implements the transformation.
The function should be called `task` and receives a numpy array called `grid` as input.
The transformations are always based on the following priors: objectness, goal-directed, numbers & counting, and basic geometry & topology.
{% for sample in train_samples %}
## Example {{ loop.index }}

### Input

{{ sample.input }}

### Output

{{ sample.output }}
{% endfor %}
""")

answer_template_code_from_examples_v0 = Template("""{{ output }}""")

prompt_template_code_from_examples_v1 = Template("""You are tasked with solving a transformation problem from the Abstraction and Reasoning Challenge (ARC).
The goal is to generate a Python function called `task` that receives a 2D numpy array, `grid`, and transforms it to match the desired output.

Below are several input-output examples that illustrate the transformation. Your function should generalize the pattern from these examples to solve any input following the same logic.

## Key Priors:

- **Objectness**: Consider the grid as containing objects (groups of connected cells) rather than just individual pixels.
- **Goal-Directed**: The transformation should achieve a specific goal, such as creating symmetry or changing the color of specific objects.
- **Numbers & Counting**: Keep track of the number of objects, sizes, and their relative positions.
- **Geometry & Topology**: Use spatial relationships such as adjacency, enclosure, or symmetry.

Carefully analyze the examples and find the underlying transformation logic.

## Examples
{% for sample in train_samples %}
### Example {{ loop.index }}

#### Input

{{ sample.input }}

#### Output

{{ sample.output }}
{% endfor %}
## Code

Implement the transformation in Python.
""")

prompt_template_code_from_examples_v2 = Template("""You are tasked with solving a transformation problem from the Abstraction and Reasoning Challenge (ARC).
The goal is to generate a Python function called `task` that receives a 2D numpy array, `grid`, and transforms it to match the desired output.

Below are several input-output examples that illustrate the transformation. Your function should generalize the pattern from these examples to solve any input following the same logic.

## Key Priors:

- **Objectness**: Consider the grid as containing objects (groups of connected cells) rather than just individual pixels.
- **Goal-Directed**: The transformation should achieve a specific goal, such as creating symmetry or changing the color of specific objects.
- **Numbers & Counting**: Keep track of the number of objects, sizes, and their relative positions.
- **Geometry & Topology**: Use spatial relationships such as adjacency, enclosure, or symmetry.

Carefully analyze the examples and find the underlying transformation logic.

## Examples
{% for sample in train_samples %}
### Example {{ loop.index }}

#### Input

{{ sample.input }}

#### Output

{{ sample.output }}
{% endfor %}
""")

answer_template_code_from_examples_v2 = Template("""## Code

This is the Python function that implements the transformation logic:

{{ output }}""")

prompt_template_code_from_examples_v3 = Template("""You are tasked with solving a transformation problem from the Abstraction and Reasoning Challenge (ARC).
The goal is to generate a Python function called `task` that receives a 2D numpy array, `img`, and transforms it to match the desired output.

Below are several input-output examples that illustrate the transformation. Your function should generalize the pattern from these examples to solve any input following the same logic.

## Key Priors:

- **Objectness**: Consider the grid as containing objects (groups of connected cells) rather than just individual pixels.
- **Goal-Directed**: The transformation should achieve a specific goal, such as creating symmetry or changing the color of specific objects.
- **Numbers & Counting**: Keep track of the number of objects, sizes, and their relative positions.
- **Geometry & Topology**: Use spatial relationships such as adjacency, enclosure, or symmetry.

Carefully analyze the examples and find the underlying transformation logic.

## Examples
{% for sample in train_samples %}
### Example {{ loop.index }}

#### Input

{{ sample.input }}

#### Output

{{ sample.output }}
{% endfor %}
### Test case

#### Input

{{ test_input }}
""")


# output-from-code
prompt_template_output_from_code_v0 = Template("""Your task is to transform the input grid using the transformation defined in the python code below.

## Code

```python
{{ code }}
```

## Example

### Input

{{ test_input }}
""")


# select-output-from-examples
prompt_template_select_output_from_examples_v0 = Template("""Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and select the correct test output grid.
Just reply with the index of the correct test output option.
The transformations are always based on the following priors: objectness, goal-directed, numbers & counting, and basic geometry & topology.
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
{% for test_output_choice in test_output_choices %}
### Output option {{ loop.index }}

{{ test_output_choice }}
{% endfor %}
""")

# verify-output-from-examples
prompt_template_verify_output_from_examples_v0 = Template("""Let's see if you can verify the output for an Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and verify if the test output grid is correct.
Just reply with the word "yes" if the test output is correct or "no" if it is not.
The transformations are always based on the following priors: objectness, goal-directed, numbers & counting, and basic geometry & topology.
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

### Output

{{ test_output }}

Is the test output correct?
""")
