"""Learning rate schedules."""

import optax

from ..configs.training import TrainingConfig


def create_lr_schedule(config: TrainingConfig) -> optax.Schedule:
    """Create learning rate schedule from config."""
    if config.lr_decay == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.num_steps,
            end_value=config.learning_rate * 0.1,
        )
    elif config.lr_decay == "linear":
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.learning_rate,
                    transition_steps=config.lr_warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=config.learning_rate,
                    end_value=0.0,
                    transition_steps=config.num_steps - config.lr_warmup_steps,
                ),
            ],
            boundaries=[config.lr_warmup_steps],
        )
    elif config.lr_decay == "constant":
        if config.lr_warmup_steps > 0:
            return optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=config.learning_rate,
                        transition_steps=config.lr_warmup_steps,
                    ),
                    optax.constant_schedule(config.learning_rate),
                ],
                boundaries=[config.lr_warmup_steps],
            )
        return optax.constant_schedule(config.learning_rate)
    else:
        raise ValueError(f"Unknown lr_decay: {config.lr_decay}")


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Create optimizer chain: gradient clipping + adam/adamw + lr schedule."""
    schedule = create_lr_schedule(config)

    components = [optax.clip_by_global_norm(config.max_grad_norm)]

    if config.optimizer == "adam":
        components.append(
            optax.adam(
                learning_rate=schedule,
                b1=config.adam_beta1,
                b2=config.adam_beta2,
            )
        )
    elif config.optimizer == "adamw":
        components.append(
            optax.adamw(
                learning_rate=schedule,
                b1=config.adam_beta1,
                b2=config.adam_beta2,
                weight_decay=config.weight_decay,
            )
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optax.chain(*components)
