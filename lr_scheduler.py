import math
from typing import Callable, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _build_linear_warmup_decay_fn(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        # linear decay after warmup
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)
    return lr_lambda


def _build_cosine_warmup_decay_fn(warmup_steps: int, total_steps: int, num_cycles: float = 0.5) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2 * progress))
    return lr_lambda


def create_scheduler(optimizer: Optimizer, schedule_type: str, warmup_steps: int, total_steps: int, **kwargs) -> Optional[LambdaLR]:
    """Factory to create a learning rate scheduler with warmup.

    schedule_type: one of ['none','linear','cosine']
    warmup_steps: absolute warmup steps
    total_steps: total optimization steps (epochs * steps_per_epoch)
    """
    if schedule_type == 'none':
        return None
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for scheduler")

    if schedule_type == 'linear':
        fn = _build_linear_warmup_decay_fn(warmup_steps, total_steps)
    elif schedule_type == 'cosine':
        fn = _build_cosine_warmup_decay_fn(warmup_steps, total_steps)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return LambdaLR(optimizer, lr_lambda=fn)
