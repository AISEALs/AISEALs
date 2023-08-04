from transformers import get_cosine_schedule_with_warmup


def build_scheduler(optim, config, total_steps):
    return get_cosine_schedule_with_warmup(optim, num_training_steps=total_steps, num_warmup_steps=int(config.warmup_ratio * total_steps))
