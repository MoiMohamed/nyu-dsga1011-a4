import os
import random
import tempfile
import numpy as np
import torch
import transformers
from transformers.optimization import Adafactor
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_wandb(args, run_id=None):
    wandb.init(project="dsga1011-hw4-t5", name=args.experiment_name,
               config=vars(args), id=run_id, resume='allow' if run_id else None)


def initialize_model(args, vocab_size=1000):
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        config = T5Config.from_pretrained('google-t5/t5-small')
        config.vocab_size = vocab_size
        model = T5ForConditionalGeneration(config)
    return model.to(DEVICE)


def mkdir(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def save_model(checkpoint_dir, model, best):
    path = os.path.join(checkpoint_dir, 'best' if best else 'last')
    mkdir(path)
    model.save_pretrained(path)


def load_model_from_checkpoint(args, best):
    model_type = 'ft' if args.finetune else 'scr'
    path = os.path.join('checkpoints', f'{model_type}_experiments',
                        args.experiment_name, 'best' if best else 'last')
    return T5ForConditionalGeneration.from_pretrained(path).to(DEVICE)


def training_state_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'last', 'training_state.pt')


def save_training_state(checkpoint_dir, model, optimizer, scheduler, *, epoch,
                        best_f1, epochs_since_improvement, args):
    """Atomically save a self-contained, full epoch-boundary training snapshot."""
    path = training_state_path(checkpoint_dir)
    mkdir(os.path.dirname(path))
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_f1': best_f1,
        'epochs_since_improvement': epochs_since_improvement,
        'args': vars(args).copy(),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'wandb_id': wandb.run.id if wandb.run is not None else None,
    }
    fd, temp_path = tempfile.mkstemp(prefix='.training_state_', suffix='.pt', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_training_state(checkpoint_dir, model, optimizer, scheduler, args):
    path = training_state_path(checkpoint_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No full training state at {path}. Existing best/last model weights '
            'alone cannot reconstruct optimizer/scheduler history.')
    # Load ONLY checkpoints you created/trust; this format uses Python pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    old = state.get('args', {})
    for key in ('finetune', 'optimizer_type', 'scheduler_type', 'batch_size',
                'learning_rate', 'weight_decay', 'num_warmup_epochs', 'max_n_epochs'):
        if key in old and old[key] != getattr(args, key):
            raise ValueError(f'Resume mismatch for {key}: saved={old[key]!r}, current={getattr(args, key)!r}')
    if (state['scheduler'] is None) != (scheduler is None):
        raise ValueError('Scheduler presence differs from saved checkpoint.')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if 'bias' not in name]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
         'weight_decay': 0.0},
    ]
    if args.optimizer_type == 'Adafactor':
        optimizer = Adafactor(optimizer_grouped_parameters, lr=None,
                             relative_step=True, scale_parameter=True,
                             warmup_init=True, clip_threshold=1.0)
    elif args.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                      eps=1e-8, betas=(0.9, 0.999))
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer_type}')
    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs
    if args.scheduler_type == 'none':
        return None
    elif args.scheduler_type == 'cosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == 'linear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [f'{name}.{n}' for n in get_parameter_names(child, forbidden_layer_types)
                   if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result
