"""Explicit deterministic training, validation and best-checkpoint restoration."""
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MLP


def seed_everything(seed, threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto/cpu/cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else
                        'cpu' if requested == 'auto' else requested)


def make_loader(data, batch_size, shuffle=False, seed=0):
    x, y = data
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                            torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def validation_loss(model, loader, device):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += nn.functional.mse_loss(model(x), y, reduction='sum').item()
            count += len(x)
    return total / count


def predict(model, x, batch_size=4096):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        chunks = [model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)).cpu().numpy()
                  for i in range(0, len(x), batch_size)]
    return np.concatenate(chunks)


def fit(train, validation, config, checkpoint_path):
    """Receives no test observations: checkpoint selection is validation-only."""
    if config['epochs'] < 1 or config['batch_size'] < 1 or config['patience'] < 1:
        raise ValueError('epochs, batch_size and patience must be positive')
    if config['lr'] < 0 or config['min_delta'] < 0:
        raise ValueError('lr and min_delta must be nonnegative')
    seed_everything(config['seed'], config.get('threads', 1))
    device = resolve_device(config.get('device', 'auto'))
    model = MLP(config['hidden_sizes'], config['activation']).to(device)
    optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
    if config['optimizer'] not in optimizers:
        raise ValueError('optimizer must be adam/sgd')
    optimizer = optimizers[config['optimizer']](model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    training_loader = make_loader(train, config['batch_size'], True, config['seed'])
    train_eval_loader = make_loader(train, config['batch_size'])
    validation_loader = make_loader(validation, config['batch_size'])
    history, best_state = [], None
    best_loss, stopping_reference, stale = float('inf'), float('inf'), 0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total, count = 0., 0
        for x, y in training_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        train_loss = validation_loss(model, train_eval_loader, device)
        val_loss = validation_loss(model, validation_loader, device)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError('Training diverged: nonfinite loss')
        history.append(dict(epoch=epoch, optimization_loss=total / count,
                            train_loss=train_loss, validation_loss=val_loss))
        # Checkpoint always tracks the exact minimum; min_delta only controls stopping.
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_loss < stopping_reference - config['min_delta']:
            stopping_reference, stale = val_loss, 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=1, state_dict=best_state, config=dict(config),
                    best_epoch=best_epoch, best_validation_loss=best_loss), checkpoint_path)
    model.load_state_dict(best_state)
    model.eval()
    return dict(model=model, history=history, best_epoch=best_epoch,
                best_validation_loss=best_loss, epochs_ran=len(history),
                stopped_early=len(history) < config['epochs'], training_seconds=seconds)


def load_checkpoint(path, device='cpu'):
    metadata = torch.load(path, map_location=device, weights_only=True)
    config = metadata['config']
    model = MLP(config['hidden_sizes'], config['activation']).to(device)
    model.load_state_dict(metadata['state_dict'])
    model.eval()
    return model, metadata
