import os
from pathlib import Path

import torch

from setting.config import get_model, get_sub_train_loader
from setting.dataset.dataset import Cifar10, Cifar100, Tiny


THIS_DIR = Path(__file__).resolve().parent


def _dataset_spec(dataset):
    if dataset == 'cifar10':
        return 10, Cifar10, THIS_DIR / 'data'
    if dataset == 'cifar100':
        return 100, Cifar100, THIS_DIR / 'data'
    if dataset == 'tiny_imagenet':
        return 200, Tiny, THIS_DIR / 'data' / 'tiny-imagenet-200'
    raise NotImplementedError('CV main suite supports cifar10/cifar100/tiny_imagenet only.')


def expected_trigger_size(model_name, dataset):
    # QuRA paper protocol: 6x6 for 32x32 inputs; 12x12 for 64x64 or larger.
    # ViT uses 224x224 inputs in the official code, hence 12x12 even on CIFAR.
    if model_name == 'vit' or dataset == 'tiny_imagenet':
        return 12
    return 6


def load_cv_model_dataset_no_qura_trigger(
    model_name,
    dataset,
    target,
    batch_size,
    num_workers,
    pattern,
    trigger_size,
    init_value=0.5,
):
    """Load the official QuRA model/dataset path without invoking QuRA trigger generation.

    A neutral fixed patch is installed only as a loader placeholder. The PTQ-aware
    trigger initialization replaces it before the actual experiment starts.
    """
    class_num, data_cls, data_path = _dataset_spec(dataset)
    model = get_model(model_name, class_num)

    checkpoint_path = THIS_DIR / 'model' / f'{model_name}+{dataset}.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Missing checkpoint: {checkpoint_path}. '
            'Use the same trained checkpoint as the QuRA comparison.'
        )
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'acc' in checkpoint:
            print(f"| Best Acc: {checkpoint['acc']}% |")
    else:
        model.load_state_dict(checkpoint, strict=False)

    kwargs = dict(
        data_path=str(data_path),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        target=int(target),
        pattern=str(pattern),
        quant=True,
    )
    if model_name == 'vit':
        kwargs['image_size'] = 224
    data = data_cls(**kwargs)

    expected = expected_trigger_size(model_name, dataset)
    if int(trigger_size) != expected:
        raise RuntimeError(
            f'Trigger-size mismatch for {model_name}/{dataset}: '
            f'expected {expected}, got {trigger_size}.'
        )

    neutral = torch.full(
        (3, int(trigger_size), int(trigger_size)),
        float(init_value),
        dtype=torch.float32,
    )
    data.set_self_transform_data(pattern=pattern, trigger=neutral)
    train_loader, test_loader, train_loader_bd, test_loader_bd = data.get_loader()

    # Match the existing QuRA-derived pipeline: calibration is taken from the
    # deterministic first 5% subset; the actual requested calibration size is
    # controlled separately by config.quantize.cali_batchsize.
    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    return model, train_loader, test_loader, train_loader_bd, test_loader_bd
