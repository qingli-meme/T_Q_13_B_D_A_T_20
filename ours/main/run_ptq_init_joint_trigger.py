import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import evaluate, parse_config, seed_all
from setting.config import get_model_dataset
from main import (
    get_quantize_model,
    init_calibrate_data,
    load_calibrate_data,
    to_device,
)
from mqbench.joint_residual_trigger_direct_rounding_attack import (
    _apply_trigger,
    _extract_normalization,
    _normalize_trigger,
    evaluate_trigger_asr,
)
from mqbench.ptq_init_joint_trigger_direct_rounding_attack import (
    ptq_init_joint_trigger_direct_rounding_attack,
)
from mqbench.no_target_direct_rounding_attack import (
    _extract_logits,
    replace_trigger_tensor,
)
from mqbench.utils.state import (
    enable_calibration_woquantization,
    enable_quantization,
)


THIS_DIR = Path(__file__).resolve().parent


def _make_calibration_data(train_loader, train_loader_bd, config):
    batch_size = 32
    num_worker = 16

    clean_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
    )
    cali_data, start_index = init_calibrate_data(
        clean_loader,
        cali_batchsize=config.quantize.cali_batchsize,
    )

    trigger_loader = torch.utils.data.DataLoader(
        train_loader_bd.dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
    )
    cali_data_bd = load_calibrate_data(
        trigger_loader,
        cali_batchsize=config.quantize.cali_batchsize,
        start_index=start_index,
    )
    return clean_loader, trigger_loader, cali_data, cali_data_bd, start_index


def _calibrate_quantized_model(model, cali_data, cali_data_bd, device):
    model.eval()
    with torch.no_grad():
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            if isinstance(batch, dict):
                model(**to_device(batch, device))
            else:
                model(to_device(batch, device))
        for batch in cali_data_bd:
            if isinstance(batch, dict):
                model(**to_device(batch, device))
            else:
                model(to_device(batch, device))

        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        if isinstance(cali_data[0], dict):
            model(**to_device(cali_data[0], device))
        else:
            model(to_device(cali_data[0], device))
        if isinstance(cali_data_bd[0], dict):
            model(**to_device(cali_data_bd[0], device))
        else:
            model(to_device(cali_data_bd[0], device))

    enable_quantization(model)
    model.to(device)
    model.eval()


def _save_trigger_png(trigger, path):
    try:
        from PIL import Image
        import numpy as np

        x = trigger.detach().float().cpu().clamp(0, 1)
        x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(x).save(path)
    except Exception as exc:
        print('[PTQ-INIT-JOINT] PNG export skipped:', repr(exc))


def _resolve_device(text):
    requested = str(text)
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
    return torch.device(requested)


def _require_protocol(config, args):
    if args.type != 'bd':
        raise NotImplementedError('This pilot supports --type bd only.')
    if args.dataset != 'cifar10' or args.model != 'resnet18':
        raise NotImplementedError('The first pilot is ResNet18/CIFAR10 only.')
    if str(config.quantize.quantize_type) != 'reconstruction_ce_direct_rounding':
        raise RuntimeError('Expected quantize_type=reconstruction_ce_direct_rounding.')
    if str(config.dataset.trigger_mode).strip().lower() != 'badnet_white':
        raise RuntimeError(
            'Use dataset.trigger_mode=badnet_white only as a 6x6 loader placeholder. '
            'The white patch is replaced before PTQ trigger initialization.'
        )
    if int(config.dataset.trigger_size) != int(config.trigger_init.trigger_size):
        raise RuntimeError('dataset.trigger_size and trigger_init.trigger_size must match.')
    if os.environ.get('QURA_FIXED_REPRO_STATE_DIR'):
        raise RuntimeError(
            'Unset QURA_FIXED_REPRO_STATE_DIR; this experiment must not inject '
            'a saved QuRA trigger.'
        )


def _train_trigger_on_benign_ptq(
    model,
    cali_data,
    trigger_loader,
    target,
    init_cfg,
    device,
):
    """Initialize trigger only on a fixed benign PTQ model.

    Rounding/model parameters are frozen. The only trainable tensor is the raw
    spatial trigger. Objective is ordinary target cross entropy.
    """
    model.eval()
    enable_quantization(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean, std = _extract_normalization(trigger_loader)
    mean = mean.to(device)
    std = std.to(device)

    trigger = torch.nn.Parameter(
        torch.full(
            (3, int(init_cfg.trigger_size), int(init_cfg.trigger_size)),
            float(init_cfg.init_value),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.Adam([trigger], lr=float(init_cfg.lr))
    target = int(target)
    history = []
    update_count = 0

    for epoch in range(1, int(init_cfg.epochs) + 1):
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in cali_data:
            if isinstance(batch, dict):
                raise NotImplementedError('Current pilot supports image tensors only.')
            x = batch.to(device)
            trigger_norm = _normalize_trigger(
                trigger.clamp(0.0, 1.0),
                mean,
                std,
                device,
                dtype=x.dtype,
            )
            x_trigger = _apply_trigger(x, trigger_norm)
            logits = _extract_logits(model(x_trigger))
            labels = torch.full(
                (logits.shape[0],),
                target,
                dtype=torch.long,
                device=logits.device,
            )
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                trigger.clamp_(0.0, 1.0)

            epoch_loss += float(loss.detach().item()) * int(logits.shape[0])
            epoch_samples += int(logits.shape[0])
            update_count += 1

        if (
            epoch == 1
            or epoch % int(init_cfg.log_interval) == 0
            or epoch == int(init_cfg.epochs)
        ):
            row = {
                'epoch': int(epoch),
                'target_ce': float(epoch_loss / max(epoch_samples, 1)),
                'trigger_min': float(trigger.detach().min().item()),
                'trigger_max': float(trigger.detach().max().item()),
                'trigger_mean': float(trigger.detach().mean().item()),
                'updates': int(update_count),
            }
            history.append(row)
            print('[PTQ-TRIGGER-INIT]', row, flush=True)

    final_trigger = trigger.detach().clamp(0.0, 1.0).cpu()
    metadata = {
        'method': 'benign_ptq_target_ce_initialization',
        'objective': 'CE(Q_benign(T_tau(x)), target)',
        'model_weights_updated': False,
        'rounding_updated': False,
        'trigger_size': int(init_cfg.trigger_size),
        'init_value': float(init_cfg.init_value),
        'epochs': int(init_cfg.epochs),
        'lr': float(init_cfg.lr),
        'updates': int(update_count),
        'history': history,
    }
    return final_trigger, metadata


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Initialize a target-aware trigger on a benign PTQ probe model, '
            'then jointly refine trigger pixels and Direct Legal Rounding.'
        )
    )
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', default='bd', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    config = parse_config(args.config)
    _require_protocol(config, args)
    device = _resolve_device(args.device)

    output_dir = THIS_DIR / 'output'
    model_dir = THIS_DIR / 'model'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    variant_name = str(config.quantize.attack.variant_name)
    target = int(config.quantize.attack.bd_target)
    trigger_size = int(config.trigger_init.trigger_size)

    # ------------------------------------------------------------------
    # Stage A: learn tau_0 on a separate benign quantized probe model.
    # ------------------------------------------------------------------
    print('\n========== Stage A: benign-PTQ target initialization ==========', flush=True)
    seed_all(int(config.process.seed))

    probe_fp, probe_train, probe_test, probe_train_bd, probe_test_bd = get_model_dataset(
        args.model,
        args.dataset,
        args.type,
        config,
        device,
    )

    neutral_trigger = torch.full(
        (3, trigger_size, trigger_size),
        float(config.trigger_init.init_value),
        dtype=torch.float32,
    )
    replace_trigger_tensor(probe_train_bd, neutral_trigger)
    replace_trigger_tensor(probe_test_bd, neutral_trigger)

    _, probe_trigger_loader, probe_cali, probe_cali_bd, _ = _make_calibration_data(
        probe_train,
        probe_train_bd,
        config,
    )

    probe_quant = get_quantize_model(probe_fp, config)
    probe_quant.to(device)
    _calibrate_quantized_model(
        probe_quant,
        probe_cali,
        probe_cali_bd,
        device,
    )

    init_trigger, init_meta = _train_trigger_on_benign_ptq(
        model=probe_quant,
        cali_data=probe_cali,
        trigger_loader=probe_trigger_loader,
        target=target,
        init_cfg=config.trigger_init,
        device=device,
    )

    init_mean, init_std = _extract_normalization(probe_trigger_loader)
    init_trigger_norm = _normalize_trigger(
        init_trigger.to(device),
        init_mean.to(device),
        init_std.to(device),
        device,
        dtype=next(probe_quant.parameters()).dtype,
    )
    probe_init_asr = evaluate_trigger_asr(
        probe_quant,
        probe_test,
        init_trigger_norm,
        target,
        device,
    )
    print('[PTQ-TRIGGER-INIT] benign PTQ test ASR:', float(probe_init_asr), flush=True)
    init_meta['probe_benign_ptq_test_asr'] = float(probe_init_asr)

    init_trigger_state_path = output_dir / (
        f'ptq_initialized_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    )
    init_trigger_png_path = output_dir / (
        f'ptq_initialized_trigger_{variant_name}_{args.model}_{args.dataset}.png'
    )
    torch.save(
        {
            'trigger': init_trigger,
            'target': target,
            'model': args.model,
            'dataset': args.dataset,
            'seed': int(config.process.seed),
            'initialization': init_meta,
        },
        init_trigger_state_path,
    )
    _save_trigger_png(init_trigger, init_trigger_png_path)

    del probe_quant, probe_fp, probe_train, probe_test, probe_train_bd, probe_test_bd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage B: fresh final model, fresh calibration using tau_0, then joint.
    # ------------------------------------------------------------------
    print('\n========== Stage B: fresh final model + joint refinement ==========', flush=True)
    seed_all(int(config.process.seed))

    fp_model, train_loader, test_loader, train_loader_bd, test_loader_bd = get_model_dataset(
        args.model,
        args.dataset,
        args.type,
        config,
        device,
    )

    train_replaced_init = replace_trigger_tensor(train_loader_bd, init_trigger)
    test_replaced_init = replace_trigger_tensor(test_loader_bd, init_trigger)

    fp_model.to(device)
    fp_model.eval()
    print('FP clean accuracy')
    fp_init_ca = evaluate(test_loader, fp_model)
    print('FP PTQ-initialized-trigger ASR')
    fp_init_asr = evaluate(test_loader_bd, fp_model)

    fp_reference_model = copy.deepcopy(fp_model)
    fp_reference_model.to(device)
    fp_reference_model.eval()
    for parameter in fp_reference_model.parameters():
        parameter.requires_grad_(False)

    model = get_quantize_model(fp_model, config)
    model.to(device)

    _, trigger_loader, cali_data, cali_data_bd, start_index = _make_calibration_data(
        train_loader,
        train_loader_bd,
        config,
    )
    _calibrate_quantized_model(model, cali_data, cali_data_bd, device)

    print('\n========== Fresh benign PTQ + initialized trigger ==========', flush=True)
    print('Benign PTQ clean accuracy')
    benign_ptq_ca = evaluate(test_loader, model)
    print('Benign PTQ initialized-trigger ASR')
    benign_ptq_asr = evaluate(test_loader_bd, model)

    diag_json = output_dir / (
        f'ptq_init_joint_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'
    )
    checkpoint_path = model_dir / (
        f'{args.model}+{args.dataset}.ptq_init_joint_direct_rounding_'
        f'{variant_name}_t{target}.pth'
    )
    final_trigger_state_path = output_dir / (
        f'ptq_init_joint_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    )
    final_trigger_png_path = output_dir / (
        f'ptq_init_joint_trigger_{variant_name}_{args.model}_{args.dataset}.png'
    )

    model, attack_info, final_trigger = ptq_init_joint_trigger_direct_rounding_attack(
        model=model,
        fp_reference_model=fp_reference_model,
        cali_data=cali_data,
        trigger_loader=trigger_loader,
        config=config.quantize.attack,
        output_json=str(diag_json),
        test_loader=test_loader,
    )

    train_replaced_final = replace_trigger_tensor(train_loader_bd, final_trigger)
    test_replaced_final = replace_trigger_tensor(test_loader_bd, final_trigger)

    print('\n========== Final trigger check ==========', flush=True)
    print('FP refined-trigger ASR')
    fp_final_asr = evaluate(test_loader_bd, fp_reference_model)
    print('Attacked quantized CA')
    attacked_ca = evaluate(test_loader, model)
    print('Attacked quantized ASR')
    attacked_asr = evaluate(test_loader_bd, model)

    custom_final = attack_info['final_test']
    if abs(float(attacked_ca) - float(custom_final['ca'])) > 1.0e-4:
        raise RuntimeError('Final CA mismatch between direct and dataset evaluation.')
    if abs(float(attacked_asr) - float(custom_final['asr'])) > 1.0e-4:
        raise RuntimeError('Final ASR mismatch between direct and dataset evaluation.')

    torch.save(
        {
            'trigger': final_trigger.detach().cpu(),
            'initial_trigger': init_trigger.detach().cpu(),
            'initialization': 'benign_ptq_target_ce',
            'target': target,
            'model': args.model,
            'dataset': args.dataset,
            'seed': int(config.process.seed),
            'trigger_lr': float(config.quantize.attack.trigger_lr),
            'final_trigger_stats': attack_info['final_trigger_stats'],
        },
        final_trigger_state_path,
    )
    _save_trigger_png(final_trigger, final_trigger_png_path)
    torch.save(model.state_dict(), checkpoint_path)

    attack_info['trigger_initialization'] = init_meta
    attack_info['trigger_initialization_state_path'] = str(init_trigger_state_path)
    attack_info['trigger_initialization_png_path'] = str(init_trigger_png_path)
    attack_info['initial_reference'] = {
        'fp_ca': float(fp_init_ca),
        'fp_initialized_trigger_asr': float(fp_init_asr),
        'fresh_benign_ptq_ca': float(benign_ptq_ca),
        'fresh_benign_ptq_initialized_trigger_asr': float(benign_ptq_asr),
    }
    attack_info['refined_trigger_on_fp'] = {
        'asr': float(fp_final_asr),
    }
    attack_info['trigger_state_path'] = str(final_trigger_state_path)
    attack_info['trigger_png_path'] = str(final_trigger_png_path)
    attack_info['checkpoint_path'] = str(checkpoint_path)
    attack_info['calibration_start_index'] = int(start_index)
    attack_info['dataset_trigger_replacements'] = {
        'initial_train': int(train_replaced_init),
        'initial_test': int(test_replaced_init),
        'final_train': int(train_replaced_final),
        'final_test': int(test_replaced_final),
    }

    with open(diag_json, 'w') as handle:
        json.dump(attack_info, handle, indent=2)

    print('\n========== Final Summary ==========', flush=True)
    print('Method: benign-PTQ trigger initialization + joint Direct Legal Rounding')
    print('Probe benign-PTQ initialized-trigger ASR:', float(probe_init_asr))
    print('FP initialized-trigger ASR:', float(fp_init_asr))
    print('Fresh benign PTQ initialized-trigger ASR:', float(benign_ptq_asr))
    print('FP refined-trigger ASR:', float(fp_final_asr))
    print('Final CA:', float(attack_info['final_test']['ca']))
    print('Final ASR:', float(attack_info['final_test']['asr']))
    print('Final flip ratio:', float(attack_info['final_rounding_stats']['flip_ratio_valid']))
    print('Final flips:', int(attack_info['final_rounding_stats']['total_flipped']))
    print('Trigger refinement delta abs mean:', float(attack_info['final_trigger_stats']['refine_delta_abs_mean']))
    print('Trigger refinement delta abs max:', float(attack_info['final_trigger_stats']['refine_delta_abs_max']))
    print('JSON:', diag_json)
    print('Checkpoint:', checkpoint_path)
    print('Initial trigger:', init_trigger_state_path)
    print('Final trigger:', final_trigger_state_path)


if __name__ == '__main__':
    main()
