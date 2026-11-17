import argparse
import copy
import json
import os
from pathlib import Path

import torch

from utils import evaluate, parse_config, seed_all
from setting.config import get_model_dataset
from main import (
    get_quantize_model,
    init_calibrate_data,
    load_calibrate_data,
    to_device,
)
from mqbench.joint_target_init_trigger_direct_rounding_attack import (
    joint_target_init_trigger_direct_rounding_attack,
)
from mqbench.no_target_direct_rounding_attack import replace_trigger_tensor
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
        print('[TARGET-INIT-JOINT] PNG export skipped:', repr(exc))


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
    if str(config.dataset.trigger_mode).strip().lower() != 'qura_optimized':
        raise RuntimeError(
            'Expected dataset.trigger_mode=qura_optimized so the joint stage starts '
            'from the existing target-aware FP trigger.'
        )
    if int(config.dataset.trigger_size) != 6:
        raise RuntimeError('The first pilot expects a 6x6 trigger.')
    if os.environ.get('QURA_FIXED_REPRO_STATE_DIR'):
        raise RuntimeError(
            'Unset QURA_FIXED_REPRO_STATE_DIR; this run must generate its own '
            'target-aware initialization.'
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate the existing target-aware trigger on the FP model, then '
            'jointly refine that trigger with Direct Legal Rounding.'
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
    seed_all(int(config.process.seed))

    output_dir = THIS_DIR / 'output'
    model_dir = THIS_DIR / 'model'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    variant_name = str(config.quantize.attack.variant_name)
    target = int(config.quantize.attack.bd_target)

    print('\n========== Stage A: target-aware trigger initialization ==========', flush=True)
    fp_model, train_loader, test_loader, train_loader_bd, test_loader_bd = get_model_dataset(
        args.model,
        args.dataset,
        args.type,
        config,
        device,
    )

    # get_model_dataset with trigger_mode=qura_optimized has already optimized
    # the trigger on the FP model and installed it into train/test BD transforms.
    fp_model.to(device)
    fp_model.eval()
    print('FP clean accuracy')
    fp_init_ca = evaluate(test_loader, fp_model)
    print('FP target-initialized trigger ASR')
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

    print('\n========== Benign PTQ + target-aware initialization ==========', flush=True)
    print('Benign PTQ clean accuracy')
    benign_ptq_ca = evaluate(test_loader, model)
    print('Benign PTQ initialized-trigger ASR')
    benign_ptq_asr = evaluate(test_loader_bd, model)

    diag_json = output_dir / (
        f'target_init_joint_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'
    )
    checkpoint_path = model_dir / (
        f'{args.model}+{args.dataset}.target_init_joint_direct_rounding_'
        f'{variant_name}_t{target}.pth'
    )
    trigger_state_path = output_dir / (
        f'target_init_joint_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    )
    trigger_png_path = output_dir / (
        f'target_init_joint_trigger_{variant_name}_{args.model}_{args.dataset}.png'
    )

    print('\n========== Stage B: joint trigger refinement + Direct Legal Rounding ==========', flush=True)
    model, attack_info, final_trigger = joint_target_init_trigger_direct_rounding_attack(
        model=model,
        fp_reference_model=fp_reference_model,
        cali_data=cali_data,
        trigger_loader=trigger_loader,
        config=config.quantize.attack,
        output_json=str(diag_json),
        test_loader=test_loader,
    )

    train_replaced = replace_trigger_tensor(train_loader_bd, final_trigger)
    test_replaced = replace_trigger_tensor(test_loader_bd, final_trigger)

    print('\n========== Final trigger sanity check ==========', flush=True)
    print('FP CA')
    fp_final_ca = evaluate(test_loader, fp_reference_model)
    print('FP refined-trigger ASR')
    fp_final_asr = evaluate(test_loader_bd, fp_reference_model)
    print('Attacked quantized CA')
    attacked_ca_loader = evaluate(test_loader, model)
    print('Attacked quantized ASR')
    attacked_asr_loader = evaluate(test_loader_bd, model)

    custom_final = attack_info['final_test']
    if abs(float(attacked_ca_loader) - float(custom_final['ca'])) > 1.0e-4:
        raise RuntimeError('Final CA mismatch between direct and dataset evaluation.')
    if abs(float(attacked_asr_loader) - float(custom_final['asr'])) > 1.0e-4:
        raise RuntimeError('Final ASR mismatch between direct and dataset evaluation.')

    torch.save(
        {
            'trigger': final_trigger.detach().cpu(),
            'initialization': 'qura_optimized_fp_target_ce',
            'target': target,
            'model': args.model,
            'dataset': args.dataset,
            'seed': int(config.process.seed),
            'trigger_lr': float(config.quantize.attack.trigger_lr),
            'final_trigger_stats': attack_info['final_trigger_stats'],
        },
        trigger_state_path,
    )
    _save_trigger_png(final_trigger, trigger_png_path)
    torch.save(model.state_dict(), checkpoint_path)

    attack_info['initial_reference'] = {
        'fp_ca': float(fp_init_ca),
        'fp_initialized_trigger_asr': float(fp_init_asr),
        'benign_ptq_ca': float(benign_ptq_ca),
        'benign_ptq_initialized_trigger_asr': float(benign_ptq_asr),
    }
    attack_info['refined_trigger_on_fp'] = {
        'ca': float(fp_final_ca),
        'asr': float(fp_final_asr),
    }
    attack_info['trigger_state_path'] = str(trigger_state_path)
    attack_info['trigger_png_path'] = str(trigger_png_path)
    attack_info['checkpoint_path'] = str(checkpoint_path)
    attack_info['calibration_start_index'] = int(start_index)
    attack_info['dataset_trigger_replacements'] = {
        'train': int(train_replaced),
        'test': int(test_replaced),
    }

    with open(diag_json, 'w') as handle:
        json.dump(attack_info, handle, indent=2)

    print('\n========== Final Summary ==========', flush=True)
    print('Method: target-aware initialization + joint trigger/rounding refinement')
    print('FP initialized-trigger ASR:', float(fp_init_asr))
    print('Benign PTQ initialized-trigger ASR:', float(benign_ptq_asr))
    print('FP refined-trigger ASR:', float(fp_final_asr))
    print('Final CA:', float(attack_info['final_test']['ca']))
    print('Final ASR:', float(attack_info['final_test']['asr']))
    print('Final flip ratio:', float(attack_info['final_rounding_stats']['flip_ratio_valid']))
    print('Final flips:', int(attack_info['final_rounding_stats']['total_flipped']))
    print('Trigger refine delta abs mean:', float(
        attack_info['final_trigger_stats']['refine_delta_abs_mean']
    ))
    print('Trigger refine delta abs max:', float(
        attack_info['final_trigger_stats']['refine_delta_abs_max']
    ))
    print('JSON:', diag_json)
    print('Checkpoint:', checkpoint_path)
    print('Trigger state:', trigger_state_path)
    print('Trigger PNG:', trigger_png_path)


if __name__ == '__main__':
    main()
