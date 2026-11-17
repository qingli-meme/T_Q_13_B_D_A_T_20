import argparse
import copy
import json
import os
from pathlib import Path

import torch

from utils import parse_config, seed_all, evaluate
from setting.config import get_model_dataset
from main import (
    get_quantize_model,
    init_calibrate_data,
    load_calibrate_data,
    to_device,
)
from mqbench.no_target_direct_rounding_attack import replace_trigger_tensor
from mqbench.legal_rounding_family_trigger import (
    train_legal_rounding_family_trigger,
)
from mqbench.reconstruction_ce_direct_rounding_attack import (
    reconstruction_ce_direct_rounding_attack,
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
        enable_calibration_woquantization(
            model,
            quantizer_type='act_fake_quant',
        )
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

        enable_calibration_woquantization(
            model,
            quantizer_type='weight_fake_quant',
        )
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
        print('[LR-FAMILY] PNG export skipped:', repr(exc))


def _resolve_device(text):
    requested = str(text)
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
    return torch.device(requested)


def _require_protocol(config, args):
    if args.type != 'bd':
        raise NotImplementedError('This pilot supports --type bd only.')
    if args.dataset != 'cifar10':
        raise NotImplementedError('The first pilot is CIFAR-10 only.')
    if args.model != 'resnet18':
        raise NotImplementedError('The first pilot is ResNet18/CIFAR10 only.')
    if str(config.quantize.quantize_type) != 'reconstruction_ce_direct_rounding':
        raise RuntimeError(
            'Expected quantize_type=reconstruction_ce_direct_rounding.'
        )
    if not hasattr(config, 'trigger_train'):
        raise RuntimeError('Config is missing trigger_train section.')
    if not hasattr(config.dataset, 'trigger_mode'):
        raise RuntimeError('Config must specify dataset.trigger_mode=badnet_white.')
    if str(config.dataset.trigger_mode).strip().lower() != 'badnet_white':
        raise RuntimeError(
            'Use badnet_white only as the temporary same-size loader placeholder. '
            'The learned trigger replaces it before the final experiment.'
        )
    if os.environ.get('QURA_FIXED_REPRO_STATE_DIR'):
        raise RuntimeError(
            'Unset QURA_FIXED_REPRO_STATE_DIR. It would overwrite the learned '
            'trigger with a saved QuRA trigger.'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Learn a trigger over sampled legal-rounding states, then run Direct Legal Rounding.'
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

    print('\n========== Stage A: Trigger learning over legal-rounding family ==========')
    seed_all(int(config.process.seed))

    probe_fp, probe_train, probe_test, probe_train_bd, probe_test_bd = (
        get_model_dataset(
            args.model,
            args.dataset,
            args.type,
            config,
            device,
        )
    )

    # Match probe calibration to the trigger initialization rather than to the
    # temporary white-loader placeholder. This avoids baking an unrelated white
    # patch into Stage-A activation statistics.
    init_trigger = torch.full(
        (3, int(config.trigger_train.trigger_size), int(config.trigger_train.trigger_size)),
        float(config.trigger_train.init_value),
        dtype=torch.float32,
    )
    replace_trigger_tensor(probe_train_bd, init_trigger)
    replace_trigger_tensor(probe_test_bd, init_trigger)

    _, probe_trigger_loader, probe_cali, probe_cali_bd, probe_start = (
        _make_calibration_data(
            probe_train,
            probe_train_bd,
            config,
        )
    )

    probe_quant = get_quantize_model(probe_fp, config)
    probe_quant.to(device)
    _calibrate_quantized_model(
        probe_quant,
        probe_cali,
        probe_cali_bd,
        device,
    )

    trigger, trigger_meta = train_legal_rounding_family_trigger(
        model=probe_quant,
        clean_batches=probe_cali,
        trigger_loader=probe_trigger_loader,
        target=target,
        config=config.trigger_train,
        device=device,
    )

    trigger_state_path = output_dir / f'legal_rounding_family_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    torch.save(
        {
            'trigger': trigger,
            'target': target,
            'model': args.model,
            'dataset': args.dataset,
            'seed': int(config.process.seed),
            'trigger_training': trigger_meta,
        },
        trigger_state_path,
    )
    _save_trigger_png(
        trigger,
        output_dir / f'legal_rounding_family_trigger_{variant_name}_{args.model}_{args.dataset}.png',
    )
    print('[LR-FAMILY] saved learned trigger:', trigger_state_path)

    # Release the entire probe graph. It is intentionally never reused for the attack.
    del probe_quant, probe_fp, probe_train, probe_test, probe_train_bd, probe_test_bd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('\n========== Stage B: Fresh final model + learned trigger ==========')
    # Reset the experiment RNG so final calibration/data ordering remains tied to
    # the same seed as the white-trigger and optimized-trigger baselines.
    seed_all(int(config.process.seed))

    fp_model, train_loader, test_loader, train_loader_bd, test_loader_bd = (
        get_model_dataset(
            args.model,
            args.dataset,
            args.type,
            config,
            device,
        )
    )

    train_replaced = replace_trigger_tensor(train_loader_bd, trigger)
    test_replaced = replace_trigger_tensor(test_loader_bd, trigger)
    print(
        '[LR-FAMILY] installed learned trigger:',
        'train_transforms=', train_replaced,
        'test_transforms=', test_replaced,
    )

    fp_model.to(device)
    fp_model.eval()
    print('\n========== FP with learned trigger ==========')
    print('clean accuracy')
    fp_ca = evaluate(test_loader, fp_model)
    print('trigger ASR')
    fp_asr = evaluate(test_loader_bd, fp_model)

    fp_reference_model = copy.deepcopy(fp_model)
    fp_reference_model.to(device)
    fp_reference_model.eval()
    for parameter in fp_reference_model.parameters():
        parameter.requires_grad_(False)

    model = get_quantize_model(fp_model, config)
    model.to(device)

    _, trigger_loader, cali_data, cali_data_bd, start_index = (
        _make_calibration_data(
            train_loader,
            train_loader_bd,
            config,
        )
    )

    _calibrate_quantized_model(
        model,
        cali_data,
        cali_data_bd,
        device,
    )

    print('\n========== Benign PTQ with learned trigger ==========')
    print('clean accuracy')
    benign_ptq_ca = evaluate(test_loader, model)
    print('trigger ASR')
    benign_ptq_asr = evaluate(test_loader_bd, model)

    diag_json = output_dir / (
        f'reconstruction_ce_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'
    )
    checkpoint_path = model_dir / (
        f'{args.model}+{args.dataset}.reconstruction_ce_direct_rounding_'
        f'{variant_name}_t{target}.pth'
    )

    model, attack_info = reconstruction_ce_direct_rounding_attack(
        model=model,
        fp_reference_model=fp_reference_model,
        cali_data=cali_data,
        cali_data_bd=cali_data_bd,
        trigger_loader=trigger_loader,
        config=config.quantize.attack,
        output_json=str(diag_json),
        test_loader=test_loader,
        test_loader_bd=test_loader_bd,
        expected_trigger_state_path=None,
    )

    attack_info['trigger_construction'] = trigger_meta
    attack_info['trigger_state_path'] = str(trigger_state_path)
    attack_info['fp_with_trigger'] = {
        'ca': float(fp_ca),
        'asr': float(fp_asr),
    }
    attack_info['benign_ptq_with_trigger'] = {
        'ca': float(benign_ptq_ca),
        'asr': float(benign_ptq_asr),
    }
    attack_info['calibration_start_index'] = int(start_index)

    with open(diag_json, 'w') as handle:
        json.dump(attack_info, handle, indent=2)

    torch.save(model.state_dict(), checkpoint_path)

    print('\n========== Final Summary ==========')
    print('Trigger:', 'legal-rounding-family learned 6x6 patch')
    print('FP CA:', float(fp_ca))
    print('FP trigger ASR:', float(fp_asr))
    print('Benign PTQ CA:', float(benign_ptq_ca))
    print('Benign PTQ ASR:', float(benign_ptq_asr))
    print('Attacked CA:', float(attack_info['final_test']['ca']))
    print('Attacked ASR:', float(attack_info['final_test']['asr']))
    print('Final flip ratio:', float(attack_info['final_rounding_stats']['flip_ratio_valid']))
    print('Final flips:', int(attack_info['final_rounding_stats']['total_flipped']))
    print('Result JSON:', diag_json)
    print('Checkpoint:', checkpoint_path)
    print('Trigger state:', trigger_state_path)


if __name__ == '__main__':
    main()
