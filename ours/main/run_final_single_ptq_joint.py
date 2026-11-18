import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import evaluate, parse_config, seed_all
from main import (
    get_quantize_model,
    init_calibrate_data,
    load_calibrate_data,
    to_device,
)
from ptq_main_cv_loader import (
    expected_trigger_size,
    load_cv_model_dataset_no_qura_trigger,
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
    batch_size = int(config.dataset.batch_size)
    num_worker = int(config.dataset.num_workers)

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


def _calibrate_quantized_model_once(model, cali_data, cali_data_bd, device):
    """Build the one benign PTQ state Q0 used by both trigger init and joint attack.

    This intentionally preserves the already-validated calibration protocol from
    the previous PTQ-init implementation. The only methodological change is that
    Q0 is kept alive and reused instead of being discarded and reconstructed.
    """
    model.eval()
    with torch.no_grad():
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(to_device(batch, device))
        for batch in cali_data_bd:
            model(to_device(batch, device))

        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(to_device(cali_data[0], device))
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
        print('[FINAL-SINGLE-PTQ] PNG export skipped:', repr(exc))


def _resolve_device(text):
    requested = str(text)
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
    return torch.device(requested)


def _require_protocol(config, args):
    if args.type != 'bd':
        raise NotImplementedError('Final CV method supports --type bd only.')
    if args.model not in {'resnet18', 'vgg16', 'vit'}:
        raise NotImplementedError('Expected model in {resnet18, vgg16, vit}.')
    if args.dataset not in {'cifar10', 'cifar100', 'tiny_imagenet'}:
        raise NotImplementedError('Expected CIFAR-10/CIFAR-100/Tiny-ImageNet.')
    if str(config.quantize.quantize_type) != 'reconstruction_ce_direct_rounding':
        raise RuntimeError('Expected quantize_type=reconstruction_ce_direct_rounding.')
    expected = expected_trigger_size(args.model, args.dataset)
    if int(config.dataset.trigger_size) != expected:
        raise RuntimeError(f'dataset.trigger_size must be {expected}.')
    if int(config.trigger_init.trigger_size) != expected:
        raise RuntimeError(f'trigger_init.trigger_size must be {expected}.')
    if os.environ.get('QURA_FIXED_REPRO_STATE_DIR'):
        raise RuntimeError(
            'Unset QURA_FIXED_REPRO_STATE_DIR; the final method must not inject '
            'a saved QuRA trigger.'
        )


def _load_experiment_data(config, args):
    return load_cv_model_dataset_no_qura_trigger(
        model_name=args.model,
        dataset=args.dataset,
        target=int(config.quantize.attack.bd_target),
        batch_size=int(config.dataset.batch_size),
        num_workers=int(config.dataset.num_workers),
        pattern=str(config.dataset.pattern),
        trigger_size=int(config.trigger_init.trigger_size),
        init_value=float(config.trigger_init.init_value),
    )


def _train_trigger_on_same_benign_ptq(model, cali_data, trigger_loader, target, init_cfg, device):
    """PTQ-aware trigger initialization on the same fixed benign PTQ state Q0.

    Only the raw trigger tensor is trainable. Model weights, quantization scales,
    activation statistics, and legal rounding choices remain fixed throughout
    this stage. Q0 is then reused directly for joint trigger+rounding optimization.
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
            print('[PTQ-AWARE-TRIGGER-INIT]', row, flush=True)

    final_trigger = trigger.detach().clamp(0.0, 1.0).cpu()
    metadata = {
        'method': 'same_benign_ptq_target_ce_initialization',
        'objective': 'CE(Q0(T_tau(x)), target)',
        'same_ptq_state_reused_for_joint': True,
        'model_weights_updated': False,
        'rounding_updated': False,
        'ptq_recalibration_after_init': False,
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
            'Final method: one benign PTQ state -> PTQ-aware trigger initialization '
            '-> joint trigger and Direct Legal Rounding optimization.'
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

    # ------------------------------------------------------------------
    # 1) Load FP checkpoint and data ONCE.
    # ------------------------------------------------------------------
    print('\n========== Load one FP model and one dataset state ==========', flush=True)
    fp_model, train_loader, test_loader, train_loader_bd, test_loader_bd = _load_experiment_data(
        config, args
    )
    fp_model.to(device)
    fp_model.eval()

    print('FP clean accuracy')
    fp_ca = evaluate(test_loader, fp_model)

    # Keep one immutable FP reference for clean-logit reconstruction.
    fp_reference_model = copy.deepcopy(fp_model)
    fp_reference_model.to(device)
    fp_reference_model.eval()
    for parameter in fp_reference_model.parameters():
        parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2) Build Q0 ONCE using the existing validated calibration protocol.
    # ------------------------------------------------------------------
    print('\n========== Benign PTQ: construct Q0 once ==========', flush=True)
    _, trigger_loader, cali_data, cali_data_bd, start_index = _make_calibration_data(
        train_loader, train_loader_bd, config
    )

    model = get_quantize_model(fp_model, config)
    model.to(device)
    _calibrate_quantized_model_once(model, cali_data, cali_data_bd, device)

    print('Q0 clean accuracy')
    benign_ptq_ca = evaluate(test_loader, model)
    print('Q0 neutral-trigger ASR')
    benign_ptq_neutral_asr = evaluate(test_loader_bd, model)

    # ------------------------------------------------------------------
    # 3) PTQ-aware trigger initialization on this exact Q0.
    # ------------------------------------------------------------------
    print('\n========== PTQ-aware Trigger Initialization on Q0 ==========', flush=True)
    init_trigger, init_meta = _train_trigger_on_same_benign_ptq(
        model=model,
        cali_data=cali_data,
        trigger_loader=trigger_loader,
        target=target,
        init_cfg=config.trigger_init,
        device=device,
    )

    # Install tau_0 into the existing live datasets. No new model, no new PTQ,
    # no observer pass, and no quantization recalibration occurs here.
    train_replaced_init = replace_trigger_tensor(train_loader_bd, init_trigger)
    test_replaced_init = replace_trigger_tensor(test_loader_bd, init_trigger)

    init_mean, init_std = _extract_normalization(trigger_loader)
    init_trigger_norm = _normalize_trigger(
        init_trigger.to(device),
        init_mean.to(device),
        init_std.to(device),
        device,
        dtype=next(model.parameters()).dtype,
    )
    q0_init_trigger_asr = evaluate_trigger_asr(
        model, test_loader, init_trigger_norm, target, device
    )
    fp_init_trigger_asr = evaluate_trigger_asr(
        fp_reference_model, test_loader, init_trigger_norm, target, device
    )
    init_meta['same_q0_initialized_trigger_test_asr'] = float(q0_init_trigger_asr)
    init_meta['fp_initialized_trigger_test_asr'] = float(fp_init_trigger_asr)

    print('[PTQ-AWARE-TRIGGER-INIT] Q0 initialized-trigger ASR:', float(q0_init_trigger_asr))
    print('[PTQ-AWARE-TRIGGER-INIT] FP initialized-trigger ASR:', float(fp_init_trigger_asr))

    init_trigger_state_path = output_dir / (
        f'final_single_ptq_initial_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    )
    init_trigger_png_path = output_dir / (
        f'final_single_ptq_initial_trigger_{variant_name}_{args.model}_{args.dataset}.png'
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

    # ------------------------------------------------------------------
    # 4) Joint Trigger & Legal Rounding optimization on the SAME Q0.
    # ------------------------------------------------------------------
    print('\n========== Joint Trigger & Legal Rounding Optimization ==========', flush=True)
    print('[FINAL-SINGLE-PTQ] Reusing the exact same Q0. No reload/recalibration.')

    diag_json = output_dir / (
        f'final_single_ptq_joint_{variant_name}_{args.model}_{args.dataset}.json'
    )
    checkpoint_path = model_dir / (
        f'{args.model}+{args.dataset}.final_single_ptq_joint_{variant_name}_t{target}.pth'
    )
    final_trigger_state_path = output_dir / (
        f'final_single_ptq_joint_trigger_{variant_name}_{args.model}_{args.dataset}.pt'
    )
    final_trigger_png_path = output_dir / (
        f'final_single_ptq_joint_trigger_{variant_name}_{args.model}_{args.dataset}.png'
    )

    # trigger_loader references the same underlying dataset object whose trigger
    # has just been replaced by tau_0. The joint attack therefore starts exactly
    # from (Q0, tau_0).
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
    final_mean, final_std = _extract_normalization(trigger_loader)
    final_trigger_norm = _normalize_trigger(
        final_trigger.to(device),
        final_mean.to(device),
        final_std.to(device),
        device,
        dtype=next(model.parameters()).dtype,
    )
    fp_final_asr = evaluate_trigger_asr(
        fp_reference_model, test_loader, final_trigger_norm, target, device
    )

    print('FP refined-trigger ASR:', float(fp_final_asr))
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
            'initialization': 'same_benign_ptq_target_ce',
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

    # Rewrite metadata so the saved artifact reflects the final one-PTQ method,
    # not the historical two-model implementation name used by the core helper.
    attack_info['method'] = 'final_single_ptq_ptq_aware_trigger_joint_legal_rounding'
    attack_info['method_flow'] = [
        'Benign PTQ (Q0)',
        'PTQ-aware Trigger Initialization on Q0',
        'Joint Trigger & Legal Rounding Optimization on the same Q0',
    ]
    attack_info['single_ptq_state'] = {
        'enabled': True,
        'ptq_build_count': 1,
        'fp_model_load_count': 1,
        'recalibration_after_trigger_init': False,
        'q0_reused_for_joint': True,
    }
    attack_info['trigger_initialization'] = init_meta
    attack_info['trigger_initialization_state_path'] = str(init_trigger_state_path)
    attack_info['trigger_initialization_png_path'] = str(init_trigger_png_path)
    attack_info['initial_reference'] = {
        'fp_ca': float(fp_ca),
        'benign_ptq_ca': float(benign_ptq_ca),
        'benign_ptq_neutral_trigger_asr': float(benign_ptq_neutral_asr),
        'same_q0_initialized_trigger_asr': float(q0_init_trigger_asr),
        'fp_initialized_trigger_asr': float(fp_init_trigger_asr),
    }
    attack_info['refined_trigger_on_fp'] = {'asr': float(fp_final_asr)}
    attack_info['trigger_state_path'] = str(final_trigger_state_path)
    attack_info['trigger_png_path'] = str(final_trigger_png_path)
    attack_info['checkpoint_path'] = str(checkpoint_path)
    attack_info['calibration_start_index'] = int(start_index)
    attack_info['experiment_identity'] = {
        'model': args.model,
        'dataset': args.dataset,
        'weight_bits': int(config.extra_prepare_dict.extra_qconfig_dict.w_qscheme.bit),
        'activation_bits': int(config.extra_prepare_dict.extra_qconfig_dict.a_qscheme.bit),
        'seed': int(config.process.seed),
    }
    attack_info['dataset_trigger_replacements'] = {
        'initial_train': int(train_replaced_init),
        'initial_test': int(test_replaced_init),
        'final_train': int(train_replaced_final),
        'final_test': int(test_replaced_final),
    }

    with open(diag_json, 'w') as handle:
        json.dump(attack_info, handle, indent=2)

    print('\n========== Final Summary ==========', flush=True)
    print('Method: Benign PTQ -> PTQ-aware Trigger Initialization -> Joint Trigger & Legal Rounding')
    print('PTQ build count: 1')
    print('PTQ recalibration after trigger init: False')
    print('Model:', args.model, 'Dataset:', args.dataset)
    print('W/A bits:', attack_info['experiment_identity']['weight_bits'], attack_info['experiment_identity']['activation_bits'])
    print('FP CA:', float(fp_ca))
    print('Q0 CA:', float(benign_ptq_ca))
    print('Q0 neutral-trigger ASR:', float(benign_ptq_neutral_asr))
    print('Q0 initialized-trigger ASR:', float(q0_init_trigger_asr))
    print('FP initialized-trigger ASR:', float(fp_init_trigger_asr))
    print('FP refined-trigger ASR:', float(fp_final_asr))
    print('Final CA:', float(attack_info['final_test']['ca']))
    print('Final ASR:', float(attack_info['final_test']['asr']))
    print('Final flip ratio:', float(attack_info['final_rounding_stats']['flip_ratio_valid']))
    print('Final flips:', int(attack_info['final_rounding_stats']['total_flipped']))
    print('JSON:', diag_json)
    print('Checkpoint:', checkpoint_path)
    print('Initial trigger:', init_trigger_state_path)
    print('Final trigger:', final_trigger_state_path)


if __name__ == '__main__':
    main()
