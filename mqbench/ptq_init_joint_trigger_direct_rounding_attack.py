import json
import os

import numpy as np
import torch

from mqbench.joint_residual_trigger_direct_rounding_attack import (
    _apply_trigger,
    _extract_normalization,
    _lr_multiplier,
    _normalize_trigger,
    calibration_metrics,
    evaluate_trigger_asr,
    reconstruction_backdoor_loss,
)
from mqbench.no_target_direct_rounding_attack import (
    _augment_clean_tensor,
    _effective_weight,
    _extract_logits,
    _to_device,
    attacked_logits,
    collect_direct_rounding_records,
    direct_rounding_stats,
    evaluate_loader,
    extract_trigger_spec,
    finalize_direct_rounding,
)
from mqbench.utils.state import enable_quantization


def _trigger_stats(current_raw, initial_raw):
    with torch.no_grad():
        delta = current_raw - initial_raw
        return {
            'raw_min': float(current_raw.min().item()),
            'raw_max': float(current_raw.max().item()),
            'raw_mean': float(current_raw.mean().item()),
            'refine_delta_abs_mean': float(delta.abs().mean().item()),
            'refine_delta_abs_max': float(delta.abs().max().item()),
            'refine_delta_l2': float(delta.pow(2).sum().sqrt().item()),
        }


def ptq_init_joint_trigger_direct_rounding_attack(
    model,
    fp_reference_model,
    cali_data,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
):
    """Jointly refine a benign-PTQ-initialized trigger and Direct Legal Rounding.

    The trigger entering this function was initialized on a separate benign
    quantized probe model using ordinary target cross-entropy. The probe model
    is discarded before this final joint optimization stage.

    Variables optimized in this stage:
      1) legal floor/ceil rounding offsets;
      2) trigger pixels initialized from the PTQ-initialized trigger.

    Objective (unchanged from Direct Legal Rounding):
        L = L_rec + alpha * L_bd

    There is no additional trigger loss, flip regularizer, alternating inner
    loop, stochastic rounding family, or residual-to-BadNet constraint.
    """
    device = next(model.parameters()).device
    model.eval()
    fp_reference_model.eval()
    fp_reference_model.to(device)
    enable_quantization(model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in fp_reference_model.parameters():
        parameter.requires_grad_(False)

    target = int(config.bd_target)
    backdoor_weight = float(config.backdoor_weight)
    if backdoor_weight <= 0:
        raise ValueError('backdoor_weight must be positive.')

    freeze_classifier = (
        bool(config.freeze_classifier)
        if hasattr(config, 'freeze_classifier')
        else True
    )
    classifier_module = (
        str(config.classifier_module)
        if hasattr(config, 'classifier_module')
        else 'linear'
    )
    frozen_modules = [classifier_module] if freeze_classifier else []

    attack_records, frozen_records = collect_direct_rounding_records(
        model,
        frozen_modules=frozen_modules,
    )

    rounding_parameters = []
    for record in attack_records:
        record['fq'].init_direct_rounding(
            _effective_weight(record['module']).detach()
        )
        rounding_parameters.extend(record['fq'].structured_parameters())

    step0_stats = direct_rounding_stats(attack_records)
    if step0_stats['total_flipped'] != 0:
        raise RuntimeError('Step-0 direct rounding differs from benign PTQ.')

    trigger_spec = extract_trigger_spec(trigger_loader)
    initial_trigger_raw = trigger_spec['trigger_raw'].detach().float().to(device)
    if initial_trigger_raw.ndim != 3:
        raise RuntimeError('Expected CHW PTQ-initialized trigger.')

    # Important: this is the benign-PTQ target initialization itself, not a residual
    # around a white BadNet patch. Joint refinement starts from this solution.
    trigger = torch.nn.Parameter(initial_trigger_raw.clone())

    mean, std = _extract_normalization(trigger_loader)
    mean = mean.to(device)
    std = std.to(device)

    initial_trigger_norm = _normalize_trigger(
        trigger.clamp(0.0, 1.0),
        mean,
        std,
        device,
        dtype=next(model.parameters()).dtype,
    )

    baseline = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        target=target,
        device=device,
        trigger_normalized=initial_trigger_norm,
    )

    print('[PTQ-INIT-JOINT] step-0:', baseline)
    print('[PTQ-INIT-JOINT] optimized modules:', [r['name'] for r in attack_records])
    print('[PTQ-INIT-JOINT] frozen modules:', [r['name'] for r in frozen_records])
    print('[PTQ-INIT-JOINT] rounding parameters:', step0_stats['rounding_parameters'])
    print('[PTQ-INIT-JOINT] trigger parameters:', int(trigger.numel()))

    rounding_lr = float(config.lr)
    trigger_lr = float(config.trigger_lr)
    trigger_grad_clip = (
        float(config.trigger_grad_clip)
        if hasattr(config, 'trigger_grad_clip')
        else 5.0
    )

    optimizer = torch.optim.Adam(
        [
            {'params': rounding_parameters, 'lr': rounding_lr},
            {'params': [trigger], 'lr': trigger_lr},
        ],
        weight_decay=0.0,
    )

    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda s: _lr_multiplier(s, steps),
            lambda s: _lr_multiplier(s, steps),
        ],
    )

    eval_interval = int(config.eval_interval)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[int(np.random.randint(len(cali_data)))],
            device,
        )
        if isinstance(batch, dict):
            raise NotImplementedError('Current pilot supports image tensors only.')

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(config.aug_pad),
            hflip_prob=float(config.hflip_prob),
        )

        with torch.no_grad():
            fp_logits = _extract_logits(fp_reference_model(clean_view)).detach()

        current_trigger_raw = trigger.clamp(0.0, 1.0)
        current_trigger_norm = _normalize_trigger(
            current_trigger_raw,
            mean,
            std,
            device,
            dtype=clean_view.dtype,
        )
        trigger_view = _apply_trigger(clean_view, current_trigger_norm)

        optimizer.zero_grad(set_to_none=True)

        clean_logits = attacked_logits(model, clean_view, device)
        trigger_logits = attacked_logits(model, trigger_view, device)
        losses = reconstruction_backdoor_loss(
            clean_logits=clean_logits,
            fp_logits=fp_logits,
            trigger_logits=trigger_logits,
            target=target,
            backdoor_weight=backdoor_weight,
        )
        loss = losses['total']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            rounding_parameters,
            float(config.grad_clip),
        )
        if trigger_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([trigger], trigger_grad_clip)

        optimizer.step()
        with torch.no_grad():
            trigger.clamp_(0.0, 1.0)
        scheduler.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            current_trigger_raw = trigger.detach().clamp(0.0, 1.0)
            current_trigger_norm = _normalize_trigger(
                current_trigger_raw,
                mean,
                std,
                device,
                dtype=next(model.parameters()).dtype,
            )
            metrics = calibration_metrics(
                model=model,
                fp_reference_model=fp_reference_model,
                clean_data=cali_data,
                target=target,
                device=device,
                trigger_normalized=current_trigger_norm,
            )
            stats = direct_rounding_stats(attack_records)
            row = {
                'step': int(step),
                'train_total_loss': float(loss.detach().item()),
                'train_reconstruction_l2': float(losses['reconstruction'].detach().item()),
                'train_backdoor_ce': float(losses['backdoor_ce'].detach().item()),
                'train_weighted_backdoor_ce': float(losses['weighted_backdoor_ce'].detach().item()),
                'train_trigger_target_probability': float(
                    losses['trigger_target_probability'].detach().item()
                ),
                'rounding_lr': float(optimizer.param_groups[0]['lr']),
                'trigger_lr': float(optimizer.param_groups[1]['lr']),
                **metrics,
                **{f'trigger_{k}': v for k, v in _trigger_stats(
                    current_trigger_raw,
                    initial_trigger_raw,
                ).items()},
                'flip_ratio_valid': float(stats['flip_ratio_valid']),
                'total_flipped': int(stats['total_flipped']),
                'rounding_parameters': int(stats['rounding_parameters']),
            }
            history.append(row)
            print('[PTQ-INIT-JOINT]', row)

    final_trigger_raw = trigger.detach().clamp(0.0, 1.0)
    final_trigger_norm = _normalize_trigger(
        final_trigger_raw,
        mean,
        std,
        device,
        dtype=next(model.parameters()).dtype,
    ).detach()

    final_calibration = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        target=target,
        device=device,
        trigger_normalized=final_trigger_norm,
    )
    final_stats = direct_rounding_stats(attack_records)

    pre_test = {}
    if test_loader is not None:
        pre_test = {
            'ca': evaluate_loader(model, test_loader, device),
            'asr': evaluate_trigger_asr(
                model,
                test_loader,
                final_trigger_norm,
                target,
                device,
            ),
        }

    finalize_direct_rounding(model, attack_records)

    post_test = {}
    if test_loader is not None:
        post_test = {
            'ca': evaluate_loader(model, test_loader, device),
            'asr': evaluate_trigger_asr(
                model,
                test_loader,
                final_trigger_norm,
                target,
                device,
            ),
        }
        if (
            abs(post_test['ca'] - pre_test['ca']) > 1.0e-9
            or abs(post_test['asr'] - pre_test['asr']) > 1.0e-9
        ):
            raise RuntimeError(
                'Finalization changed behavior: pre={} post={}'.format(
                    pre_test, post_test
                )
            )

    leaked_parameters = [
        name
        for name, parameter in model.named_parameters()
        if 'rounding_offset' in name and parameter is not None
    ]
    if leaked_parameters:
        raise RuntimeError(
            'Temporary rounding offsets remain: {}'.format(leaked_parameters[:20])
        )

    final_trigger_stats = _trigger_stats(final_trigger_raw, initial_trigger_raw)
    result = {
        'method': 'ptq_initialized_joint_trigger_direct_legal_rounding',
        'variant_name': str(config.variant_name),
        'target': target,
        'objective': {
            'clean': 'L2 reconstruction between quantized and FP logits',
            'trigger': 'target CE using jointly refined PTQ-initialized trigger',
            'total': 'L_rec + alpha * L_bd',
            'backdoor_weight': float(backdoor_weight),
            'extra_trigger_loss': False,
            'round_regularizer': False,
            'flip_regularizer': False,
            'alternating_optimization': False,
        },
        'initialization': {
            'trigger': 'target-CE trigger initialized on a benign quantized probe model',
            'rounding': 'benign nearest rounding',
        },
        'variables': {
            'rounding': 'direct legal floor/ceil offsets',
            'trigger': 'raw trigger pixels initialized from PTQ-initialized trigger',
            'trigger_parameters': int(trigger.numel()),
        },
        'freeze_classifier': freeze_classifier,
        'attacked_modules': [r['name'] for r in attack_records],
        'frozen_modules': [r['name'] for r in frozen_records],
        'rounding_scope': (
            'all_quantized_weights'
            if not freeze_classifier
            else 'feature_extractor_only'
        ),
        'baseline_calibration': baseline,
        'history': history,
        'final_step': int(steps),
        'final_calibration': final_calibration,
        'final_rounding_stats': final_stats,
        'initial_trigger_stats': _trigger_stats(initial_trigger_raw, initial_trigger_raw),
        'final_trigger_stats': final_trigger_stats,
        'final_test': post_test,
        'pre_finalize_test': pre_test,
        'temporary_rounding_state_removed': True,
        'trigger_size': int(trigger_spec['trigger_size']),
        'implementation': {
            'steps': int(config.steps),
            'rounding_lr': float(config.lr),
            'trigger_lr': float(config.trigger_lr),
            'rounding_grad_clip': float(config.grad_clip),
            'trigger_grad_clip': float(trigger_grad_clip),
            'aug_pad': int(config.aug_pad),
            'hflip_prob': float(config.hflip_prob),
            'optimizer': 'Adam with two parameter groups',
            'scheduler': 'shared relative cosine multiplier from 1.0 to 0.1',
            'one_backward_per_step': True,
        },
    }

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as handle:
            json.dump(result, handle, indent=2)
        print('[PTQ-INIT-JOINT] saved:', output_json)

    return model, result, final_trigger_raw.detach().cpu()
