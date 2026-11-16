import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.no_target_direct_rounding_attack import (
    _to_device,
    _extract_logits,
    _forward_logits,
    _augment_clean_tensor,
    _apply_trigger,
    attacked_logits,
    collect_direct_rounding_records,
    direct_rounding_stats,
    evaluate_loader,
    extract_trigger_spec,
    finalize_direct_rounding,
    _effective_weight,
)
from mqbench.utils.state import enable_quantization


def reconstruction_backdoor_loss(
    clean_logits,
    fp_logits,
    trigger_logits,
    target,
    backdoor_weight=1.0,
):
    """Standard unlabeled PTQ reconstruction + targeted backdoor CE.

    Clean term follows the L2 output-reconstruction form used by optimized
    PTQ/AdaRound-style calibration:

        L_rec = mean_i sum_c |z_q(x_i)_c - z_fp(x_i)_c|^2

    Malicious term is ordinary target cross entropy:

        L_bd = CE(z_q(T(x)), t)

    Final objective:

        L = L_rec + L_bd

    No auxiliary clean loss, target margin/depth, projection, feature
    prototype, sparsity loss, or clean/attack lambda is used.
    """
    fp_logits = fp_logits.detach()

    rec_per_sample = (
        clean_logits - fp_logits
    ).abs().pow(2).sum(dim=1)
    rec_loss = rec_per_sample.mean()

    target_tensor = torch.full(
        (trigger_logits.shape[0],),
        int(target),
        device=trigger_logits.device,
        dtype=torch.long,
    )
    backdoor_loss = F.cross_entropy(
        trigger_logits,
        target_tensor,
    )

    total = rec_loss + float(backdoor_weight) * backdoor_loss

    trigger_prob = F.softmax(
        trigger_logits,
        dim=1,
    )[:, int(target)]

    return {
        'total': total,
        'reconstruction': rec_loss,
        'backdoor_ce': backdoor_loss,
        'weighted_backdoor_ce': float(backdoor_weight) * backdoor_loss,
        'trigger_target_probability': trigger_prob.mean(),
    }


@torch.no_grad()
def calibration_metrics(
    model,
    fp_reference_model,
    clean_data,
    bd_data,
    target,
    device,
    trigger,
):
    target = int(target)

    clean_total = 0
    fp_agree = 0
    rec_sum = 0.0

    paired_total = 0
    paired_hits = 0
    trigger_ce_sum = 0.0
    trigger_prob_all = []
    per_source = {}

    fp_reference_model.eval()
    model.eval()

    for batch in clean_data:
        batch = _to_device(batch, device)

        fp_logits = _extract_logits(
            fp_reference_model(batch)
            if not isinstance(batch, dict)
            else fp_reference_model(**batch)
        ).detach()

        attacked = attacked_logits(
            model,
            batch,
            device,
        )

        rec_per_sample = (
            attacked - fp_logits
        ).abs().pow(2).sum(dim=1)

        fp_pred = fp_logits.argmax(dim=1)
        attacked_pred = attacked.argmax(dim=1)

        fp_agree += int(
            (attacked_pred == fp_pred).sum().item()
        )
        clean_total += int(fp_pred.numel())
        rec_sum += float(
            rec_per_sample.sum().item()
        )

        trigger_batch = _apply_trigger(
            batch,
            trigger,
        )
        trigger_logits = attacked_logits(
            model,
            trigger_batch,
            device,
        )

        target_tensor = torch.full(
            (trigger_logits.shape[0],),
            target,
            device=trigger_logits.device,
            dtype=torch.long,
        )

        trigger_ce = F.cross_entropy(
            trigger_logits,
            target_tensor,
            reduction='none',
        )
        trigger_prob = F.softmax(
            trigger_logits,
            dim=1,
        )[:, target]
        trigger_pred = trigger_logits.argmax(dim=1)
        hit = trigger_pred == target

        paired_hits += int(hit.sum().item())
        paired_total += int(hit.numel())
        trigger_ce_sum += float(
            trigger_ce.sum().item()
        )
        trigger_prob_all.append(
            trigger_prob.detach().cpu()
        )

        for cls in fp_pred.unique().tolist():
            cls = int(cls)
            mask = fp_pred == cls
            count = int(mask.sum().item())
            item = per_source.setdefault(
                str(cls),
                {
                    'count': 0,
                    'hits': 0,
                    'target_probability_sum': 0.0,
                },
            )
            item['count'] += count
            item['hits'] += int(
                hit[mask].sum().item()
            )
            item['target_probability_sum'] += float(
                trigger_prob[mask].sum().item()
            )

    bd_total = 0
    bd_hits = 0
    for batch in bd_data:
        batch = _to_device(batch, device)
        logits = attacked_logits(
            model,
            batch,
            device,
        )
        pred = logits.argmax(dim=1)
        bd_hits += int(
            (pred == target).sum().item()
        )
        bd_total += int(pred.numel())

    trigger_prob_all = torch.cat(
        trigger_prob_all
    ).float()

    source_summary = {}
    for cls, item in per_source.items():
        count = int(item['count'])
        source_summary[cls] = {
            'count': count,
            'paired_asr': float(
                100.0 * item['hits'] / max(count, 1)
            ),
            'target_probability_mean': float(
                item['target_probability_sum'] / max(count, 1)
            ),
        }

    return {
        'clean_fp_agreement': float(
            100.0 * fp_agree / max(clean_total, 1)
        ),
        'clean_reconstruction_l2': float(
            rec_sum / max(clean_total, 1)
        ),
        'asr': float(
            100.0 * bd_hits / max(bd_total, 1)
        ),
        'paired_asr': float(
            100.0 * paired_hits / max(paired_total, 1)
        ),
        'trigger_ce': float(
            trigger_ce_sum / max(paired_total, 1)
        ),
        'trigger_target_probability_mean': float(
            trigger_prob_all.mean().item()
        ),
        'trigger_target_probability_p10': float(
            torch.quantile(trigger_prob_all, 0.10).item()
        ),
        'trigger_target_probability_min': float(
            trigger_prob_all.min().item()
        ),
        'clean_samples': int(clean_total),
        'paired_samples': int(paired_total),
        'bd_samples': int(bd_total),
        'per_source': source_summary,
    }


def reconstruction_ce_direct_rounding_attack(
    model,
    fp_reference_model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
    expected_trigger_state_path=None,
):
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
    backdoor_weight = (
        float(config.backdoor_weight)
        if hasattr(config, 'backdoor_weight')
        else 1.0
    )
    if backdoor_weight <= 0:
        raise ValueError('backdoor_weight must be positive')

    # Keep backward compatibility with the previous same_scope run:
    # if freeze_classifier is absent, default to True.
    freeze_classifier = (
        bool(config.freeze_classifier)
        if hasattr(config, 'freeze_classifier')
        else True
    )

    classifier_module = str(
        config.classifier_module
        if hasattr(config, 'classifier_module')
        else 'linear'
    )

    frozen_modules = (
        [classifier_module]
        if freeze_classifier
        else []
    )

    attack_records, frozen_records = (
        collect_direct_rounding_records(
            model,
            frozen_modules=frozen_modules,
        )
    )

    # The all-weight experiment is specifically meant to remove the
    # historical classifier restriction. Fail loudly if the classifier
    # is not actually part of the legal-rounding search space.
    if not freeze_classifier:
        attacked_names = {
            r['name']
            for r in attack_records
        }
        if classifier_module not in attacked_names:
            raise RuntimeError(
                'All-weight mode requested, but classifier module '
                '{} is not in the optimized quantized modules.'.format(
                    classifier_module
                )
            )

    rounding_parameters = []
    for rec in attack_records:
        rec['fq'].init_direct_rounding(
            _effective_weight(
                rec['module']
            ).detach()
        )
        rounding_parameters.extend(
            rec['fq'].structured_parameters()
        )

    step0_stats = direct_rounding_stats(
        attack_records
    )
    if step0_stats['total_flipped'] != 0:
        raise RuntimeError(
            'Step-0 direct rounding differs from benign PTQ.'
        )

    trigger_spec = extract_trigger_spec(
        trigger_loader
    )
    trigger_state_check = None
    if expected_trigger_state_path is not None:
        trigger_state = torch.load(
            expected_trigger_state_path,
            map_location='cpu',
        )
        expected_trigger = trigger_state['trigger'].detach().float().cpu()
        actual_trigger = trigger_spec['trigger_raw'].detach().float().cpu()
        if expected_trigger.shape != actual_trigger.shape:
            raise RuntimeError(
                'Reproduction trigger shape mismatch: expected {} got {}'.format(
                    tuple(expected_trigger.shape),
                    tuple(actual_trigger.shape),
                )
            )
        trigger_max_abs_diff = float(
            (expected_trigger - actual_trigger).abs().max().item()
        )
        if trigger_max_abs_diff != 0.0:
            raise RuntimeError(
                'Reproduction trigger mismatch: max_abs_diff={}'.format(
                    trigger_max_abs_diff
                )
            )
        if int(trigger_state['target']) != target:
            raise RuntimeError('Reproduction target mismatch.')
        trigger_state_check = {
            'path': str(expected_trigger_state_path),
            'exact_match': True,
            'max_abs_diff': trigger_max_abs_diff,
        }
        print('[RECON-CE] exact reproduction trigger: PASS')
    trigger = trigger_spec[
        'trigger_normalized'
    ].to(device)

    baseline = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        trigger=trigger,
    )

    print('[RECON-CE] step-0:', baseline)
    print(
        '[RECON-CE] optimized modules:',
        [r['name'] for r in attack_records],
    )
    print(
        '[RECON-CE] frozen modules:',
        [r['name'] for r in frozen_records],
    )
    print(
        '[RECON-CE] temporary rounding parameters:',
        step0_stats['rounding_parameters'],
    )

    optimizer = torch.optim.Adam(
        rounding_parameters,
        lr=float(config.lr),
        weight_decay=0.0,
    )

    steps = int(config.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(config.lr) * 0.1,
    )

    eval_interval = int(config.eval_interval)
    history = []

    for step in range(1, steps + 1):
        batch = _to_device(
            cali_data[
                int(np.random.randint(len(cali_data)))
            ],
            device,
        )

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(config.aug_pad),
            hflip_prob=float(config.hflip_prob),
        )

        with torch.no_grad():
            if isinstance(clean_view, dict):
                fp_logits = _extract_logits(
                    fp_reference_model(**clean_view)
                ).detach()
            else:
                fp_logits = _extract_logits(
                    fp_reference_model(clean_view)
                ).detach()

        trigger_view = _apply_trigger(
            clean_view,
            trigger,
        )

        optimizer.zero_grad()

        clean_logits = attacked_logits(
            model,
            clean_view,
            device,
        )
        trigger_logits = attacked_logits(
            model,
            trigger_view,
            device,
        )

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

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % eval_interval == 0
            or step == steps
        ):
            metrics = calibration_metrics(
                model=model,
                fp_reference_model=fp_reference_model,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
                trigger=trigger,
            )
            stats = direct_rounding_stats(
                attack_records
            )

            row = {
                'step': int(step),
                'train_total_loss': float(
                    loss.detach().item()
                ),
                'train_reconstruction_l2': float(
                    losses['reconstruction'].detach().item()
                ),
                'train_backdoor_ce': float(
                    losses['backdoor_ce'].detach().item()
                ),
                'train_weighted_backdoor_ce': float(
                    losses['weighted_backdoor_ce'].detach().item()
                ),
                'backdoor_weight': float(backdoor_weight),
                'train_trigger_target_probability': float(
                    losses['trigger_target_probability'].detach().item()
                ),
                **metrics,
                'flip_ratio_valid': float(
                    stats['flip_ratio_valid']
                ),
                'total_flipped': int(
                    stats['total_flipped']
                ),
                'rounding_parameters': int(
                    stats['rounding_parameters']
                ),
            }
            history.append(row)
            print('[RECON-CE]', row)

    final_calibration = calibration_metrics(
        model=model,
        fp_reference_model=fp_reference_model,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        trigger=trigger,
    )
    final_stats = direct_rounding_stats(
        attack_records
    )

    pre_test = {}
    if test_loader is not None and test_loader_bd is not None:
        pre_test = {
            'ca': evaluate_loader(
                model,
                test_loader,
                device,
            ),
            'asr': evaluate_loader(
                model,
                test_loader_bd,
                device,
            ),
        }

    print(
        '[RECON-CE] fixed-budget final state:',
        {
            'step': int(steps),
            'calibration': final_calibration,
            'test': pre_test,
        },
    )

    finalize_direct_rounding(
        model,
        attack_records,
    )

    post_test = {}
    if test_loader is not None and test_loader_bd is not None:
        post_test = {
            'ca': evaluate_loader(
                model,
                test_loader,
                device,
            ),
            'asr': evaluate_loader(
                model,
                test_loader_bd,
                device,
            ),
        }
        if (
            abs(post_test['ca'] - pre_test['ca']) > 1.0e-9
            or abs(post_test['asr'] - pre_test['asr']) > 1.0e-9
        ):
            raise RuntimeError(
                'Finalization changed behavior: pre={} post={}'.format(
                    pre_test,
                    post_test,
                )
            )

    leaked_parameters = [
        name
        for name, parameter in model.named_parameters()
        if (
            'rounding_offset' in name
            and parameter is not None
        )
    ]
    if leaked_parameters:
        raise RuntimeError(
            'Temporary rounding offsets remain: {}'.format(
                leaked_parameters[:20]
            )
        )

    result = {
        'method': 'reconstruction_ce_direct_legal_rounding',
        'variant_name': str(config.variant_name),
        'target': target,
        'threat_model': 'unlabeled calibration data',
        'objective': {
            'clean': 'L2 reconstruction between quantized and FP logits',
            'trigger': 'target cross entropy',
            'total': 'L_rec + alpha * L_bd',
            'backdoor_weight': float(backdoor_weight),
            'kl': False,
            'pseudo_labels': False,
            'target_geometry': False,
            'round_regularizer': False,
        },
        'freeze_classifier': freeze_classifier,
        'attacked_modules': [
            r['name'] for r in attack_records
        ],
        'frozen_modules': [
            r['name'] for r in frozen_records
        ],
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
        'final_test': post_test,
        'pre_finalize_test': pre_test,
        'temporary_state_removed': True,
        'trigger_size': int(
            trigger_spec['trigger_size']
        ),
        'reproduction_trigger_state': trigger_state_check,
        'implementation': {
            'steps': int(config.steps),
            'lr': float(config.lr),
            'grad_clip': float(config.grad_clip),
            'aug_pad': int(config.aug_pad),
            'hflip_prob': float(config.hflip_prob),
            'optimizer': 'Adam',
            'scheduler': 'CosineAnnealingLR',
            'checkpoint_selection': None,
            'test_during_training': False,
        },
    }

    if output_json is not None:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as handle:
            json.dump(result, handle, indent=2)
        print('[RECON-CE] saved:', output_json)

    return model, result
