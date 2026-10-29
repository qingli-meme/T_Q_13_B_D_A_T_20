import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.utils.state import enable_quantization

# Reuse the exact, already-validated V6/V7 infrastructure.
from mqbench.structured_decision_core_attack import (
    _to_device,
    _effective_weight,
    _teacher_forward,
    extract_trigger_spec,
    _augment_clean_tensor,
    _apply_trigger,
    _extract_quantized_classifier,
    _build_decision_geometry,
    decision_code,
    _structured_stats,
    _evaluate_cv_loader,
    _capture_field_state,
    _restore_field_state,
    _finalize_structured_rounding,
)

from mqbench.structured_decision_translation_attack import (
    V7FeatureTapper,
    _attacked_forward,
    _collect_attack_records,
    _verify_classifier_geometry,
    build_class_decision_cores,
)


# ============================================================
# H3: shared target direction
# ============================================================

@torch.no_grad()
def build_shared_target_direction(
    class_cores,
    target,
):
    """
    Compress V7's source-dependent class translations

        d_y = c_t - c_y

    into the best rank-1 approximation

        d_y ~= alpha_y * v_t

    using SVD.

    v_t:
        one shared target direction.

    alpha_y:
        source-dependent scalar travel distance.

    This is the Eckart-Young optimal rank-1 approximation of the
    V7 translation matrix in squared Euclidean error.
    """

    target = int(target)

    num_classes = int(
        class_cores.shape[0]
    )

    source_classes = [
        cls
        for cls in range(
            num_classes
        )
        if cls != target
    ]

    target_core = class_cores[
        target
    ]

    translations = torch.stack(
        [
            (
                target_core
                - class_cores[
                    cls
                ]
            )
            for cls in source_classes
        ],
        dim=0,
    ).float()

    # D = U S V^T
    _, singular_values, vh = torch.linalg.svd(
        translations,
        full_matrices=False,
    )

    direction = (
        vh[0]
        .detach()
        .clone()
    )

    direction = (
        direction
        / direction.norm(
            p=2
        ).clamp_min(
            1e-12
        )
    )

    amplitudes_source = (
        translations
        @ direction
    )

    # Global sign ambiguity of the singular vector.
    if float(
        amplitudes_source.mean()
        .item()
    ) < 0:
        direction = (
            -direction
        )

        amplitudes_source = (
            -amplitudes_source
        )

    # H3 means one common DIRECTION, not merely one common line.
    # Therefore all source translations must project positively.
    min_amplitude = float(
        amplitudes_source.min()
        .item()
    )

    if min_amplitude <= 0:
        raise RuntimeError(
            "H3 shared-direction premise failed: "
            "at least one class requires the opposite direction. "
            "min alpha = {}".format(
                min_amplitude
            )
        )

    total_energy = float(
        singular_values.pow(2)
        .sum()
        .item()
    )

    first_energy = float(
        singular_values[0]
        .pow(2)
        .item()
    )

    explained_energy = (
        first_energy
        / max(
            total_energy,
            1e-12,
        )
    )

    rank1 = (
        amplitudes_source.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
    )

    residual = (
        translations
        - rank1
    )

    residual_ratio = float(
        residual.norm(
            p="fro"
        ).item()
        / max(
            translations.norm(
                p="fro"
            ).item(),
            1e-12,
        )
    )

    cosines = F.cosine_similarity(
        translations,
        rank1,
        dim=1,
        eps=1e-12,
    )

    # Class-indexed alpha table.
    # target class is unused and receives 0 for convenience.
    class_amplitudes = torch.zeros(
        num_classes,
        device=class_cores.device,
        dtype=class_cores.dtype,
    )

    for (
        idx,
        cls,
    ) in enumerate(
        source_classes
    ):
        class_amplitudes[
            cls
        ] = amplitudes_source[
            idx
        ].to(
            device=class_cores.device,
            dtype=class_cores.dtype,
        )

    per_source = {}

    for (
        idx,
        cls,
    ) in enumerate(
        source_classes
    ):
        full_norm = float(
            translations[
                idx
            ]
            .norm(
                p=2
            )
            .item()
        )

        approx_norm = float(
            rank1[
                idx
            ]
            .norm(
                p=2
            )
            .item()
        )

        per_source[
            str(cls)
        ] = {
            "full_translation_norm": (
                full_norm
            ),
            "shared_direction_amplitude": float(
                amplitudes_source[
                    idx
                ].item()
            ),
            "rank1_approx_norm": (
                approx_norm
            ),
            "translation_cosine_to_shared_direction": float(
                cosines[
                    idx
                ].item()
            ),
            "rank1_residual_norm": float(
                residual[
                    idx
                ]
                .norm(
                    p=2
                )
                .item()
            ),
        }

    print(
        "[V8-H3] shared-direction geometry:",
        {
            "explained_energy": (
                explained_energy
            ),
            "rank1_residual_ratio": (
                residual_ratio
            ),
            "min_amplitude": (
                min_amplitude
            ),
            "max_amplitude": float(
                amplitudes_source.max()
                .item()
            ),
            "per_source": (
                per_source
            ),
        },
    )

    return {
        "direction": (
            direction.detach()
        ),
        "class_amplitudes": (
            class_amplitudes.detach()
        ),
        "singular_values": (
            singular_values.detach()
            .cpu()
            .tolist()
        ),
        "explained_energy": (
            explained_energy
        ),
        "rank1_residual_ratio": (
            residual_ratio
        ),
        "per_source": (
            per_source
        ),
    }


# ============================================================
# Geometry sanity for the rank-1 H3 target
# ============================================================

@torch.no_grad()
def shared_direction_geometry_sanity(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    direction,
    class_amplitudes,
    target,
):
    """
    Without training, replace V7's full class translation

        c_t - c_y

    by its rank-1 H3 approximation

        alpha_y * v_t

    and measure how often the frozen classifier already predicts target.

    This is logged inside the real V8 run; it is not a separate probe.
    """

    target = int(
        target
    )

    direction = direction.to(
        device=device
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=device
        )
    )

    total = 0
    hits = 0

    per_source = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        per_source[
            cls
        ] = {
            "count": 0,
            "hits": 0,
        }

    for batch in cali_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        pseudo = logits.argmax(
            dim=1
        )

        code = decision_code(
            feature,
            geometry[
                "basis"
            ],
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_code = code[
            source_mask
        ]

        source_cls = pseudo[
            source_mask
        ]

        alpha = (
            class_amplitudes[
                source_cls
            ]
            .to(
                dtype=source_code.dtype
            )
        )

        translated = (
            source_code
            + alpha.unsqueeze(
                1
            )
            * direction.to(
                dtype=source_code.dtype
            ).unsqueeze(
                0
            )
        )

        translated_logits = F.linear(
            translated.float(),
            geometry[
                "decision_weight"
            ],
            geometry[
                "bias"
            ],
        )

        pred = translated_logits.argmax(
            dim=1
        )

        hit = (
            pred
            == target
        )

        total += int(
            hit.numel()
        )

        hits += int(
            hit.sum()
            .item()
        )

        for cls in per_source:
            cls_mask = (
                source_cls
                == cls
            )

            count = int(
                cls_mask.sum()
                .item()
            )

            if count == 0:
                continue

            per_source[
                cls
            ][
                "count"
            ] += count

            per_source[
                cls
            ][
                "hits"
            ] += int(
                hit[
                    cls_mask
                ]
                .sum()
                .item()
            )

    overall = (
        100.0
        * hits
        / max(
            total,
            1,
        )
    )

    per_source_rates = {}

    for (
        cls,
        item,
    ) in per_source.items():
        rate = (
            100.0
            * item[
                "hits"
            ]
            / max(
                item[
                    "count"
                ],
                1,
            )
        )

        per_source_rates[
            str(cls)
        ] = rate

    worst = min(
        per_source_rates.values()
    )

    result = {
        "overall": (
            overall
        ),
        "worst_source": (
            worst
        ),
        "per_source": (
            per_source_rates
        ),
    }

    print(
        "[V8-H3] rank1 geometry sanity:",
        result,
    )

    return result


# ============================================================
# ONE H3 paired-state objective
# ============================================================

def shared_direction_response_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    direction,
    class_amplitudes,
    basis,
):
    """
    Clean:
        u_phi(x)
            -> u_0(x)

    Trigger response:
        Delta_phi(x)
          = u_phi(T(x)) - u_phi(x)

        Delta_phi(x)
            -> alpha_y * v_t

    where every source shares the SAME v_t and only the scalar
    travel distance alpha_y varies.

    No CE.
    No margin.
    No target core endpoint.
    No 9-D source-specific translation vector.
    """

    clean_code = decision_code(
        clean_feature,
        basis,
    )

    benign_code = decision_code(
        clean_reference,
        basis,
    )

    trigger_code = decision_code(
        trigger_feature,
        basis,
    )

    clean_source_code = (
        clean_code[
            source_mask
        ]
    )

    actual_delta = (
        trigger_code
        - clean_source_code
    )

    direction = direction.to(
        device=actual_delta.device,
        dtype=actual_delta.dtype,
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=actual_delta.device,
            dtype=actual_delta.dtype,
        )
    )

    alpha = (
        class_amplitudes[
            source_labels
        ]
    )

    desired_delta = (
        alpha.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
    )

    clean_d2 = (
        (
            clean_code
            - benign_code
        )
        .pow(2)
        .sum(
            dim=1
        )
    )

    response_d2 = (
        (
            actual_delta
            - desired_delta
        )
        .pow(2)
        .sum(
            dim=1
        )
    )

    rank = max(
        int(
            clean_code.shape[
                1
            ]
        ),
        1,
    )

    # Same unified sample-level metric used in V7.
    all_d2 = torch.cat(
        [
            clean_d2,
            response_d2,
        ],
        dim=0,
    )

    total = (
        all_d2.mean()
        / float(
            rank
        )
    )

    # Diagnostics only.
    actual_alpha = (
        actual_delta
        @ direction
    )

    amplitude_error = (
        actual_alpha
        - alpha
    ).abs()

    actual_norm = (
        actual_delta.norm(
            p=2,
            dim=1,
        )
    )

    direction_cosine = (
        actual_alpha
        / actual_norm.clamp_min(
            1e-12
        )
    )

    orthogonal = (
        actual_delta
        - actual_alpha.unsqueeze(
            1
        )
        * direction.unsqueeze(
            0
        )
    )

    orthogonal_d2 = (
        orthogonal.pow(2)
        .sum(
            dim=1
        )
    )

    return {
        "total": (
            total
        ),
        "clean_code_distance": (
            clean_d2.mean()
            / float(
                rank
            )
        ),
        "shared_direction_response_distance": (
            response_d2.mean()
            / float(
                rank
            )
        ),
        "mean_direction_cosine": (
            direction_cosine.mean()
        ),
        "mean_amplitude_error": (
            amplitude_error.mean()
        ),
        "orthogonal_response_energy": (
            orthogonal_d2.mean()
            / float(
                rank
            )
        ),
    }


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def _calibration_metrics(
    model,
    attack_records,
    tapper,
    clean_data,
    bd_data,
    target,
    device,
    geometry,
    direction,
    class_amplitudes,
    trigger,
):
    target = int(
        target
    )

    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    direction = direction.to(
        device=device
    )

    class_amplitudes = (
        class_amplitudes.to(
            device=device
        )
    )

    rank = max(
        int(
            geometry[
                "rank"
            ]
        ),
        1,
    )

    clean_total = 0
    clean_agree = 0
    clean_distance_sum = 0.0

    response_total = 0
    response_distance_sum = 0.0
    paired_hits = 0

    cosine_sum = 0.0
    amplitude_error_sum = 0.0
    orthogonal_energy_sum = 0.0

    per_source = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        per_source[
            cls
        ] = {
            "count": 0,
            "hits": 0,
            "response_d2_sum": 0.0,
            "cosine_sum": 0.0,
            "amplitude_error_sum": 0.0,
        }

    for batch in clean_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            teacher_logits,
            teacher_feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )

        (
            attacked_logits,
            attacked_feature,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pseudo = teacher_logits.argmax(
            dim=1
        )

        clean_agree += int(
            (
                attacked_logits.argmax(
                    dim=1
                )
                == pseudo
            )
            .sum()
            .item()
        )

        teacher_code = decision_code(
            teacher_feature,
            basis.to(
                dtype=teacher_feature.dtype
            ),
        )

        attacked_code = decision_code(
            attacked_feature,
            basis.to(
                dtype=attacked_feature.dtype
            ),
        )

        clean_d2 = (
            (
                attacked_code
                - teacher_code
            )
            .pow(2)
            .sum(
                dim=1
            )
        )

        clean_distance_sum += float(
            clean_d2.sum()
            .item()
        )

        clean_total += int(
            pseudo.numel()
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_labels = pseudo[
            source_mask
        ]

        triggered = _apply_trigger(
            batch[
                source_mask
            ],
            trigger,
        )

        (
            trigger_logits,
            trigger_feature,
        ) = _attacked_forward(
            model,
            tapper,
            triggered,
            device,
        )

        trigger_code = decision_code(
            trigger_feature,
            basis.to(
                dtype=trigger_feature.dtype
            ),
        )

        clean_source_code = (
            attacked_code[
                source_mask
            ]
        )

        actual_delta = (
            trigger_code
            - clean_source_code
        )

        alpha = (
            class_amplitudes[
                source_labels
            ]
            .to(
                dtype=actual_delta.dtype
            )
        )

        direction_t = direction.to(
            dtype=actual_delta.dtype
        )

        desired_delta = (
            alpha.unsqueeze(
                1
            )
            * direction_t.unsqueeze(
                0
            )
        )

        response_d2 = (
            (
                actual_delta
                - desired_delta
            )
            .pow(2)
            .sum(
                dim=1
            )
        )

        actual_alpha = (
            actual_delta
            @ direction_t
        )

        actual_norm = (
            actual_delta.norm(
                p=2,
                dim=1,
            )
        )

        direction_cosine = (
            actual_alpha
            / actual_norm.clamp_min(
                1e-12
            )
        )

        amplitude_error = (
            actual_alpha
            - alpha
        ).abs()

        orthogonal = (
            actual_delta
            - actual_alpha.unsqueeze(
                1
            )
            * direction_t.unsqueeze(
                0
            )
        )

        orthogonal_d2 = (
            orthogonal.pow(2)
            .sum(
                dim=1
            )
        )

        hit = (
            trigger_logits.argmax(
                dim=1
            )
            == target
        )

        n = int(
            hit.numel()
        )

        response_total += n

        paired_hits += int(
            hit.sum()
            .item()
        )

        response_distance_sum += float(
            response_d2.sum()
            .item()
        )

        cosine_sum += float(
            direction_cosine.sum()
            .item()
        )

        amplitude_error_sum += float(
            amplitude_error.sum()
            .item()
        )

        orthogonal_energy_sum += float(
            orthogonal_d2.sum()
            .item()
        )

        for cls in per_source:
            cls_mask = (
                source_labels
                == cls
            )

            count = int(
                cls_mask.sum()
                .item()
            )

            if count == 0:
                continue

            per_source[
                cls
            ][
                "count"
            ] += count

            per_source[
                cls
            ][
                "hits"
            ] += int(
                hit[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "response_d2_sum"
            ] += float(
                response_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "cosine_sum"
            ] += float(
                direction_cosine[
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "amplitude_error_sum"
            ] += float(
                amplitude_error[
                    cls_mask
                ]
                .sum()
                .item()
            )

    # Standard repository calibration ASR.
    bd_total = 0
    bd_hits = 0

    for batch in bd_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            _,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pred = logits.argmax(
            dim=1
        )

        bd_hits += int(
            (
                pred
                == target
            )
            .sum()
            .item()
        )

        bd_total += int(
            pred.numel()
        )

    per_source_result = {}

    for (
        cls,
        item,
    ) in per_source.items():
        count = max(
            item[
                "count"
            ],
            1,
        )

        per_source_result[
            str(cls)
        ] = {
            "count": int(
                item[
                    "count"
                ]
            ),
            "paired_asr": (
                100.0
                * item[
                    "hits"
                ]
                / count
            ),
            "shared_direction_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "mean_direction_cosine": (
                item[
                    "cosine_sum"
                ]
                / count
            ),
            "mean_amplitude_error": (
                item[
                    "amplitude_error_sum"
                ]
                / count
            ),
        }

    return {
        "clean_teacher_agreement": (
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_code_distance": (
            clean_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(
                rank
            )
        ),
        "asr": (
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "paired_asr": (
            100.0
            * paired_hits
            / max(
                response_total,
                1,
            )
        ),
        "shared_direction_response_distance": (
            response_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "mean_direction_cosine": (
            cosine_sum
            / max(
                response_total,
                1,
            )
        ),
        "mean_amplitude_error": (
            amplitude_error_sum
            / max(
                response_total,
                1,
            )
        ),
        "orthogonal_response_energy": (
            orthogonal_energy_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "clean_samples": int(
            clean_total
        ),
        "response_samples": int(
            response_total
        ),
        "bd_samples": int(
            bd_total
        ),
        "per_source": (
            per_source_result
        ),
    }


# ============================================================
# Main V8/H3 attack
# ============================================================

def structured_shared_direction_attack(
    model,
    cali_data,
    cali_data_bd,
    trigger_loader,
    config,
    output_json=None,
    test_loader=None,
    test_loader_bd=None,
):
    """
    V8 / H3:
        Structured Rounding Field
        + Shared Target Direction

    V7:
        Delta_phi(x) -> c_t - c_y

    V8/H3:
        c_t - c_y ~= alpha_y * v_t
        Delta_phi(x) -> alpha_y * v_t

    One shared direction v_t.
    Source identity only changes scalar travel distance alpha_y.
    """

    device = next(
        model.parameters()
    ).device

    model.eval()

    enable_quantization(
        model
    )

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    tapper = V7FeatureTapper(
        model,
        str(
            config.semantic_module
        ),
    )

    (
        attack_records,
        frozen_records,
    ) = _collect_attack_records(
        model,
        list(
            config.frozen_field_modules
        ),
    )

    # --------------------------------------------------------
    # EXACT same structured-field capacity as V7.
    # --------------------------------------------------------

    field_params = []

    for record in attack_records:
        fq = record[
            "fq"
        ]

        fq.init_structured_attack(
            _effective_weight(
                record[
                    "module"
                ]
            ).detach(),
            int(
                config.field_rank
            ),
            float(
                config.delta_bound
            ),
            float(
                config.round_temperature
            ),
            float(
                config.field_init_std
            ),
        )

        field_params.extend(
            [
                fq.field_u,
                fq.field_v,
            ]
        )

    step0_stats = _structured_stats(
        attack_records
    )

    if (
        step0_stats[
            "total_flipped"
        ]
        != 0
    ):
        raise RuntimeError(
            "Step-0 structured field changed rounding."
        )

    for record in frozen_records:
        if getattr(
            record[
                "fq"
            ],
            "structured_attack",
            False,
        ):
            raise RuntimeError(
                "Frozen classifier unexpectedly has active structured field."
            )

    # --------------------------------------------------------
    # Trigger.
    # --------------------------------------------------------

    trigger_spec = extract_trigger_spec(
        trigger_loader
    )

    trigger = (
        trigger_spec[
            "trigger_normalized"
        ]
        .to(
            device
        )
    )

    print(
        "[V8-H3] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Frozen classifier geometry.
    # --------------------------------------------------------

    (
        _,
        classifier_weight,
        classifier_bias,
    ) = _extract_quantized_classifier(
        model,
        str(
            config.classifier_module
        ),
    )

    geometry = _build_decision_geometry(
        classifier_weight,
        classifier_bias,
        svd_tol=float(
            config.decision_svd_tol
        ),
    )

    print(
        "[V8-H3] classifier geometry:",
        {
            "feature_dim": (
                geometry[
                    "feature_dim"
                ]
            ),
            "num_classes": (
                geometry[
                    "num_classes"
                ]
            ),
            "decision_rank": (
                geometry[
                    "rank"
                ]
            ),
            "null_dim": (
                geometry[
                    "null_dim"
                ]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
        },
    )

    if (
        geometry[
            "rank"
        ]
        > (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1."
        )

    geometry_check = _verify_classifier_geometry(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        batch=_to_device(
            cali_data[
                0
            ],
            device,
        ),
        device=device,
        geometry=geometry,
        tolerance=float(
            config.geometry_logit_tolerance
        ),
    )

    # --------------------------------------------------------
    # Same benign class cores as V7.
    # They are used ONLY to derive the rank-1 target structure.
    # --------------------------------------------------------

    class_info = build_class_decision_cores(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        min_class_count=int(
            config.min_class_count
        ),
    )

    class_cores = class_info[
        "cores"
    ]

    # --------------------------------------------------------
    # H3 compression:
    # nine V7 vectors -> one direction + nine scalar amplitudes.
    # --------------------------------------------------------

    shared_info = build_shared_target_direction(
        class_cores=class_cores,
        target=target,
    )

    direction = shared_info[
        "direction"
    ]

    class_amplitudes = shared_info[
        "class_amplitudes"
    ]

    rank1_geometry = shared_direction_geometry_sanity(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        direction=direction,
        class_amplitudes=class_amplitudes,
        target=target,
    )

    # --------------------------------------------------------
    # Step-0 metrics.
    # --------------------------------------------------------

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        direction=direction,
        class_amplitudes=class_amplitudes,
        trigger=trigger,
    )

    print(
        "[V8-H3] step-0 calibration:",
        baseline,
    )

    print(
        "[V8-H3] step-0 structured stats:",
        step0_stats,
    )

    if (
        baseline[
            "clean_teacher_agreement"
        ]
        < 99.999
    ):
        raise RuntimeError(
            "Step-0 attacked model != benign PTQ."
        )

    if (
        baseline[
            "clean_code_distance"
        ]
        > 1e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-code distance is not zero."
        )

    # --------------------------------------------------------
    # EXACT same optimizer/schedule as V7.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        field_params,
        lr=float(
            config.lr
        ),
        weight_decay=float(
            config.weight_decay
        ),
    )

    steps = int(
        config.steps
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=(
                float(
                    config.lr
                )
                * 0.1
            ),
        )
    )

    eval_interval = int(
        config.eval_interval
    )

    test_interval = int(
        config.diagnostic_test_interval
    )

    # --------------------------------------------------------
    # Same calibration-only checkpoint selection as V7.
    # --------------------------------------------------------

    best_state = _capture_field_state(
        attack_records
    )

    clean0 = baseline[
        "clean_teacher_agreement"
    ]

    asr0 = baseline[
        "asr"
    ]

    best_score = (
        2.0
        * clean0
        * asr0
        / max(
            clean0
            + asr0,
            1e-12,
        )
    )

    best_step = 0
    history = []

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    for step in range(
        1,
        steps
        + 1,
    ):
        batch = _to_device(
            cali_data[
                int(
                    np.random.randint(
                        len(
                            cali_data
                        )
                    )
                )
            ],
            device,
        )

        clean_view = _augment_clean_tensor(
            batch,
            pad=int(
                config.aug_pad
            ),
            hflip_prob=float(
                config.hflip_prob
            ),
        )

        (
            teacher_logits,
            teacher_feature,
        ) = _teacher_forward(
            model,
            attack_records,
            tapper,
            clean_view,
            device,
        )

        pseudo = teacher_logits.argmax(
            dim=1
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        source_labels = pseudo[
            source_mask
        ]

        triggered_view = _apply_trigger(
            clean_view[
                source_mask
            ],
            trigger,
        )

        optimizer.zero_grad()

        (
            _,
            clean_feature,
        ) = _attacked_forward(
            model,
            tapper,
            clean_view,
            device,
        )

        (
            _,
            trigger_feature,
        ) = _attacked_forward(
            model,
            tapper,
            triggered_view,
            device,
        )

        losses = shared_direction_response_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            direction=direction,
            class_amplitudes=class_amplitudes,
            basis=geometry[
                "basis"
            ],
        )

        loss = losses[
            "total"
        ]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            field_params,
            float(
                config.grad_clip
            ),
        )

        optimizer.step()
        scheduler.step()

        # ----------------------------------------------------
        # Periodic metrics.
        # ----------------------------------------------------

        if (
            step == 1
            or step
            % eval_interval
            == 0
            or step
            == steps
        ):
            metrics = _calibration_metrics(
                model=model,
                attack_records=attack_records,
                tapper=tapper,
                clean_data=cali_data,
                bd_data=cali_data_bd,
                target=target,
                device=device,
                geometry=geometry,
                direction=direction,
                class_amplitudes=class_amplitudes,
                trigger=trigger,
            )

            stats = _structured_stats(
                attack_records
            )

            clean_agreement = metrics[
                "clean_teacher_agreement"
            ]

            asr = metrics[
                "asr"
            ]

            score = (
                2.0
                * clean_agreement
                * asr
                / max(
                    clean_agreement
                    + asr,
                    1e-12,
                )
            )

            row = {
                "step": int(
                    step
                ),
                "train_total_loss": float(
                    loss.detach()
                    .item()
                ),
                "train_clean_code_distance": float(
                    losses[
                        "clean_code_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_shared_direction_response_distance": float(
                    losses[
                        "shared_direction_response_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_mean_direction_cosine": float(
                    losses[
                        "mean_direction_cosine"
                    ]
                    .detach()
                    .item()
                ),
                "train_mean_amplitude_error": float(
                    losses[
                        "mean_amplitude_error"
                    ]
                    .detach()
                    .item()
                ),
                "train_orthogonal_response_energy": float(
                    losses[
                        "orthogonal_response_energy"
                    ]
                    .detach()
                    .item()
                ),
                "score": float(
                    score
                ),
                **metrics,
                "flip_ratio_valid": float(
                    stats[
                        "flip_ratio_valid"
                    ]
                ),
                "total_flipped": int(
                    stats[
                        "total_flipped"
                    ]
                ),
                "field_parameters": int(
                    stats[
                        "field_parameters"
                    ]
                ),
            }

            if (
                bool(
                    config.diagnostic_test_eval
                )
                and test_loader
                is not None
                and test_loader_bd
                is not None
                and (
                    step
                    % test_interval
                    == 0
                    or step
                    == steps
                )
            ):
                row[
                    "diagnostic_test_ca"
                ] = _evaluate_cv_loader(
                    model,
                    test_loader,
                    device,
                )

                row[
                    "diagnostic_test_asr"
                ] = _evaluate_cv_loader(
                    model,
                    test_loader_bd,
                    device,
                )

            history.append(
                row
            )

            print(
                "[V8-H3]",
                row,
            )

            if score > best_score:
                best_score = float(
                    score
                )

                best_step = int(
                    step
                )

                best_state = _capture_field_state(
                    attack_records
                )

                print(
                    "[V8-H3] new best:",
                    {
                        "step": (
                            best_step
                        ),
                        "score": (
                            best_score
                        ),
                        "clean_agreement": (
                            clean_agreement
                        ),
                        "asr": (
                            asr
                        ),
                        "clean_code_distance": (
                            metrics[
                                "clean_code_distance"
                            ]
                        ),
                        "response_distance": (
                            metrics[
                                "shared_direction_response_distance"
                            ]
                        ),
                        "direction_cosine": (
                            metrics[
                                "mean_direction_cosine"
                            ]
                        ),
                        "amplitude_error": (
                            metrics[
                                "mean_amplitude_error"
                            ]
                        ),
                    },
                )

    # --------------------------------------------------------
    # Restore best calibration-selected field.
    # --------------------------------------------------------

    _restore_field_state(
        attack_records,
        best_state,
    )

    restored = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        direction=direction,
        class_amplitudes=class_amplitudes,
        trigger=trigger,
    )

    stats = _structured_stats(
        attack_records
    )

    test = {}

    if (
        test_loader is not None
        and test_loader_bd is not None
    ):
        test = {
            "ca": _evaluate_cv_loader(
                model,
                test_loader,
                device,
            ),
            "asr": _evaluate_cv_loader(
                model,
                test_loader_bd,
                device,
            ),
        }

    print(
        "[V8-H3] restored best step:",
        best_step,
    )

    print(
        "[V8-H3] restored calibration:",
        restored,
    )

    print(
        "[V8-H3] restored test:",
        test,
    )

    print(
        "[V8-H3] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "shared_target_direction"
        ),
        "semantic_module": str(
            config.semantic_module
        ),
        "classifier_module": str(
            config.classifier_module
        ),
        "frozen_field_modules": [
            record[
                "name"
            ]
            for record in frozen_records
        ],
        "classifier_geometry": {
            "feature_dim": int(
                geometry[
                    "feature_dim"
                ]
            ),
            "num_classes": int(
                geometry[
                    "num_classes"
                ]
            ),
            "decision_rank": int(
                geometry[
                    "rank"
                ]
            ),
            "null_dim": int(
                geometry[
                    "null_dim"
                ]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
            "geometry_max_logit_error": (
                geometry_check[
                    "max_logit_error"
                ]
            ),
            "decision_code_argmax_agreement": (
                geometry_check[
                    "argmax_agreement"
                ]
            ),
        },
        "class_cores": {
            "counts": (
                class_info[
                    "counts"
                ]
            ),
            "within_class_mse": (
                class_info[
                    "within_class_mse"
                ]
            ),
            "core_self_agreement": (
                class_info[
                    "core_self_agreement"
                ]
            ),
        },
        "shared_direction": {
            "singular_values": (
                shared_info[
                    "singular_values"
                ]
            ),
            "explained_energy": float(
                shared_info[
                    "explained_energy"
                ]
            ),
            "rank1_residual_ratio": float(
                shared_info[
                    "rank1_residual_ratio"
                ]
            ),
            "per_source": (
                shared_info[
                    "per_source"
                ]
            ),
            "rank1_geometry_sanity": (
                rank1_geometry
            ),
        },
        "baseline_calibration": (
            baseline
        ),
        "history": (
            history
        ),
        "best_step": int(
            best_step
        ),
        "best_score": float(
            best_score
        ),
        "best_calibration": (
            restored
        ),
        "best_test": (
            test
        ),
        "structured_stats": (
            stats
        ),
        "trigger_size": int(
            trigger_spec[
                "trigger_size"
            ]
        ),
        "config": {
            "bd_target": int(
                config.bd_target
            ),
            "field_rank": int(
                config.field_rank
            ),
            "delta_bound": float(
                config.delta_bound
            ),
            "round_temperature": float(
                config.round_temperature
            ),
            "decision_svd_tol": float(
                config.decision_svd_tol
            ),
            "steps": int(
                config.steps
            ),
            "lr": float(
                config.lr
            ),
        },
    }

    # --------------------------------------------------------
    # Materialize malicious feature-extractor rounding only.
    # --------------------------------------------------------

    _finalize_structured_rounding(
        model,
        attack_records,
    )

    tapper.close()

    if output_json is not None:
        output_dir = os.path.dirname(
            output_json
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        with open(
            output_json,
            "w",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
            )

        print(
            "[V8-H3] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

