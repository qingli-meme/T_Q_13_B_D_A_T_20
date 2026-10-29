import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.utils.state import enable_quantization

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

from mqbench.structured_target_compatible_attack import (
    build_target_margin_geometry,
    target_margin_coordinates,
    target_compatible_endpoint,
)


# ============================================================
# Numerical projection solver
# ============================================================

_DYKSTRA_CYCLES = 100


@torch.no_grad()
def project_to_deep_target_set(
    teacher_code,
    margin_geometry,
    depth_floor,
    target_core,
):
    """
    Euclidean projection in the EXACT classifier decision space.

    Deep target set:
        C_t(gamma)
          = {
                u :
                q_j(u) >= gamma,
                for every j != t
            }

    where
        q(u) = A_t u + beta_t

    are the normalized target-vs-competitor classifier margins.

    For each benign teacher code u0:

        u* = argmin_u 1/2 ||u-u0||_2^2
             s.t. A_t u + beta_t >= gamma.

    The projection is computed with Dykstra's algorithm over the
    C-1 half-spaces.

    Important:
      - teacher_code lives in the 9-D decision space;
      - the 503-D classifier-null space never enters this solver;
      - gamma is NOT tuned. It is the target-core worst margin;
      - target_core is a feasible point, therefore the projected
        movement can never exceed movement to the V6 target core.

    Numerical setting:
        100 Dykstra cycles.

    This is a solver-accuracy constant, not an attack hyperparameter.
    It is cross-checked against SLSQP in the generated experiment doc.
    """

    code = (
        teacher_code.detach()
        .double()
    )

    matrix = (
        margin_geometry[
            "matrix"
        ]
        .to(
            device=code.device,
            dtype=torch.double,
        )
    )

    offset = (
        margin_geometry[
            "offset"
        ]
        .to(
            device=code.device,
            dtype=torch.double,
        )
    )

    core = (
        target_core.detach()
        .to(
            device=code.device,
            dtype=torch.double,
        )
    )

    if core.ndim == 1:
        core = core.unsqueeze(
            0
        )

    gamma = float(
        depth_floor
    )

    if code.ndim != 2:
        raise ValueError(
            "Deep-target projection expects [B, r] decision codes."
        )

    if matrix.ndim != 2:
        raise ValueError(
            "Target-margin matrix must be 2-D."
        )

    if (
        matrix.shape[
            1
        ]
        != code.shape[
            1
        ]
    ):
        raise ValueError(
            "Decision-code dimension mismatch."
        )

    # Verify that the V6 target core is a feasible candidate.
    core_margin = (
        core
        @ matrix.transpose(
            0,
            1
        )
        + offset.unsqueeze(
            0
        )
    )

    if (
        core_margin.min()
        .item()
        + 1.0e-10
        < gamma
    ):
        raise RuntimeError(
            (
                "Target core is not feasible for deep target set: "
                "core_depth={}, gamma={}"
            ).format(
                float(
                    core_margin.min()
                    .item()
                ),
                gamma,
            )
        )

    batch_size = int(
        code.shape[
            0
        ]
    )

    num_constraints = int(
        matrix.shape[
            0
        ]
    )

    rank = int(
        code.shape[
            1
        ]
    )

    y = code.clone()

    corrections = torch.zeros(
        (
            batch_size,
            num_constraints,
            rank,
        ),
        device=code.device,
        dtype=torch.double,
    )

    norm_sq = (
        matrix.pow(
            2
        )
        .sum(
            dim=1
        )
        .clamp_min(
            1.0e-18
        )
    )

    for _ in range(
        _DYKSTRA_CYCLES
    ):
        for j in range(
            num_constraints
        ):
            z = (
                y
                + corrections[
                    :,
                    j,
                    :
                ]
            )

            normal = matrix[
                j
            ]

            score = (
                z
                @ normal
                + offset[
                    j
                ]
                - gamma
            )

            step = (
                (-score)
                .clamp_min(
                    0.0
                )
                / norm_sq[
                    j
                ]
            )

            projected = (
                z
                + step.unsqueeze(
                    1
                )
                * normal.unsqueeze(
                    0
                )
            )

            corrections[
                :,
                j,
                :
            ] = (
                z
                - projected
            )

            y = projected

    endpoint = (
        y
    )

    endpoint_margin = (
        endpoint
        @ matrix.transpose(
            0,
            1
        )
        + offset.unsqueeze(
            0
        )
    )

    endpoint_depth = (
        endpoint_margin.min(
            dim=1
        ).values
    )

    violation = (
        gamma
        - endpoint_margin
    ).clamp_min(
        0.0
    )

    max_violation = float(
        violation.max()
        .item()
    )

    if max_violation > 5.0e-7:
        raise RuntimeError(
            (
                "Deep-target projection did not converge: "
                "max_violation={}"
            ).format(
                max_violation
            )
        )

    movement_d2 = (
        (
            endpoint
            - code
        )
        .pow(
            2
        )
        .sum(
            dim=1
        )
    )

    core_d2 = (
        (
            core
            - code
        )
        .pow(
            2
        )
        .sum(
            dim=1
        )
    )

    # Projection theorem:
    # because target_core is feasible, the nearest feasible point
    # cannot be farther away than target_core.
    movement_tol = (
        5.0e-6
        * torch.maximum(
            torch.ones_like(
                core_d2
            ),
            core_d2,
        )
    )

    if (
        movement_d2
        > core_d2
        + movement_tol
    ).any():
        worst = float(
            (
                movement_d2
                - core_d2
            )
            .max()
            .item()
        )

        raise RuntimeError(
            (
                "Projection movement exceeded V6 target-core movement: "
                "worst excess={}"
            ).format(
                worst
            )
        )

    active = (
        (
            endpoint_margin
            - gamma
        )
        .abs()
        <= 1.0e-5
    )

    return {
        "endpoint": (
            endpoint.to(
                dtype=teacher_code.dtype
            )
            .detach()
        ),
        "endpoint_margin": (
            endpoint_margin.to(
                dtype=teacher_code.dtype
            )
            .detach()
        ),
        "endpoint_depth": (
            endpoint_depth.to(
                dtype=teacher_code.dtype
            )
            .detach()
        ),
        "movement_d2": (
            movement_d2.to(
                dtype=teacher_code.dtype
            )
            .detach()
        ),
        "v6_core_d2": (
            core_d2.to(
                dtype=teacher_code.dtype
            )
            .detach()
        ),
        "active_constraints": (
            active.sum(
                dim=1
            )
            .detach()
        ),
        "max_violation": float(
            max_violation
        ),
    }


# ============================================================
# Static geometry summary
# ============================================================

@torch.no_grad()
def build_deep_target_projection_summary(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    class_cores,
    margin_geometry,
    target,
    depth_floor,
):
    """
    Measure the geometry BEFORE training.

    Key guarantees checked here:
      1) every projected endpoint has target depth >= gamma;
      2) every projected movement <= V6 movement, sample-wise;
      3) compare V12 movement against V7 and V11 endpoints.
    """

    target = int(
        target
    )

    cores = class_cores.to(
        device=device
    )

    target_core = (
        cores[
            target
        ]
        .unsqueeze(
            0
        )
    )

    all_projected_d2 = []
    all_v6_d2 = []
    all_v7_d2 = []
    all_v11_d2 = []
    all_depth = []
    all_active = []
    all_labels = []

    ideal_total = 0
    ideal_hits = 0

    per_source_accum = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        per_source_accum[
            cls
        ] = {
            "count": 0,
            "v12_d2": 0.0,
            "v6_d2": 0.0,
            "v7_d2": 0.0,
            "v11_d2": 0.0,
            "depth_sum": 0.0,
            "depth_min": float(
                "inf"
            ),
            "active_sum": 0.0,
            "strictly_better_than_v6": 0,
        }

    max_projection_violation = 0.0

    for batch in cali_data:
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

        pseudo = teacher_logits.argmax(
            dim=1
        )

        source_mask = (
            pseudo
            != target
        )

        if not source_mask.any():
            continue

        labels = (
            pseudo[
                source_mask
            ]
        )

        code = decision_code(
            teacher_feature[
                source_mask
            ],
            geometry[
                "basis"
            ],
        )

        projection = project_to_deep_target_set(
            teacher_code=code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
        )

        endpoint = projection[
            "endpoint"
        ]

        endpoint_depth = projection[
            "endpoint_depth"
        ]

        movement_d2 = projection[
            "movement_d2"
        ]

        v6_d2 = projection[
            "v6_core_d2"
        ]

        max_projection_violation = max(
            max_projection_violation,
            float(
                projection[
                    "max_violation"
                ]
            ),
        )

        # V7 residual-preserving class translation.
        source_core = (
            cores[
                labels
            ]
        )

        v7_endpoint = (
            code
            + target_core
            - source_core
        )

        v7_d2 = (
            (
                v7_endpoint
                - code
            )
            .pow(
                2
            )
            .sum(
                dim=1
            )
        )

        # V11 target-compatible residual endpoint.
        v11 = target_compatible_endpoint(
            teacher_code=code,
            source_labels=labels,
            class_cores=class_cores,
            margin_geometry=margin_geometry,
            target=target,
        )

        v11_d2 = (
            (
                v11[
                    "endpoint"
                ]
                - code
            )
            .pow(
                2
            )
            .sum(
                dim=1
            )
        )

        hit = (
            endpoint_depth
            >= float(
                depth_floor
            )
            - 5.0e-5
        )

        ideal_total += int(
            hit.numel()
        )

        ideal_hits += int(
            hit.sum()
            .item()
        )

        all_projected_d2.append(
            movement_d2.cpu()
        )

        all_v6_d2.append(
            v6_d2.cpu()
        )

        all_v7_d2.append(
            v7_d2.cpu()
        )

        all_v11_d2.append(
            v11_d2.cpu()
        )

        all_depth.append(
            endpoint_depth.cpu()
        )

        all_active.append(
            projection[
                "active_constraints"
            ]
            .cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        for cls in per_source_accum:
            cls_mask = (
                labels
                == cls
            )

            count = int(
                cls_mask.sum()
                .item()
            )

            if count == 0:
                continue

            item = per_source_accum[
                cls
            ]

            item[
                "count"
            ] += count

            item[
                "v12_d2"
            ] += float(
                movement_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "v6_d2"
            ] += float(
                v6_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "v7_d2"
            ] += float(
                v7_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "v11_d2"
            ] += float(
                v11_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            cls_depth = (
                endpoint_depth[
                    cls_mask
                ]
            )

            item[
                "depth_sum"
            ] += float(
                cls_depth.sum()
                .item()
            )

            item[
                "depth_min"
            ] = min(
                item[
                    "depth_min"
                ],
                float(
                    cls_depth.min()
                    .item()
                ),
            )

            item[
                "active_sum"
            ] += float(
                projection[
                    "active_constraints"
                ][
                    cls_mask
                ]
                .float()
                .sum()
                .item()
            )

            item[
                "strictly_better_than_v6"
            ] += int(
                (
                    movement_d2[
                        cls_mask
                    ]
                    < v6_d2[
                        cls_mask
                    ]
                    - 1.0e-6
                )
                .sum()
                .item()
            )

    v12_d2 = torch.cat(
        all_projected_d2,
        dim=0,
    ).float()

    v6_d2 = torch.cat(
        all_v6_d2,
        dim=0,
    ).float()

    v7_d2 = torch.cat(
        all_v7_d2,
        dim=0,
    ).float()

    v11_d2 = torch.cat(
        all_v11_d2,
        dim=0,
    ).float()

    depth = torch.cat(
        all_depth,
        dim=0,
    ).float()

    active = torch.cat(
        all_active,
        dim=0,
    ).float()

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    if (
        v12_d2
        > v6_d2
        + (
            5.0e-5
            * torch.maximum(
                torch.ones_like(
                    v6_d2
                ),
                v6_d2,
            )
        )
    ).any():
        raise RuntimeError(
            "Static V12 movement guarantee vs V6 failed."
        )

    per_source = {}

    for (
        cls,
        item,
    ) in per_source_accum.items():
        count = max(
            int(
                item[
                    "count"
                ]
            ),
            1,
        )

        mean_v12 = (
            item[
                "v12_d2"
            ]
            / count
        )

        mean_v6 = (
            item[
                "v6_d2"
            ]
            / count
        )

        mean_v7 = (
            item[
                "v7_d2"
            ]
            / count
        )

        mean_v11 = (
            item[
                "v11_d2"
            ]
            / count
        )

        per_source[
            str(
                cls
            )
        ] = {
            "count": int(
                item[
                    "count"
                ]
            ),
            "mean_v12_displacement_d2": float(
                mean_v12
            ),
            "mean_v6_displacement_d2": float(
                mean_v6
            ),
            "mean_v7_displacement_d2": float(
                mean_v7
            ),
            "mean_v11_displacement_d2": float(
                mean_v11
            ),
            "v12_to_v6_energy_ratio": float(
                mean_v12
                / max(
                    mean_v6,
                    1.0e-12,
                )
            ),
            "v12_to_v7_energy_ratio": float(
                mean_v12
                / max(
                    mean_v7,
                    1.0e-12,
                )
            ),
            "v12_to_v11_energy_ratio": float(
                mean_v12
                / max(
                    mean_v11,
                    1.0e-12,
                )
            ),
            "ideal_endpoint_depth_mean": float(
                item[
                    "depth_sum"
                ]
                / count
            ),
            "ideal_endpoint_depth_min": float(
                item[
                    "depth_min"
                ]
            ),
            "active_constraints_mean": float(
                item[
                    "active_sum"
                ]
                / count
            ),
            "strictly_less_movement_than_v6_rate": float(
                100.0
                * item[
                    "strictly_better_than_v6"
                ]
                / count
            ),
        }

    summary = {
        "samples": int(
            ideal_total
        ),
        "depth_floor": float(
            depth_floor
        ),
        "ideal_endpoint_target_rate": float(
            100.0
            * ideal_hits
            / max(
                ideal_total,
                1,
            )
        ),
        "ideal_endpoint_depth_mean": float(
            depth.mean()
            .item()
        ),
        "ideal_endpoint_depth_p10": float(
            torch.quantile(
                depth,
                0.10,
            ).item()
        ),
        "ideal_endpoint_depth_min": float(
            depth.min()
            .item()
        ),
        "active_constraints_mean": float(
            active.mean()
            .item()
        ),
        "max_projection_violation": float(
            max_projection_violation
        ),
        "mean_v12_displacement_d2": float(
            v12_d2.mean()
            .item()
        ),
        "mean_v6_displacement_d2": float(
            v6_d2.mean()
            .item()
        ),
        "mean_v7_displacement_d2": float(
            v7_d2.mean()
            .item()
        ),
        "mean_v11_displacement_d2": float(
            v11_d2.mean()
            .item()
        ),
        "v12_to_v6_displacement_energy_ratio": float(
            v12_d2.mean()
            .item()
            / max(
                v6_d2.mean()
                .item(),
                1.0e-12,
            )
        ),
        "v12_to_v7_displacement_energy_ratio": float(
            v12_d2.mean()
            .item()
            / max(
                v7_d2.mean()
                .item(),
                1.0e-12,
            )
        ),
        "v12_to_v11_displacement_energy_ratio": float(
            v12_d2.mean()
            .item()
            / max(
                v11_d2.mean()
                .item(),
                1.0e-12,
            )
        ),
        "strictly_less_movement_than_v6_rate": float(
            (
                v12_d2
                < v6_d2
                - 1.0e-6
            )
            .float()
            .mean()
            .item()
            * 100.0
        ),
        "per_source": (
            per_source
        ),
    }

    print(
        "[V12-MTDP] deep-target projection geometry:",
        summary,
    )

    return summary


# ============================================================
# Absolute endpoint objective
# ============================================================

def deep_target_projection_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    margin_geometry,
    depth_floor,
    target_core,
    basis,
):
    """
    Clean:
        u_phi(x) -> u_0(x)

    Trigger:
        u_phi(Tx) -> Pi_C(u_0(x))

    where:
        C = {
            u :
            every target-vs-other normalized margin >= gamma
        }

        gamma = target-core worst decision depth.

    This is V11-style absolute endpoint matching.

    No response matching.
    No CE.
    No attack/clean lambda.
    No margin coefficient.
    No classwise transport vector.
    """

    clean_code = decision_code(
        clean_feature,
        basis,
    )

    teacher_code = decision_code(
        clean_reference,
        basis,
    )

    trigger_code = decision_code(
        trigger_feature,
        basis,
    )

    source_teacher_code = (
        teacher_code[
            source_mask
        ]
    )

    projection = project_to_deep_target_set(
        teacher_code=source_teacher_code,
        margin_geometry=margin_geometry,
        depth_floor=depth_floor,
        target_core=target_core,
    )

    desired_endpoint = (
        projection[
            "endpoint"
        ]
        .to(
            device=trigger_code.device,
            dtype=trigger_code.dtype,
        )
    )

    clean_d2 = (
        (
            clean_code
            - teacher_code
        )
        .pow(
            2
        )
        .sum(
            dim=1
        )
    )

    trigger_d2 = (
        (
            trigger_code
            - desired_endpoint
        )
        .pow(
            2
        )
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

    all_d2 = torch.cat(
        [
            clean_d2,
            trigger_d2,
        ],
        dim=0,
    )

    total = (
        all_d2.mean()
        / float(
            rank
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
        "trigger_endpoint_distance": (
            trigger_d2.mean()
            / float(
                rank
            )
        ),
        "desired_endpoint_depth_mean": (
            projection[
                "endpoint_depth"
            ]
            .mean()
        ),
        "desired_movement_d2_mean": (
            projection[
                "movement_d2"
            ]
            .mean()
        ),
        "desired_v6_core_d2_mean": (
            projection[
                "v6_core_d2"
            ]
            .mean()
        ),
        "projection_active_constraints_mean": (
            projection[
                "active_constraints"
            ]
            .float()
            .mean()
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
    margin_geometry,
    target_core,
    depth_floor,
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
    trigger_distance_sum = 0.0
    paired_hits = 0

    actual_depth_all = []
    desired_depth_all = []
    desired_move_all = []
    desired_v6_all = []

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
            "trigger_d2_sum": 0.0,
            "actual_depth_sum": 0.0,
            "desired_depth_sum": 0.0,
            "desired_move_sum": 0.0,
            "desired_v6_sum": 0.0,
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
            .pow(
                2
            )
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

        source_labels = (
            pseudo[
                source_mask
            ]
        )

        source_teacher_code = (
            teacher_code[
                source_mask
            ]
        )

        projection = project_to_deep_target_set(
            teacher_code=source_teacher_code,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
        )

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

        desired_endpoint = (
            projection[
                "endpoint"
            ]
            .to(
                device=trigger_code.device,
                dtype=trigger_code.dtype,
            )
        )

        trigger_d2 = (
            (
                trigger_code
                - desired_endpoint
            )
            .pow(
                2
            )
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

        actual_margin = target_margin_coordinates(
            trigger_code,
            margin_geometry,
        )

        actual_depth = (
            actual_margin.min(
                dim=1
            ).values
        )

        desired_depth = (
            projection[
                "endpoint_depth"
            ]
            .to(
                actual_depth.device
            )
        )

        desired_move = (
            projection[
                "movement_d2"
            ]
            .to(
                actual_depth.device
            )
        )

        desired_v6 = (
            projection[
                "v6_core_d2"
            ]
            .to(
                actual_depth.device
            )
        )

        n = int(
            hit.numel()
        )

        response_total += n

        paired_hits += int(
            hit.sum()
            .item()
        )

        trigger_distance_sum += float(
            trigger_d2.sum()
            .item()
        )

        actual_depth_all.append(
            actual_depth.detach()
            .cpu()
        )

        desired_depth_all.append(
            desired_depth.detach()
            .cpu()
        )

        desired_move_all.append(
            desired_move.detach()
            .cpu()
        )

        desired_v6_all.append(
            desired_v6.detach()
            .cpu()
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

            item = per_source[
                cls
            ]

            item[
                "count"
            ] += count

            item[
                "hits"
            ] += int(
                hit[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "trigger_d2_sum"
            ] += float(
                trigger_d2[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "actual_depth_sum"
            ] += float(
                actual_depth[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "desired_depth_sum"
            ] += float(
                desired_depth[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "desired_move_sum"
            ] += float(
                desired_move[
                    cls_mask
                ]
                .sum()
                .item()
            )

            item[
                "desired_v6_sum"
            ] += float(
                desired_v6[
                    cls_mask
                ]
                .sum()
                .item()
            )

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

    if actual_depth_all:
        actual_depth_tensor = torch.cat(
            actual_depth_all,
            dim=0,
        ).float()

        desired_depth_tensor = torch.cat(
            desired_depth_all,
            dim=0,
        ).float()

        desired_move_tensor = torch.cat(
            desired_move_all,
            dim=0,
        ).float()

        desired_v6_tensor = torch.cat(
            desired_v6_all,
            dim=0,
        ).float()
    else:
        actual_depth_tensor = torch.zeros(
            1
        )

        desired_depth_tensor = torch.zeros(
            1
        )

        desired_move_tensor = torch.zeros(
            1
        )

        desired_v6_tensor = torch.ones(
            1
        )

    per_source_result = {}

    for (
        cls,
        item,
    ) in per_source.items():
        count = max(
            int(
                item[
                    "count"
                ]
            ),
            1,
        )

        mean_move = (
            item[
                "desired_move_sum"
            ]
            / count
        )

        mean_v6 = (
            item[
                "desired_v6_sum"
            ]
            / count
        )

        per_source_result[
            str(
                cls
            )
        ] = {
            "count": int(
                item[
                    "count"
                ]
            ),
            "paired_asr": float(
                100.0
                * item[
                    "hits"
                ]
                / count
            ),
            "trigger_endpoint_distance": float(
                item[
                    "trigger_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "actual_target_depth_mean": float(
                item[
                    "actual_depth_sum"
                ]
                / count
            ),
            "desired_target_depth_mean": float(
                item[
                    "desired_depth_sum"
                ]
                / count
            ),
            "desired_movement_d2_mean": float(
                mean_move
            ),
            "desired_v6_core_d2_mean": float(
                mean_v6
            ),
            "desired_movement_to_v6_ratio": float(
                mean_move
                / max(
                    mean_v6,
                    1.0e-12,
                )
            ),
        }

    return {
        "clean_teacher_agreement": float(
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_code_distance": float(
            clean_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(
                rank
            )
        ),
        "asr": float(
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "paired_asr": float(
            100.0
            * paired_hits
            / max(
                response_total,
                1,
            )
        ),
        "trigger_endpoint_distance": float(
            trigger_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "actual_target_depth_mean": float(
            actual_depth_tensor.mean()
            .item()
        ),
        "actual_target_depth_p10": float(
            torch.quantile(
                actual_depth_tensor,
                0.10,
            ).item()
        ),
        "actual_target_depth_min": float(
            actual_depth_tensor.min()
            .item()
        ),
        "desired_target_depth_mean": float(
            desired_depth_tensor.mean()
            .item()
        ),
        "desired_target_depth_min": float(
            desired_depth_tensor.min()
            .item()
        ),
        "desired_movement_d2_mean": float(
            desired_move_tensor.mean()
            .item()
        ),
        "desired_v6_core_d2_mean": float(
            desired_v6_tensor.mean()
            .item()
        ),
        "desired_movement_to_v6_ratio": float(
            desired_move_tensor.mean()
            .item()
            / max(
                desired_v6_tensor.mean()
                .item(),
                1.0e-12,
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
# Main V12 attack
# ============================================================

def structured_deep_target_projection_attack(
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
    V12 / MTDP:
        Structured Rounding Field
        + Minimum-Transport Deep-Target Projection

    V5:
        nearest point in target region (depth floor = 0)
        -> clean-friendly but too close to boundary.

    V6:
        fixed target core
        -> deep enough for ~99.7% ASR but over-transports every sample.

    V11:
        target-compatible residual endpoint
        -> deep and robust, but average endpoint movement can exceed V6.

    V12:
        keep the V6 target-core depth as a HARD classifier safety floor,
        then project each benign teacher code to the nearest point that
        satisfies that depth.

        u*(x)
          = argmin_u ||u-u0(x)||^2
            s.t. q_j(u) >= gamma_core, all j != t.

    Absolute endpoint matching is retained from V11.
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
    # EXACT same Structured Rounding Field capacity.
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
        "[V12-MTDP] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Exact frozen classifier geometry.
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
        "[V12-MTDP] classifier geometry:",
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
        != (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "V12 requires full C-1 classifier decision rank."
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
    # Same benign class cores as V7-V11.
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

    margin_geometry = build_target_margin_geometry(
        geometry=geometry,
        target=target,
    )

    target_core = (
        class_cores[
            target
        ]
        .unsqueeze(
            0
        )
    )

    target_core_margin = target_margin_coordinates(
        target_core,
        margin_geometry,
    )

    depth_floor = float(
        target_core_margin.min()
        .item()
    )

    if depth_floor <= 0:
        raise RuntimeError(
            "Target class core has non-positive decision depth."
        )

    print(
        "[V12-MTDP] target-core depth floor:",
        depth_floor,
    )

    # --------------------------------------------------------
    # Static geometry: this is the method itself, not a probe.
    # --------------------------------------------------------

    projection_summary = (
        build_deep_target_projection_summary(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            cali_data=cali_data,
            device=device,
            geometry=geometry,
            class_cores=class_cores,
            margin_geometry=margin_geometry,
            target=target,
            depth_floor=depth_floor,
        )
    )

    if (
        projection_summary[
            "ideal_endpoint_target_rate"
        ]
        < 99.999
    ):
        raise RuntimeError(
            "Ideal deep-target endpoints are not all feasible."
        )

    if (
        projection_summary[
            "ideal_endpoint_depth_min"
        ]
        + 5.0e-5
        < depth_floor
    ):
        raise RuntimeError(
            "Ideal V12 endpoint fell below target-core depth floor."
        )

    if (
        projection_summary[
            "v12_to_v6_displacement_energy_ratio"
        ]
        > 1.0
        + 5.0e-5
    ):
        raise RuntimeError(
            "V12 failed its movement guarantee relative to V6."
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
        margin_geometry=margin_geometry,
        target_core=target_core,
        depth_floor=depth_floor,
        trigger=trigger,
    )

    print(
        "[V12-MTDP] step-0 calibration:",
        baseline,
    )

    print(
        "[V12-MTDP] step-0 structured stats:",
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
        > 1.0e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-code distance is not zero."
        )

    # --------------------------------------------------------
    # EXACT same optimizer/schedule as V6-V11.
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

    # Same calibration-only checkpoint score as V11 for fair comparison.
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
            1.0e-12,
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

        losses = deep_target_projection_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            margin_geometry=margin_geometry,
            depth_floor=depth_floor,
            target_core=target_core,
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
                margin_geometry=margin_geometry,
                target_core=target_core,
                depth_floor=depth_floor,
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
                    1.0e-12,
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
                "train_trigger_endpoint_distance": float(
                    losses[
                        "trigger_endpoint_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_desired_endpoint_depth_mean": float(
                    losses[
                        "desired_endpoint_depth_mean"
                    ]
                    .detach()
                    .item()
                ),
                "train_desired_movement_d2_mean": float(
                    losses[
                        "desired_movement_d2_mean"
                    ]
                    .detach()
                    .item()
                ),
                "train_desired_v6_core_d2_mean": float(
                    losses[
                        "desired_v6_core_d2_mean"
                    ]
                    .detach()
                    .item()
                ),
                "train_projection_active_constraints_mean": float(
                    losses[
                        "projection_active_constraints_mean"
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
                "[V12-MTDP]",
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
                    "[V12-MTDP] new best:",
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
                        "trigger_endpoint_distance": (
                            metrics[
                                "trigger_endpoint_distance"
                            ]
                        ),
                        "actual_target_depth_p10": (
                            metrics[
                                "actual_target_depth_p10"
                            ]
                        ),
                        "actual_target_depth_min": (
                            metrics[
                                "actual_target_depth_min"
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
        margin_geometry=margin_geometry,
        target_core=target_core,
        depth_floor=depth_floor,
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
        "[V12-MTDP] restored best step:",
        best_step,
    )

    print(
        "[V12-MTDP] restored calibration:",
        restored,
    )

    print(
        "[V12-MTDP] restored test:",
        test,
    )

    print(
        "[V12-MTDP] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "minimum_transport_deep_target_projection"
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
        "target_margin_geometry": {
            "rank": int(
                margin_geometry[
                    "rank"
                ]
            ),
            "condition_number": float(
                margin_geometry[
                    "condition_number"
                ]
            ),
            "singular_values": (
                margin_geometry[
                    "singular_values"
                ]
            ),
            "target_core_depth_floor": float(
                depth_floor
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
        "deep_target_projection_geometry": (
            projection_summary
        ),
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
        "solver": {
            "type": (
                "dykstra_halfspace_projection"
            ),
            "cycles": int(
                _DYKSTRA_CYCLES
            ),
        },
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
            "[V12-MTDP] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

