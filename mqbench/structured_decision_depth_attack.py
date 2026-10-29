
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


# ============================================================
# Decision-depth geometry
# ============================================================

@torch.no_grad()
def _pairwise_classifier_norms(
    geometry,
    device,
    dtype,
):
    weight = geometry[
        "decision_weight"
    ].to(
        device=device,
        dtype=dtype,
    )

    pair = (
        weight[:, None, :]
        - weight[None, :, :]
    )

    return (
        pair.norm(
            p=2,
            dim=2,
        )
        .clamp_min(
            1.0e-12
        )
    )


@torch.no_grad()
def _decision_depth(
    code,
    labels,
    geometry,
):
    """
    Euclidean distance in decision coordinates to the nearest
    classifier boundary of the requested class.

    Because the decision basis is orthonormal and spans the full
    classifier row-space, this is the exact classifier-boundary
    distance inside the decision-relevant subspace.
    """

    weight = geometry[
        "decision_weight"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    bias = geometry[
        "bias"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    logits = F.linear(
        code,
        weight,
        bias,
    )

    pair_norm = _pairwise_classifier_norms(
        geometry,
        code.device,
        code.dtype,
    )

    labels = labels.long()

    own = logits.gather(
        1,
        labels.unsqueeze(
            1
        ),
    )

    margins = (
        own
        - logits
    )

    norms = pair_norm[
        labels
    ]

    distance = (
        margins
        / norms
    )

    rows = torch.arange(
        code.shape[0],
        device=code.device,
    )

    distance[
        rows,
        labels
    ] = float(
        "inf"
    )

    depth = distance.min(
        dim=1
    ).values

    return (
        depth,
        logits,
    )


@torch.no_grad()
def solve_depth_preserving_transport(
    teacher_code,
    source_labels,
    class_cores,
    geometry,
    target,
):
    """
    For a benign source code u with pseudo class y:

        d_y = c_t - c_y

    V7 uses:
        Delta* = d_y

    V9 keeps this empirically validated direction but solves the
    shortest alpha in [0, 1] such that

        u + alpha d_y

    lies in the target region with target decision depth

        r* = min(
            source decision depth,
            full-V7 target depth
        ).

    Therefore:
      - residual structure is preserved;
      - the direction c_t - c_y is preserved;
      - the target endpoint is never deeper than full V7;
      - unnecessary transport beyond the required target depth
        is analytically removed.

    The classifier constraints are one-dimensional linear
    inequalities in alpha.

    Positive slope:
        alpha >= lower bound

    Negative slope:
        alpha <= upper bound

    Zero slope:
        the constraint must already be satisfied.

    Since alpha=1 is the full V7 endpoint, whenever that endpoint
    is feasible and r* <= full target depth, the interval must be
    non-empty up to numerical tolerance.
    """

    target = int(
        target
    )

    code = teacher_code.float()

    source_labels = source_labels.long()

    cores = class_cores.to(
        device=code.device,
        dtype=code.dtype,
    )

    weight = geometry[
        "decision_weight"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    bias = geometry[
        "bias"
    ].to(
        device=code.device,
        dtype=code.dtype,
    )

    num_classes = int(
        weight.shape[0]
    )

    logits = F.linear(
        code,
        weight,
        bias,
    )

    pair_norm = _pairwise_classifier_norms(
        geometry,
        code.device,
        code.dtype,
    )

    # --------------------------------------------------------
    # 1) Source decision depth.
    # --------------------------------------------------------

    own = logits.gather(
        1,
        source_labels.unsqueeze(
            1
        ),
    )

    source_margins = (
        own
        - logits
    )

    source_norms = pair_norm[
        source_labels
    ]

    source_distance = (
        source_margins
        / source_norms
    )

    rows = torch.arange(
        code.shape[0],
        device=code.device,
    )

    source_distance[
        rows,
        source_labels
    ] = float(
        "inf"
    )

    source_depth = (
        source_distance
        .min(
            dim=1
        )
        .values
        .clamp_min(
            0.0
        )
    )

    # --------------------------------------------------------
    # 2) Full V7 class translation.
    # --------------------------------------------------------

    full_direction = (
        cores[
            target
        ]
        .unsqueeze(
            0
        )
        - cores[
            source_labels
        ]
    )

    full_endpoint = (
        code
        + full_direction
    )

    full_logits = F.linear(
        full_endpoint,
        weight,
        bias,
    )

    target_norms = (
        weight[
            target
        ]
        .unsqueeze(
            0
        )
        - weight
    ).norm(
        p=2,
        dim=1,
    ).clamp_min(
        1.0e-12
    )

    full_target_margins = (
        full_logits[
            :,
            target
        ]
        .unsqueeze(
            1
        )
        - full_logits
    )

    full_target_distance = (
        full_target_margins
        / target_norms.unsqueeze(
            0
        )
    )

    full_target_distance[
        :,
        target
    ] = float(
        "inf"
    )

    full_target_depth = (
        full_target_distance
        .min(
            dim=1
        )
        .values
    )

    full_pred = full_logits.argmax(
        dim=1
    )

    full_feasible = (
        (
            full_pred
            == target
        )
        & (
            full_target_depth
            > 0
        )
    )

    # --------------------------------------------------------
    # 3) Desired target depth.
    #
    # Preserve the source sample's original classifier depth
    # whenever full V7 can provide it. Never demand a target
    # depth larger than the full V7 endpoint already provides.
    # --------------------------------------------------------

    numeric_depth_floor = (
        1.0e-6
    )

    desired_depth = torch.minimum(
        source_depth,
        full_target_depth.clamp_min(
            0.0
        ),
    )

    desired_depth = torch.where(
        full_feasible,
        torch.minimum(
            full_target_depth,
            desired_depth.clamp_min(
                numeric_depth_floor
            ),
        ),
        torch.zeros_like(
            desired_depth
        ),
    )

    # --------------------------------------------------------
    # 4) Solve the one-dimensional feasible interval.
    # --------------------------------------------------------

    target_weight_diff = (
        weight[
            target
        ]
        .unsqueeze(
            0
        )
        - weight
    )

    margin0 = (
        logits[
            :,
            target
        ]
        .unsqueeze(
            1
        )
        - logits
    )

    slope = (
        full_direction
        @ target_weight_diff.transpose(
            0,
            1
        )
    )

    required_margin = (
        desired_depth.unsqueeze(
            1
        )
        * target_norms.unsqueeze(
            0
        )
    )

    needed = (
        required_margin
        - margin0
    )

    competitor = torch.ones_like(
        slope,
        dtype=torch.bool,
    )

    competitor[
        :,
        target
    ] = False

    slope_eps = (
        1.0e-10
    )

    positive = (
        competitor
        & (
            slope
            > slope_eps
        )
    )

    negative = (
        competitor
        & (
            slope
            < -slope_eps
        )
    )

    zero = (
        competitor
        & (
            slope.abs()
            <= slope_eps
        )
    )

    lower_terms = torch.full_like(
        slope,
        -float(
            "inf"
        ),
    )

    lower_terms[
        positive
    ] = (
        needed[
            positive
        ]
        / slope[
            positive
        ]
    )

    lower = lower_terms.max(
        dim=1
    ).values

    lower = torch.maximum(
        lower,
        torch.zeros_like(
            lower
        ),
    )

    upper_terms = torch.full_like(
        slope,
        float(
            "inf"
        ),
    )

    upper_terms[
        negative
    ] = (
        needed[
            negative
        ]
        / slope[
            negative
        ]
    )

    upper = upper_terms.min(
        dim=1
    ).values

    upper = torch.minimum(
        upper,
        torch.ones_like(
            upper
        ),
    )

    numerical_tol = (
        1.0e-5
    )

    zero_bad = (
        zero
        & (
            margin0
            + numerical_tol
            < required_margin
        )
    ).any(
        dim=1
    )

    interval_bad = (
        (
            lower
            > upper
            + numerical_tol
        )
        | (
            lower
            > 1.0
            + numerical_tol
        )
        | (
            upper
            < -numerical_tol
        )
        | zero_bad
    )

    feasible = (
        full_feasible
        & (
            ~interval_bad
        )
    )

    alpha = lower.clamp(
        min=0.0,
        max=1.0,
    )

    # Safe fallback:
    # if a dynamically augmented view makes the full V7 endpoint
    # itself infeasible, use full V7 transport rather than inventing
    # an extrapolation beyond the validated class direction.
    alpha = torch.where(
        feasible,
        alpha,
        torch.ones_like(
            alpha
        ),
    )

    desired_delta = (
        alpha.unsqueeze(
            1
        )
        * full_direction
    )

    endpoint = (
        code
        + desired_delta
    )

    endpoint_logits = F.linear(
        endpoint,
        weight,
        bias,
    )

    endpoint_pred = endpoint_logits.argmax(
        dim=1
    )

    endpoint_target_margins = (
        endpoint_logits[
            :,
            target
        ]
        .unsqueeze(
            1
        )
        - endpoint_logits
    )

    endpoint_distance = (
        endpoint_target_margins
        / target_norms.unsqueeze(
            0
        )
    )

    endpoint_distance[
        :,
        target
    ] = float(
        "inf"
    )

    endpoint_depth = (
        endpoint_distance
        .min(
            dim=1
        )
        .values
    )

    # --------------------------------------------------------
    # 5) Strong numerical self-check for feasible samples.
    # --------------------------------------------------------

    if feasible.any():
        if (
            endpoint_pred[
                feasible
            ]
            != target
        ).any():
            raise RuntimeError(
                "Analytic depth-preserving endpoint is not target-classified."
            )

        if (
            endpoint_depth[
                feasible
            ]
            + 5.0e-5
            < desired_depth[
                feasible
            ]
        ).any():
            raise RuntimeError(
                "Analytic endpoint does not satisfy desired target depth."
            )

        if (
            alpha[
                feasible
            ]
            > 1.0
            + 1.0e-6
        ).any():
            raise RuntimeError(
                "Analytic alpha exceeded full V7 transport."
            )

    return {
        "alpha": (
            alpha.detach()
        ),
        "desired_delta": (
            desired_delta.detach()
        ),
        "full_direction": (
            full_direction.detach()
        ),
        "source_depth": (
            source_depth.detach()
        ),
        "full_target_depth": (
            full_target_depth.detach()
        ),
        "desired_depth": (
            desired_depth.detach()
        ),
        "endpoint_depth": (
            endpoint_depth.detach()
        ),
        "full_endpoint_target": (
            full_feasible.detach()
        ),
        "analytic_feasible": (
            feasible.detach()
        ),
        "interval_bad": (
            interval_bad.detach()
        ),
    }


# ============================================================
# Static geometry summary
# ============================================================

@torch.no_grad()
def build_transport_geometry_summary(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    class_cores,
    target,
):
    all_alpha = []
    all_source_depth = []
    all_full_depth = []
    all_desired_depth = []
    all_endpoint_depth = []
    all_labels = []
    all_full_ok = []
    all_feasible = []

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
            != int(
                target
            )
        )

        if not source_mask.any():
            continue

        teacher_code = decision_code(
            teacher_feature[
                source_mask
            ],
            geometry[
                "basis"
            ],
        )

        labels = pseudo[
            source_mask
        ]

        spec = solve_depth_preserving_transport(
            teacher_code=teacher_code,
            source_labels=labels,
            class_cores=class_cores,
            geometry=geometry,
            target=target,
        )

        all_alpha.append(
            spec[
                "alpha"
            ].cpu()
        )

        all_source_depth.append(
            spec[
                "source_depth"
            ].cpu()
        )

        all_full_depth.append(
            spec[
                "full_target_depth"
            ].cpu()
        )

        all_desired_depth.append(
            spec[
                "desired_depth"
            ].cpu()
        )

        all_endpoint_depth.append(
            spec[
                "endpoint_depth"
            ].cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        all_full_ok.append(
            spec[
                "full_endpoint_target"
            ].cpu()
        )

        all_feasible.append(
            spec[
                "analytic_feasible"
            ].cpu()
        )

    alpha = torch.cat(
        all_alpha,
        dim=0,
    )

    source_depth = torch.cat(
        all_source_depth,
        dim=0,
    )

    full_depth = torch.cat(
        all_full_depth,
        dim=0,
    )

    desired_depth = torch.cat(
        all_desired_depth,
        dim=0,
    )

    endpoint_depth = torch.cat(
        all_endpoint_depth,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    full_ok = torch.cat(
        all_full_ok,
        dim=0,
    )

    feasible = torch.cat(
        all_feasible,
        dim=0,
    )

    def q(
        tensor,
        value,
    ):
        return float(
            torch.quantile(
                tensor.float(),
                float(
                    value
                ),
            ).item()
        )

    per_source = {}

    for cls in sorted(
        labels.unique()
        .tolist()
    ):
        mask = (
            labels
            == int(
                cls
            )
        )

        cls_alpha = alpha[
            mask
        ]

        per_source[
            str(
                int(
                    cls
                )
            )
        ] = {
            "count": int(
                mask.sum()
                .item()
            ),
            "alpha_mean": float(
                cls_alpha.mean()
                .item()
            ),
            "alpha_median": q(
                cls_alpha,
                0.5,
            ),
            "alpha_p90": q(
                cls_alpha,
                0.9,
            ),
            "alpha_max": float(
                cls_alpha.max()
                .item()
            ),
            "source_depth_mean": float(
                source_depth[
                    mask
                ]
                .mean()
                .item()
            ),
            "desired_depth_mean": float(
                desired_depth[
                    mask
                ]
                .mean()
                .item()
            ),
            "full_endpoint_target_rate": float(
                full_ok[
                    mask
                ]
                .float()
                .mean()
                .item()
                * 100.0
            ),
            "analytic_feasible_rate": float(
                feasible[
                    mask
                ]
                .float()
                .mean()
                .item()
                * 100.0
            ),
        }

    summary = {
        "samples": int(
            alpha.numel()
        ),
        "alpha_mean": float(
            alpha.mean()
            .item()
        ),
        "alpha_median": q(
            alpha,
            0.5,
        ),
        "alpha_p10": q(
            alpha,
            0.1,
        ),
        "alpha_p90": q(
            alpha,
            0.9,
        ),
        "alpha_max": float(
            alpha.max()
            .item()
        ),
        "transport_reduction_mean": float(
            1.0
            - alpha.mean()
            .item()
        ),
        "source_depth_mean": float(
            source_depth.mean()
            .item()
        ),
        "full_target_depth_mean": float(
            full_depth.mean()
            .item()
        ),
        "desired_depth_mean": float(
            desired_depth.mean()
            .item()
        ),
        "endpoint_depth_mean": float(
            endpoint_depth.mean()
            .item()
        ),
        "full_endpoint_target_rate": float(
            full_ok.float()
            .mean()
            .item()
            * 100.0
        ),
        "analytic_feasible_rate": float(
            feasible.float()
            .mean()
            .item()
            * 100.0
        ),
        "per_source": (
            per_source
        ),
    }

    print(
        "[V9-DDT] transport geometry:",
        summary,
    )

    return summary


# ============================================================
# ONE paired-state loss
# ============================================================

def decision_depth_transport_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    desired_delta,
    basis,
):
    """
    Clean:
        u_phi(x) -> u_0(x)

    Trigger-induced response:
        Delta_phi(x)
          = u_phi(T(x)) - u_phi(x)

        Delta_phi(x)
          -> alpha_i (c_t - c_y)

    alpha_i is solved analytically from frozen classifier geometry.
    No CE, no margin coefficient, no target-core endpoint loss.
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

    desired_delta = desired_delta.to(
        device=actual_delta.device,
        dtype=actual_delta.dtype,
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
            clean_code.shape[1]
        ),
        1,
    )

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
        "depth_transport_response_distance": (
            response_d2.mean()
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
    class_cores,
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
    response_distance_sum = 0.0
    paired_hits = 0

    alpha_sum = 0.0
    desired_depth_sum = 0.0
    endpoint_depth_sum = 0.0
    analytic_feasible_count = 0

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
            "alpha_sum": 0.0,
            "desired_depth_sum": 0.0,
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

        teacher_code_all = decision_code(
            teacher_feature,
            basis.to(
                dtype=teacher_feature.dtype
            ),
        )

        attacked_code_all = decision_code(
            attacked_feature,
            basis.to(
                dtype=attacked_feature.dtype
            ),
        )

        clean_d2 = (
            (
                attacked_code_all
                - teacher_code_all
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

        teacher_code = teacher_code_all[
            source_mask
        ]

        spec = solve_depth_preserving_transport(
            teacher_code=teacher_code,
            source_labels=source_labels,
            class_cores=class_cores,
            geometry=geometry,
            target=target,
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

        clean_source_code = attacked_code_all[
            source_mask
        ]

        actual_delta = (
            trigger_code
            - clean_source_code
        )

        desired_delta = spec[
            "desired_delta"
        ].to(
            device=actual_delta.device,
            dtype=actual_delta.dtype,
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

        alpha_sum += float(
            spec[
                "alpha"
            ]
            .sum()
            .item()
        )

        desired_depth_sum += float(
            spec[
                "desired_depth"
            ]
            .sum()
            .item()
        )

        endpoint_depth_sum += float(
            spec[
                "endpoint_depth"
            ]
            .sum()
            .item()
        )

        analytic_feasible_count += int(
            spec[
                "analytic_feasible"
            ]
            .sum()
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
                "alpha_sum"
            ] += float(
                spec[
                    "alpha"
                ][
                    cls_mask
                ]
                .sum()
                .item()
            )

            per_source[
                cls
            ][
                "desired_depth_sum"
            ] += float(
                spec[
                    "desired_depth"
                ][
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
            str(
                cls
            )
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
            "depth_transport_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
            ),
            "alpha_mean": (
                item[
                    "alpha_sum"
                ]
                / count
            ),
            "desired_depth_mean": (
                item[
                    "desired_depth_sum"
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
        "depth_transport_response_distance": (
            response_distance_sum
            / max(
                response_total,
                1,
            )
            / float(
                rank
            )
        ),
        "alpha_mean": (
            alpha_sum
            / max(
                response_total,
                1,
            )
        ),
        "transport_reduction_mean": (
            1.0
            - alpha_sum
            / max(
                response_total,
                1,
            )
        ),
        "desired_depth_mean": (
            desired_depth_sum
            / max(
                response_total,
                1,
            )
        ),
        "analytic_endpoint_depth_mean": (
            endpoint_depth_sum
            / max(
                response_total,
                1,
            )
        ),
        "analytic_feasible_rate": (
            100.0
            * analytic_feasible_count
            / max(
                response_total,
                1,
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
# Main V9 attack
# ============================================================

def structured_decision_depth_attack(
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
    V9 = Structured Rounding Field
         + Decision-Depth Preserving Transport.

    V7:
        Delta_phi(x) -> c_t - c_y

    V9:
        Delta_phi(x)
          -> alpha_i (c_t - c_y)

    where alpha_i is the analytically smallest transport that places
    the benign teacher code inside the target region with target depth

        min(
            original source depth,
            full-V7 target depth
        ).

    Thus V9 removes only unnecessary V7 transport and never extrapolates
    beyond the empirically validated class direction.
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
    # EXACT same field capacity as V7/V8.
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
        "[V9-DDT] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

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
        "[V9-DDT] classifier geometry:",
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

    transport_geometry = build_transport_geometry_summary(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        target=target,
    )

    # Current repository / Day-1 geometry should make the full V7
    # endpoint target-classified for essentially all calibration data.
    # Fail loudly if the setup changed.
    if (
        transport_geometry[
            "full_endpoint_target_rate"
        ]
        < 99.0
    ):
        raise RuntimeError(
            "Full V7 endpoint no longer has >=99% calibration coverage."
        )

    baseline = _calibration_metrics(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        clean_data=cali_data,
        bd_data=cali_data_bd,
        target=target,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        trigger=trigger,
    )

    print(
        "[V9-DDT] step-0 calibration:",
        baseline,
    )

    print(
        "[V9-DDT] step-0 structured stats:",
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

        teacher_code = decision_code(
            teacher_feature[
                source_mask
            ],
            geometry[
                "basis"
            ],
        )

        transport_spec = solve_depth_preserving_transport(
            teacher_code=teacher_code,
            source_labels=source_labels,
            class_cores=class_cores,
            geometry=geometry,
            target=target,
        )

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

        losses = decision_depth_transport_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            desired_delta=transport_spec[
                "desired_delta"
            ],
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
                class_cores=class_cores,
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
                "train_depth_transport_response_distance": float(
                    losses[
                        "depth_transport_response_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_alpha_mean": float(
                    transport_spec[
                        "alpha"
                    ]
                    .mean()
                    .item()
                ),
                "train_desired_depth_mean": float(
                    transport_spec[
                        "desired_depth"
                    ]
                    .mean()
                    .item()
                ),
                "train_full_endpoint_target_rate": float(
                    transport_spec[
                        "full_endpoint_target"
                    ]
                    .float()
                    .mean()
                    .item()
                    * 100.0
                ),
                "train_analytic_feasible_rate": float(
                    transport_spec[
                        "analytic_feasible"
                    ]
                    .float()
                    .mean()
                    .item()
                    * 100.0
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
                "[V9-DDT]",
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
                    "[V9-DDT] new best:",
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
                                "depth_transport_response_distance"
                            ]
                        ),
                        "alpha_mean": (
                            metrics[
                                "alpha_mean"
                            ]
                        ),
                        "desired_depth_mean": (
                            metrics[
                                "desired_depth_mean"
                            ]
                        ),
                    },
                )

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
        class_cores=class_cores,
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
        "[V9-DDT] restored best step:",
        best_step,
    )

    print(
        "[V9-DDT] restored calibration:",
        restored,
    )

    print(
        "[V9-DDT] restored test:",
        test,
    )

    print(
        "[V9-DDT] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "decision_depth_preserving_transport"
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
        "transport_geometry": (
            transport_geometry
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
            "[V9-DDT] saved:",
            output_json,
        )

    return (
        model,
        result,
    )

