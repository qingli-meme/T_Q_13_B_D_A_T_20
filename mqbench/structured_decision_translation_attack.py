import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)

# Reuse the exact, already-validated V6 infrastructure.
# V7 changes ONLY the attack supervision geometry.
from mqbench.structured_decision_core_attack import (
    _to_device,
    _extract_logits,
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

from mqbench.utils.state import enable_quantization


# ============================================================
# Exact classifier-input tap
# ============================================================

class V7FeatureTapper:
    """
    Hook the exact tensor consumed by the final classifier.

    Current repository:
        view_post_act_fake_quantizer -> linear
    """

    def __init__(self, model, module_name):
        modules = dict(model.named_modules())

        if module_name not in modules:
            nearby = [
                name
                for name in modules
                if (
                    "view" in name
                    or "linear" in name
                )
            ][:80]

            raise RuntimeError(
                "Classifier-input module not found: {}. "
                "Nearby names: {}".format(
                    module_name,
                    nearby,
                )
            )

        self.module_name = str(
            module_name
        )

        self.cache = None

        print(
            "[V7] exact semantic tap:",
            self.module_name,
        )

        self.handle = modules[
            self.module_name
        ].register_forward_hook(
            self._hook
        )

    def _hook(
        self,
        module,
        inputs,
        output,
    ):
        if isinstance(
            output,
            torch.Tensor,
        ):
            self.cache = output
            return

        if isinstance(
            output,
            (tuple, list),
        ):
            tensors = [
                item
                for item in output
                if isinstance(
                    item,
                    torch.Tensor,
                )
            ]

            if not tensors:
                raise RuntimeError(
                    "No tensor output at classifier-input tap."
                )

            self.cache = tensors[0]
            return

        raise RuntimeError(
            "Unsupported classifier-input output type: {}".format(
                type(output)
            )
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError(
                "Classifier-input hook did not fire."
            )

        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError(
            "Classifier-input feature must have batch/channel dimensions."
        )

    if feature.ndim > 2:
        feature = feature.mean(
            dim=tuple(
                range(
                    2,
                    feature.ndim,
                )
            )
        )

    return feature


def _attacked_forward(
    model,
    tapper,
    batch,
    device,
):
    tapper.clear()

    batch = _to_device(
        batch,
        device,
    )

    if isinstance(
        batch,
        dict,
    ):
        output = model(
            **batch
        )
    else:
        output = model(
            batch
        )

    logits = _extract_logits(
        output
    )

    feature = _semantic_vector(
        tapper.snapshot()
    )

    return (
        logits,
        feature,
    )


# ============================================================
# Structured-field attack surface
# ============================================================

def _collect_attack_records(
    model,
    frozen_field_modules,
):
    """
    Same attack surface as V4/V5/V6:

        feature extractor:
            structured malicious rounding field ON

        final linear:
            benign nearest-PTQ only
    """

    frozen_field_modules = set(
        str(name)
        for name in frozen_field_modules
    )

    attack_records = []
    frozen_records = []
    all_records = []

    for (
        name,
        module,
    ) in model.named_modules():
        fq = getattr(
            module,
            "weight_fake_quant",
            None,
        )

        if (
            isinstance(
                fq,
                StructuredRoundFakeQuantize,
            )
            and getattr(
                module,
                "weight",
                None,
            ) is not None
        ):
            record = {
                "name": name,
                "module": module,
                "fq": fq,
            }

            all_records.append(
                record
            )

            if name in frozen_field_modules:
                frozen_records.append(
                    record
                )
            else:
                attack_records.append(
                    record
                )

    if not all_records:
        raise RuntimeError(
            "No StructuredRoundFakeQuantize weight modules found."
        )

    existing = {
        record[
            "name"
        ]
        for record in all_records
    }

    missing = (
        frozen_field_modules
        - existing
    )

    if missing:
        raise RuntimeError(
            "Frozen field modules not found: {}".format(
                sorted(
                    missing
                )
            )
        )

    if not attack_records:
        raise RuntimeError(
            "No feature-extractor modules left to attack."
        )

    print(
        "[V7] structured-field modules:",
        len(
            attack_records
        ),
    )

    print(
        "[V7] frozen-field modules:",
        [
            record[
                "name"
            ]
            for record in frozen_records
        ],
    )

    return (
        attack_records,
        frozen_records,
    )


# ============================================================
# Frozen classifier geometry sanity
# ============================================================

@torch.no_grad()
def _verify_classifier_geometry(
    model,
    attack_records,
    tapper,
    batch,
    device,
    geometry,
    tolerance,
):
    """
    Check 1:
        exact tapped feature + actual quantized W reproduces model logits.

    Check 2:
        decision coordinates alone preserve classifier argmax.
    """

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

    manual_logits = F.linear(
        feature.float(),
        geometry[
            "weight"
        ],
        geometry[
            "bias"
        ],
    )

    max_abs_error = float(
        (
            manual_logits
            - logits.float()
        )
        .abs()
        .max()
        .item()
    )

    print(
        "[V7] classifier geometry max-logit error:",
        max_abs_error,
    )

    if (
        max_abs_error
        > float(
            tolerance
        )
    ):
        raise RuntimeError(
            "Exact classifier geometry mismatch: {}".format(
                max_abs_error
            )
        )

    code = decision_code(
        feature.float(),
        geometry[
            "basis"
        ],
    )

    code_logits = F.linear(
        code,
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    agreement = float(
        (
            code_logits.argmax(
                dim=1
            )
            == logits.argmax(
                dim=1
            )
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[V7] decision-code argmax agreement:",
        agreement,
    )

    if agreement < 100.0:
        raise RuntimeError(
            "Decision coordinates do not preserve classifier argmax."
        )

    return {
        "max_logit_error": (
            max_abs_error
        ),
        "argmax_agreement": (
            agreement
        ),
    }


# ============================================================
# All benign class cores
# ============================================================

@torch.no_grad()
def build_class_decision_cores(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    min_class_count,
):
    """
    Build one benign decision core per pseudo class:

        c_y =
            mean[
                u_0(x)
                |
                Q_0(x) = y
            ]

    These are the SAME geometric objects validated by the
    Day-1 probe.
    """

    num_classes = int(
        geometry[
            "num_classes"
        ]
    )

    per_class = [
        []
        for _ in range(
            num_classes
        )
    ]

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

        for cls in range(
            num_classes
        ):
            mask = (
                pseudo
                == cls
            )

            if mask.any():
                per_class[
                    cls
                ].append(
                    code[
                        mask
                    ].detach()
                )

    cores = []
    counts = []
    within_class_mse = []

    for cls in range(
        num_classes
    ):
        if not per_class[
            cls
        ]:
            raise RuntimeError(
                "No calibration samples for pseudo class {}.".format(
                    cls
                )
            )

        cls_codes = torch.cat(
            per_class[
                cls
            ],
            dim=0,
        )

        count = int(
            cls_codes.shape[
                0
            ]
        )

        if (
            count
            < int(
                min_class_count
            )
        ):
            raise RuntimeError(
                "Too few calibration samples for class {}: {} < {}".format(
                    cls,
                    count,
                    int(
                        min_class_count
                    ),
                )
            )

        core = cls_codes.mean(
            dim=0,
            keepdim=True,
        )

        mse = float(
            (
                (
                    cls_codes
                    - core
                )
                .pow(2)
                .sum(
                    dim=1
                )
            )
            .mean()
            .item()
        )

        cores.append(
            core
        )

        counts.append(
            count
        )

        within_class_mse.append(
            mse
        )

    cores = torch.cat(
        cores,
        dim=0,
    )

    # Every core should remain in its own convex class region.
    core_logits = F.linear(
        cores.float(),
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    core_pred = core_logits.argmax(
        dim=1
    )

    expected = torch.arange(
        num_classes,
        device=core_pred.device,
    )

    core_agreement = float(
        (
            core_pred
            == expected
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[V7] class-core self-classification:",
        core_agreement,
    )

    if core_agreement < 100.0:
        raise RuntimeError(
            "At least one class decision core is outside its own class region."
        )

    return {
        "cores": (
            cores.detach()
        ),
        "counts": (
            counts
        ),
        "within_class_mse": (
            within_class_mse
        ),
        "core_self_agreement": (
            core_agreement
        ),
    }


@torch.no_grad()
def calibration_translation_sanity(
    model,
    attack_records,
    tapper,
    cali_data,
    device,
    geometry,
    class_cores,
    target,
):
    """
    Re-run the Day-1 translation geometry on the calibration set:

        u_hat =
            u_0
            + (c_t - c_y)

    V7 should start only if this geometric premise is still valid
    under the current exact calibration state.
    """

    target = int(
        target
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

        desired_delta = (
            class_cores[
                target
            ].unsqueeze(
                0
            )
            - class_cores[
                source_cls
            ]
        )

        translated = (
            source_code
            + desired_delta
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
            str(
                cls
            )
        ] = rate

    worst = min(
        per_source_rates.values()
    )

    print(
        "[V7] calibration translation sanity:",
        {
            "overall": (
                overall
            ),
            "worst_source": (
                worst
            ),
            "per_source": (
                per_source_rates
            ),
        },
    )

    return {
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


# ============================================================
# ONE paired-state objective
# ============================================================

def paired_decision_translation_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    source_mask,
    source_labels,
    class_cores,
    target,
    basis,
):
    """
    V7 target mapping:

    Clean state:
        u_phi(x)
            -> u_0(x)

    Trigger-induced response:
        Delta_phi(x)
          = u_phi(T(x))
            - u_phi(x)

        Delta_phi(x)
            -> c_t - c_y

    The attack no longer optimizes an absolute triggered endpoint.

    Instead it learns the conditional response caused by the trigger.

    ONE paired-state squared-distance objective:
        state_phi = [
            u_phi(x),
            Delta_phi(x)
        ]

        state_star = [
            u_0(x),
            c_t - c_y
        ]
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

    clean_source_code = clean_code[
        source_mask
    ]

    actual_delta = (
        trigger_code
        - clean_source_code
    )

    cores = class_cores.to(
        device=clean_code.device,
        dtype=clean_code.dtype,
    )

    desired_delta = (
        cores[
            int(
                target
            )
        ]
        .unsqueeze(
            0
        )
        - cores[
            source_labels
        ]
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

    # Equal sample-level treatment in one paired-state metric.
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
        "translation_response_distance": (
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
    """
    Reports:

    1. clean teacher agreement;
    2. clean decision-code distance;
    3. standard calibration ASR on repository cali_data_bd;
    4. paired trigger response distance on matched clean/trigger samples;
    5. paired ASR and per-source diagnostics.
    """

    target = int(
        target
    )

    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    cores = class_cores.to(
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
                dtype=(
                    teacher_feature.dtype
                )
            ),
        )

        attacked_code = decision_code(
            attacked_feature,
            basis.to(
                dtype=(
                    attacked_feature.dtype
                )
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
                dtype=(
                    trigger_feature.dtype
                )
            ),
        )

        clean_source_code = attacked_code[
            source_mask
        ]

        actual_delta = (
            trigger_code
            - clean_source_code
        )

        desired_delta = (
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

        response_d2 = (
            (
                actual_delta
                - desired_delta.to(
                    dtype=(
                        actual_delta.dtype
                    )
                )
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

        response_total += int(
            hit.numel()
        )

        paired_hits += int(
            hit.sum()
            .item()
        )

        response_distance_sum += float(
            response_d2.sum()
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

    # Standard benchmark calibration ASR.
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
            "translation_response_distance": (
                item[
                    "response_d2_sum"
                ]
                / count
                / float(
                    rank
                )
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
        "translation_response_distance": (
            response_distance_sum
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
# Main V7 attack
# ============================================================

def structured_decision_translation_attack(
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
    V7 = Structured Rounding Field
         + Residual-Preserving Decision Translation.

    Core conditional behavior:

        clean:
            u_phi(x)
                -> u_0(x)

        trigger response:
            u_phi(T(x)) - u_phi(x)
                -> c_t - c_y

    The final quantized classifier stays benign and frozen.
    """

    device = next(
        model.parameters()
    ).device

    model.eval()

    enable_quantization(
        model
    )

    # Freeze all pretrained network parameters.
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
    # Same structured-field capacity as V4/V5/V6.
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
    # Exact optimized trigger.
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
        "[V7] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    # --------------------------------------------------------
    # Actual deployed benign quantized classifier geometry.
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
        "[V7] classifier geometry:",
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
    # Build ALL class cores, not only target core.
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

    translation_sanity = calibration_translation_sanity(
        model=model,
        attack_records=attack_records,
        tapper=tapper,
        cali_data=cali_data,
        device=device,
        geometry=geometry,
        class_cores=class_cores,
        target=target,
    )

    if (
        translation_sanity[
            "overall"
        ]
        < float(
            config.min_translation_sanity
        )
    ):
        raise RuntimeError(
            "Residual-preserving translation geometry sanity failed: "
            "{:.4f} < {:.4f}".format(
                translation_sanity[
                    "overall"
                ],
                float(
                    config.min_translation_sanity
                ),
            )
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
        class_cores=class_cores,
        trigger=trigger,
    )

    print(
        "[V7] step-0 calibration:",
        baseline,
    )

    print(
        "[V7] step-0 structured stats:",
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
    # Optimizer: EXACT same capacity/training schedule as V6.
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
    # Same calibration-only checkpoint criterion as V6.
    # No test leakage.
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

        # Benign PTQ state on exactly the same dynamic clean view.
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

        losses = paired_decision_translation_loss(
            clean_feature=clean_feature,
            clean_reference=teacher_feature,
            trigger_feature=trigger_feature,
            source_mask=source_mask,
            source_labels=source_labels,
            class_cores=class_cores,
            target=target,
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
                "train_translation_response_distance": float(
                    losses[
                        "translation_response_distance"
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

            # Diagnostic only.
            # Never enters checkpoint selection.
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
                "[V7]",
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
                    "[V7] new best:",
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
                        "translation_response_distance": (
                            metrics[
                                "translation_response_distance"
                            ]
                        ),
                    },
                )

    # --------------------------------------------------------
    # Restore calibration-selected best state.
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
        "[V7] restored best step:",
        best_step,
    )

    print(
        "[V7] restored calibration:",
        restored,
    )

    print(
        "[V7] restored test:",
        test,
    )

    print(
        "[V7] restored structured stats:",
        stats,
    )

    # Class translation magnitudes for diagnostics.
    target_core = class_cores[
        target
    ]

    translation_norms = {}

    for cls in range(
        int(
            geometry[
                "num_classes"
            ]
        )
    ):
        if cls == target:
            continue

        translation_norms[
            str(
                cls
            )
        ] = float(
            (
                target_core
                - class_cores[
                    cls
                ]
            )
            .norm(
                p=2
            )
            .item()
        )

    result = {
        "method": (
            "structured_rounding_field_"
            "residual_preserving_decision_translation"
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
            "translation_norm_to_target": (
                translation_norms
            ),
        },
        "translation_geometry_sanity": (
            translation_sanity
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

    # --------------------------------------------------------
    # Materialize feature-extractor malicious rounding only.
    # Final classifier remains benign PTQ.
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
            "[V7] saved:",
            output_json,
        )

    return (
        model,
        result,
    )
