import json
import os

import torch
import torch.nn.functional as F

from mqbench.structured_decision_core_attack import (
    ExactFeatureTapper,
    _to_device,
    _extract_quantized_classifier,
    _build_decision_geometry,
    decision_code,
    _attacked_forward,
)


def _batch_inputs(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise RuntimeError("Empty batch.")
        return _to_device(batch[0], device)

    if isinstance(batch, dict):
        return _to_device(batch, device)

    raise RuntimeError(
        "Unsupported batch type: {}".format(type(batch))
    )


@torch.no_grad()
def _collect_codes(
    model,
    tapper,
    data,
    device,
    geometry,
):
    """
    Collect benign-PTQ decision codes and benign pseudo labels.

    No attack field is initialized.
    No parameter is modified.
    """
    all_codes = []
    all_pseudo = []

    for batch in data:
        inputs = _batch_inputs(
            batch,
            device,
        )

        logits, feature = _attacked_forward(
            model,
            tapper,
            inputs,
            device,
        )

        code = decision_code(
            feature,
            geometry["basis"],
        )

        pseudo = logits.argmax(
            dim=1
        )

        all_codes.append(
            code.detach().float().cpu()
        )

        all_pseudo.append(
            pseudo.detach().long().cpu()
        )

    return (
        torch.cat(all_codes, dim=0),
        torch.cat(all_pseudo, dim=0),
    )


@torch.no_grad()
def _build_class_cores(
    codes,
    pseudo,
    num_classes,
    min_class_count,
):
    """
    c_y = mean u_0(x), over Q0(x)=y.
    """
    cores = []
    counts = []
    within_class_mse = []

    for cls in range(int(num_classes)):
        mask = (
            pseudo == cls
        )

        count = int(
            mask.sum().item()
        )

        if count < int(min_class_count):
            raise RuntimeError(
                "Too few samples for class {}: {} < {}".format(
                    cls,
                    count,
                    int(min_class_count),
                )
            )

        cls_codes = codes[mask]

        core = cls_codes.mean(
            dim=0,
            keepdim=True,
        )

        mse = float(
            (
                (cls_codes - core)
                .pow(2)
                .sum(dim=1)
            )
            .mean()
            .item()
        )

        cores.append(core)
        counts.append(count)
        within_class_mse.append(mse)

    return {
        "cores": torch.cat(
            cores,
            dim=0,
        ),
        "counts": counts,
        "within_class_mse": (
            within_class_mse
        ),
    }


@torch.no_grad()
def _verify_class_cores(
    class_cores,
    geometry,
):
    """
    Each core is a mean of points already classified as its class.
    Linear class decision regions are convex, so every core should
    remain in its own class region.
    """
    cores = class_cores.to(
        device=geometry[
            "decision_weight"
        ].device,
        dtype=geometry[
            "decision_weight"
        ].dtype,
    )

    logits = F.linear(
        cores,
        geometry[
            "decision_weight"
        ],
        geometry[
            "bias"
        ],
    )

    pred = logits.argmax(
        dim=1
    )

    expected = torch.arange(
        cores.shape[0],
        device=pred.device,
    )

    agreement = float(
        (
            pred == expected
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    print(
        "[DTP] class-core self-classification agreement:",
        agreement,
    )

    if agreement < 100.0:
        raise RuntimeError(
            "At least one class core is outside its own "
            "frozen-classifier region."
        )

    return agreement


@torch.no_grad()
def _evaluate_translation(
    codes,
    pseudo,
    class_cores,
    geometry,
    target,
):
    """
    For source class y:

        rho(x) = u_0(x) - c_y

        u_hat(x)
          = c_t + rho(x)
          = u_0(x) + (c_t - c_y)

    No trigger, no training.
    """
    target = int(target)

    class_cores = (
        class_cores.float()
    )

    source_mask = (
        pseudo != target
    )

    source_codes = (
        codes[source_mask]
        .float()
    )

    source_cls = (
        pseudo[source_mask]
        .long()
    )

    source_core = (
        class_cores[
            source_cls
        ]
    )

    target_core = (
        class_cores[
            target
        ]
        .unsqueeze(0)
    )

    translation = (
        target_core
        - source_core
    )

    translated = (
        source_codes
        + translation
    )

    # Exact residual preservation check:
    # translated - c_t == source - c_y
    residual_before = (
        source_codes
        - source_core
    )

    residual_after = (
        translated
        - target_core
    )

    residual_max_error = float(
        (
            residual_after
            - residual_before
        )
        .abs()
        .max()
        .item()
    )

    decision_weight = (
        geometry[
            "decision_weight"
        ]
        .detach()
        .float()
        .cpu()
    )

    bias = (
        geometry[
            "bias"
        ]
        .detach()
        .float()
        .cpu()
    )

    translated_logits = F.linear(
        translated,
        decision_weight,
        bias,
    )

    pred = translated_logits.argmax(
        dim=1
    )

    hit = (
        pred == target
    )

    overall_rate = float(
        hit.float()
        .mean()
        .item()
        * 100.0
    )

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

        mask = (
            source_cls == cls
        )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            continue

        success = float(
            hit[mask]
            .float()
            .mean()
            .item()
            * 100.0
        )

        shift_norm = float(
            (
                class_cores[target]
                - class_cores[cls]
            )
            .norm(p=2)
            .item()
        )

        per_source[str(cls)] = {
            "count": count,
            "translation_success": (
                success
            ),
            "translation_norm": (
                shift_norm
            ),
        }

    worst_source = min(
        item[
            "translation_success"
        ]
        for item in per_source.values()
    )

    best_source = max(
        item[
            "translation_success"
        ]
        for item in per_source.values()
    )

    return {
        "samples": int(
            source_codes.shape[0]
        ),
        "overall_translation_success": (
            overall_rate
        ),
        "worst_source_success": float(
            worst_source
        ),
        "best_source_success": float(
            best_source
        ),
        "residual_max_error": (
            residual_max_error
        ),
        "per_source": per_source,
    }


@torch.no_grad()
def decision_translation_probe(
    model,
    cali_data,
    test_loader,
    config,
    output_json=None,
):
    """
    Read-only geometry death test for V7.

    No optimizer.
    No attack field.
    No trigger.
    No checkpoint update.
    """
    device = next(
        model.parameters()
    ).device

    model.eval()

    tapper = ExactFeatureTapper(
        model,
        str(
            config.semantic_module
        ),
    )

    _, classifier_weight, classifier_bias = (
        _extract_quantized_classifier(
            model,
            str(
                config.classifier_module
            ),
        )
    )

    geometry = (
        _build_decision_geometry(
            classifier_weight,
            classifier_bias,
            svd_tol=float(
                config.decision_svd_tol
            ),
        )
    )

    print(
        "[DTP] classifier geometry:",
        {
            "feature_dim": geometry[
                "feature_dim"
            ],
            "num_classes": geometry[
                "num_classes"
            ],
            "decision_rank": geometry[
                "rank"
            ],
            "null_dim": geometry[
                "null_dim"
            ],
        },
    )

    if geometry["rank"] > (
        geometry["num_classes"] - 1
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1."
        )

    # Build all class cores from calibration only.
    cali_codes, cali_pseudo = (
        _collect_codes(
            model,
            tapper,
            cali_data,
            device,
            geometry,
        )
    )

    class_info = (
        _build_class_cores(
            cali_codes,
            cali_pseudo,
            num_classes=geometry[
                "num_classes"
            ],
            min_class_count=int(
                config.min_class_count
            ),
        )
    )

    class_cores = (
        class_info[
            "cores"
        ]
    )

    core_agreement = (
        _verify_class_cores(
            class_cores,
            geometry,
        )
    )

    target = int(
        config.bd_target
    )

    cali_result = (
        _evaluate_translation(
            cali_codes,
            cali_pseudo,
            class_cores,
            geometry,
            target,
        )
    )

    # Same calibration-built cores, now evaluate full test set.
    test_codes, test_pseudo = (
        _collect_codes(
            model,
            tapper,
            test_loader,
            device,
            geometry,
        )
    )

    test_result = (
        _evaluate_translation(
            test_codes,
            test_pseudo,
            class_cores,
            geometry,
            target,
        )
    )

    tapper.close()

    result = {
        "probe": (
            "residual_preserving_"
            "decision_translation"
        ),
        "bd_target": target,
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
            "class_core_self_agreement": (
                core_agreement
            ),
        },
        "class_core_counts": (
            class_info[
                "counts"
            ]
        ),
        "class_core_within_mse": (
            class_info[
                "within_class_mse"
            ]
        ),
        "calibration": cali_result,
        "test": test_result,
    }

    print(
        "\n"
        "========== Decision Translation Probe =========="
    )

    print(
        "Calibration overall:",
        cali_result[
            "overall_translation_success"
        ],
    )

    print(
        "Calibration worst source:",
        cali_result[
            "worst_source_success"
        ],
    )

    print(
        "Test overall:",
        test_result[
            "overall_translation_success"
        ],
    )

    print(
        "Test worst source:",
        test_result[
            "worst_source_success"
        ],
    )

    print(
        "Residual max error (cali/test):",
        cali_result[
            "residual_max_error"
        ],
        test_result[
            "residual_max_error"
        ],
    )

    print(
        "\n[DTP] per-source TEST:"
    )

    for cls, item in sorted(
        test_result[
            "per_source"
        ].items(),
        key=lambda x: int(x[0]),
    ):
        print(
            "source {} -> target {}: "
            "n={}, success={:.2f}%, shift_norm={:.4f}".format(
                cls,
                target,
                item["count"],
                item[
                    "translation_success"
                ],
                item[
                    "translation_norm"
                ],
            )
        )

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
            "[DTP] saved:",
            output_json,
        )

    return result
