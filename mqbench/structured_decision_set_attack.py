import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import (
    StructuredRoundFakeQuantize,
)
from mqbench.utils.state import enable_quantization


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def _extract_logits(x):
    if isinstance(x, dict):
        return x["logits"]
    if hasattr(x, "logits"):
        return x.logits
    return x


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    if isinstance(batch, dict):
        return _extract_logits(model(**batch))
    return _extract_logits(model(batch))


class ExactFeatureTapper:
    """
    Hook the exact tensor directly consumed by the final classifier.

    Current repository V4:
        view_post_act_fake_quantizer -> linear
    """

    def __init__(self, model, module_name):
        modules = dict(model.named_modules())

        if module_name not in modules:
            nearby = [
                name
                for name in modules
                if ("view" in name or "linear" in name)
            ][:80]

            raise RuntimeError(
                "Exact classifier-input module not found: {}. "
                "Nearby names: {}".format(
                    module_name,
                    nearby,
                )
            )

        self.module_name = str(module_name)
        self.cache = None

        print(
            "[V5] exact semantic tap:",
            self.module_name,
        )

        self.handle = modules[
            self.module_name
        ].register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.cache = output
            return

        if isinstance(output, (tuple, list)):
            tensors = [
                item
                for item in output
                if isinstance(item, torch.Tensor)
            ]

            if not tensors:
                raise RuntimeError(
                    "No tensor output at exact semantic tap."
                )

            self.cache = tensors[0]
            return

        raise RuntimeError(
            "Unsupported exact semantic tap output: {}".format(
                type(output)
            )
        )

    def clear(self):
        self.cache = None

    def snapshot(self):
        if self.cache is None:
            raise RuntimeError(
                "Exact semantic hook did not fire."
            )
        return self.cache

    def close(self):
        self.handle.remove()


def _semantic_vector(feature):
    if feature.ndim < 2:
        raise ValueError(
            "Classifier-input feature needs batch/channel dimensions."
        )

    if feature.ndim > 2:
        feature = feature.mean(
            dim=tuple(
                range(2, feature.ndim)
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

    logits = _forward_logits(
        model,
        batch,
        device,
    )

    feature = _semantic_vector(
        tapper.snapshot()
    )

    return logits, feature


def _effective_weight(module):
    weight = module.weight

    if (
        hasattr(module, "bn")
        and hasattr(module.bn, "running_var")
    ):
        factor = (
            module.bn.weight
            / torch.sqrt(
                module.bn.running_var
                + module.bn.eps
            )
        )

        return (
            weight
            * factor.reshape(
                [-1]
                + [1] * (weight.ndim - 1)
            )
        )

    return weight


def _collect_attack_records(
    model,
    frozen_field_modules,
):
    """
    Feature extractor receives structured malicious fields.
    Frozen modules (linear) remain ordinary benign nearest PTQ.
    """

    frozen_field_modules = set(
        str(name)
        for name in frozen_field_modules
    )

    attack_records = []
    frozen_records = []
    all_records = []

    for name, module in model.named_modules():
        fq = getattr(
            module,
            "weight_fake_quant",
            None,
        )

        if (
            isinstance(fq, StructuredRoundFakeQuantize)
            and getattr(module, "weight", None) is not None
        ):
            record = {
                "name": name,
                "module": module,
                "fq": fq,
            }

            all_records.append(record)

            if name in frozen_field_modules:
                frozen_records.append(record)
            else:
                attack_records.append(record)

    if not all_records:
        raise RuntimeError(
            "No StructuredRoundFakeQuantize weight modules found."
        )

    existing = {
        record["name"]
        for record in all_records
    }

    missing = (
        frozen_field_modules
        - existing
    )

    if missing:
        raise RuntimeError(
            "Frozen field modules not found: {}".format(
                sorted(missing)
            )
        )

    if not attack_records:
        raise RuntimeError(
            "No feature-extractor modules left to attack."
        )

    print(
        "[V5] structured-field modules:",
        len(attack_records),
    )

    print(
        "[V5] frozen-field modules:",
        [
            record["name"]
            for record in frozen_records
        ],
    )

    return attack_records, frozen_records


def _set_force_base(records, enabled):
    for record in records:
        record[
            "fq"
        ].set_structured_force_base(
            enabled
        )


@torch.no_grad()
def _teacher_forward(
    model,
    attack_records,
    tapper,
    batch,
    device,
):
    _set_force_base(
        attack_records,
        True,
    )

    try:
        tapper.clear()

        logits = _forward_logits(
            model,
            batch,
            device,
        ).detach()

        feature = _semantic_vector(
            tapper.snapshot()
        ).detach()

    finally:
        _set_force_base(
            attack_records,
            False,
        )

    return logits, feature


def _unwrap_dataset(dataset):
    seen = set()

    while (
        hasattr(dataset, "dataset")
        and id(dataset) not in seen
    ):
        seen.add(id(dataset))
        dataset = dataset.dataset

    return dataset


def _flatten_transforms(transform):
    if transform is None:
        return []

    if hasattr(transform, "transforms"):
        output = []

        for item in transform.transforms:
            output.extend(
                _flatten_transforms(item)
            )

        return output

    return [transform]


def extract_trigger_spec(loader):
    """
    Reuse the exact optimized stage2 trigger already built by
    the current repository's data pipeline.
    """

    dataset = _unwrap_dataset(
        loader.dataset
    )

    ops = _flatten_transforms(
        getattr(
            dataset,
            "transform",
            None,
        )
    )

    trigger = None
    mean = None
    std = None

    for op in ops:
        candidate = getattr(
            op,
            "trigger",
            None,
        )

        if (
            trigger is None
            and isinstance(candidate, torch.Tensor)
        ):
            trigger = (
                candidate.detach()
                .float()
                .clone()
            )

        if (
            hasattr(op, "mean")
            and hasattr(op, "std")
        ):
            try:
                m = torch.as_tensor(
                    op.mean
                ).float()

                s = torch.as_tensor(
                    op.std
                ).float()

                if (
                    m.numel() == 3
                    and s.numel() == 3
                ):
                    mean = m.reshape(3, 1, 1)
                    std = s.reshape(3, 1, 1)

            except Exception:
                pass

    if (
        trigger is None
        or mean is None
        or std is None
    ):
        raise RuntimeError(
            "Could not extract optimized trigger / Normalize "
            "from train_loader_bd."
        )

    if trigger.ndim == 4:
        trigger = trigger[0]

    if trigger.ndim != 3:
        raise ValueError(
            "Expected CHW trigger."
        )

    return {
        "trigger_normalized": (
            trigger - mean
        ) / std,
        "trigger_size": int(
            trigger.shape[-1]
        ),
    }


def _augment_clean_tensor(
    x,
    pad=4,
    hflip_prob=0.5,
):
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 4
    ):
        raise ValueError(
            "Expected BCHW clean tensor."
        )

    if pad > 0:
        padded = F.pad(
            x,
            (
                pad,
                pad,
                pad,
                pad,
            ),
            mode="reflect",
        )

        batch_size, _, h, w = x.shape
        crops = []

        for i in range(batch_size):
            top = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )

            left = int(
                torch.randint(
                    0,
                    2 * pad + 1,
                    (1,),
                    device=x.device,
                ).item()
            )

            crops.append(
                padded[
                    i : i + 1,
                    :,
                    top : top + h,
                    left : left + w,
                ]
            )

        x = torch.cat(
            crops,
            dim=0,
        )

    if hflip_prob > 0:
        mask = (
            torch.rand(
                x.shape[0],
                device=x.device,
            )
            < float(hflip_prob)
        )

        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(
                x[mask],
                dims=[3],
            )

    return x


def _apply_trigger(x, trigger):
    x = x.clone()

    trigger = trigger.to(
        device=x.device,
        dtype=x.dtype,
    )

    size = int(
        trigger.shape[-1]
    )

    x[
        :,
        :,
        -size:,
        -size:,
    ] = trigger.unsqueeze(0)

    return x


# ============================================================
# Frozen quantized classifier geometry
# ============================================================

@torch.no_grad()
def _extract_quantized_classifier(
    model,
    classifier_name,
):
    """
    Build geometry from the ACTUAL frozen quantized classifier,
    not from the FP checkpoint.
    """

    modules = dict(
        model.named_modules()
    )

    if classifier_name not in modules:
        raise RuntimeError(
            "Classifier module not found: {}".format(
                classifier_name
            )
        )

    classifier = modules[
        classifier_name
    ]

    if not hasattr(
        classifier,
        "weight",
    ):
        raise RuntimeError(
            "Classifier has no weight."
        )

    fq = getattr(
        classifier,
        "weight_fake_quant",
        None,
    )

    if fq is None:
        raise RuntimeError(
            "Classifier has no weight_fake_quant."
        )

    if getattr(
        fq,
        "structured_attack",
        False,
    ):
        raise RuntimeError(
            "Classifier structured field must remain OFF in V5."
        )

    weight = (
        fq(
            classifier.weight.detach()
        )
        .detach()
        .float()
    )

    if classifier.bias is None:
        bias = torch.zeros(
            weight.shape[0],
            device=weight.device,
            dtype=weight.dtype,
        )
    else:
        bias = (
            classifier.bias
            .detach()
            .float()
        )

    if weight.ndim != 2:
        raise RuntimeError(
            "V5 requires linear classifier weight [C, D]."
        )

    return classifier, weight, bias


@torch.no_grad()
def _build_decision_geometry(
    classifier_weight,
    classifier_bias,
    target,
    svd_tol,
):
    """
    Frozen classifier:
        z = W h + b

    Clean-equivalent set:
        E(h0) = {h : Wc h = Wc h0}

    Target decision region:
        Rt = {h : (wt-wj)^T h + (bt-bj) >= 0, all j != t}
    """

    weight = classifier_weight.float()
    bias = classifier_bias.float()

    num_classes, feature_dim = weight.shape

    if not (
        0 <= int(target) < num_classes
    ):
        raise ValueError(
            "Invalid target class."
        )

    centered = (
        weight
        - weight.mean(
            dim=0,
            keepdim=True,
        )
    )

    (
        _,
        singular_values,
        vh,
    ) = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    if singular_values.numel() == 0:
        raise RuntimeError(
            "Empty classifier SVD."
        )

    threshold = (
        float(svd_tol)
        * float(
            singular_values.max().item()
        )
    )

    rank = int(
        (
            singular_values
            > threshold
        )
        .sum()
        .item()
    )

    if rank <= 0:
        raise RuntimeError(
            "Classifier decision rank is zero."
        )

    # [D, r], orthonormal columns spanning decision row-space.
    basis = (
        vh[
            :rank
        ]
        .transpose(
            0,
            1,
        )
        .contiguous()
    )

    other_classes = [
        cls
        for cls in range(num_classes)
        if cls != int(target)
    ]

    target_normals = (
        weight[
            int(target)
        ]
        .unsqueeze(0)
        - weight[
            other_classes
        ]
    )

    target_offsets = (
        bias[
            int(target)
        ]
        .unsqueeze(0)
        - bias[
            other_classes
        ]
    )

    return {
        "weight": weight,
        "bias": bias,
        "basis": basis,
        "rank": int(rank),
        "null_dim": int(
            feature_dim - rank
        ),
        "singular_values": (
            singular_values.detach()
            .cpu()
            .tolist()
        ),
        "target_normals": (
            target_normals
        ),
        "target_offsets": (
            target_offsets
        ),
        "feature_dim": int(
            feature_dim
        ),
        "num_classes": int(
            num_classes
        ),
    }


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
    Strong check:
        model_logits == F.linear(tapped_feature, quantized_W, b)
    """

    logits, feature = (
        _teacher_forward(
            model,
            attack_records,
            tapper,
            batch,
            device,
        )
    )

    manual = F.linear(
        feature.float(),
        geometry["weight"],
        geometry["bias"],
    )

    max_abs_error = float(
        (
            manual
            - logits.float()
        )
        .abs()
        .max()
        .item()
    )

    print(
        "[V5] classifier geometry max-logit error:",
        max_abs_error,
    )

    if (
        max_abs_error
        > float(tolerance)
    ):
        raise RuntimeError(
            "Tapped feature / quantized classifier geometry "
            "does not reproduce model logits. max error = {}".format(
                max_abs_error
            )
        )

    return max_abs_error


# ============================================================
# Decision-set geometry
# ============================================================

def clean_equivalence_distance_sq(
    attacked_feature,
    benign_feature,
    decision_basis,
):
    """
    E(h0) = h0 + Null(Wc)

    With orthonormal decision basis B:
        d^2(h, E(h0))
        = ||(h-h0) B||^2
    """

    delta = (
        attacked_feature
        - benign_feature
    )

    coeff = (
        delta
        @ decision_basis
    )

    return (
        coeff.pow(2)
        .sum(dim=1)
    )


@torch.no_grad()
def project_to_target_region(
    feature,
    target_normals,
    target_offsets,
    cycles=100,
):
    """
    Euclidean projection onto target decision polyhedron
    via Dykstra projection over C-1 half-spaces.

    For target t beating class j:
        a_j^T h + beta_j >= 0
    """

    if feature.ndim != 2:
        raise ValueError(
            "Target projection expects [B, D] features."
        )

    normals = target_normals.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    offsets = target_offsets.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    batch_size, dim = feature.shape
    num_constraints = int(
        normals.shape[0]
    )

    y = (
        feature.detach()
        .clone()
    )

    corrections = torch.zeros(
        batch_size,
        num_constraints,
        dim,
        device=feature.device,
        dtype=feature.dtype,
    )

    norm_sq = (
        normals.pow(2)
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    for _ in range(
        int(cycles)
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

            normal = normals[j]

            score = (
                z @ normal
                + offsets[j]
            )

            step = (
                (-score)
                .clamp_min(0.0)
                / norm_sq[j]
            )

            projected = (
                z
                + step.unsqueeze(1)
                * normal.unsqueeze(0)
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

    return y


@torch.no_grad()
def target_region_max_violation(
    feature,
    target_normals,
    target_offsets,
):
    normals = target_normals.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    offsets = target_offsets.to(
        device=feature.device,
        dtype=feature.dtype,
    )

    scores = (
        feature
        @ normals.transpose(0, 1)
        + offsets.unsqueeze(0)
    )

    violation = (
        (-scores)
        .clamp_min(0.0)
    )

    return (
        violation.max(
            dim=1
        ).values
    )


def conditional_set_projection_loss(
    clean_feature,
    clean_reference,
    trigger_feature,
    geometry,
    projection_cycles,
):
    """
    ONE conditional set-distance objective.

    clean:
        target set = benign-equivalent affine set E(x)

    trigger:
        target set = frozen classifier target region Rt

    No CE / KL / target margin / target exemplars.
    """

    basis = geometry[
        "basis"
    ].to(
        device=clean_feature.device,
        dtype=clean_feature.dtype,
    )

    clean_d2 = (
        clean_equivalence_distance_sq(
            clean_feature,
            clean_reference,
            basis,
        )
    )

    target_projection = (
        project_to_target_region(
            trigger_feature.detach(),
            geometry["target_normals"],
            geometry["target_offsets"],
            cycles=int(
                projection_cycles
            ),
        )
    )

    trigger_d2 = (
        (
            trigger_feature
            - target_projection
        )
        .pow(2)
        .sum(dim=1)
    )

    rank = max(
        int(
            geometry["rank"]
        ),
        1,
    )

    # Both distances live in the same classifier decision geometry.
    # Concatenate them into one sample-level set-distance objective.
    all_d2 = torch.cat(
        [
            clean_d2,
            trigger_d2,
        ],
        dim=0,
    )

    total = (
        all_d2.mean()
        / float(rank)
    )

    return {
        "total": total,
        "clean_set_distance": (
            clean_d2.mean()
            / float(rank)
        ),
        "trigger_set_distance": (
            trigger_d2.mean()
            / float(rank)
        ),
    }


# ============================================================
# Diagnostics
# ============================================================

@torch.no_grad()
def _structured_stats(records):
    total_valid = 0
    total_flipped = 0
    total_params = 0
    layers = []

    for record in records:
        fq = record["fq"]

        valid = (
            fq.structured_valid_mask
            .bool()
        )

        flips = (
            fq.current_structured_flip_mask()
        )

        field = (
            fq.structured_field()
            .detach()
        )

        valid_count = int(
            valid.sum().item()
        )

        flip_count = int(
            flips.sum().item()
        )

        param_count = int(
            fq.field_u.numel()
            + fq.field_v.numel()
        )

        total_valid += valid_count
        total_flipped += flip_count
        total_params += param_count

        layers.append(
            {
                "name": record["name"],
                "valid": valid_count,
                "flipped": flip_count,
                "flip_ratio_valid": (
                    flip_count
                    / max(
                        valid_count,
                        1,
                    )
                ),
                "field_rank": int(
                    fq.structured_rank
                ),
                "field_params": (
                    param_count
                ),
                "field_abs_mean": float(
                    field.abs()
                    .mean()
                    .item()
                ),
                "field_abs_max": float(
                    field.abs()
                    .max()
                    .item()
                ),
            }
        )

    return {
        "total_valid": int(
            total_valid
        ),
        "total_flipped": int(
            total_flipped
        ),
        "flip_ratio_valid": (
            total_flipped
            / max(
                total_valid,
                1,
            )
        ),
        "field_parameters": int(
            total_params
        ),
        "layers": layers,
    }


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
    projection_cycles,
):
    basis = geometry[
        "basis"
    ].to(
        device=device
    )

    clean_total = 0
    clean_agree = 0
    clean_set_distance_sum = 0.0

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

        pseudo = (
            teacher_logits.argmax(
                dim=1
            )
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

        clean_d2 = (
            clean_equivalence_distance_sq(
                attacked_feature,
                teacher_feature,
                basis.to(
                    dtype=(
                        attacked_feature.dtype
                    )
                ),
            )
        )

        clean_set_distance_sum += float(
            clean_d2.sum().item()
        )

        clean_total += int(
            pseudo.numel()
        )

    bd_total = 0
    bd_hits = 0
    target_set_feasible = 0
    trigger_set_distance_sum = 0.0
    projection_violation_max = 0.0

    for batch in bd_data:
        batch = _to_device(
            batch,
            device,
        )

        (
            logits,
            feature,
        ) = _attacked_forward(
            model,
            tapper,
            batch,
            device,
        )

        pred = (
            logits.argmax(
                dim=1
            )
        )

        bd_hits += int(
            (
                pred
                == int(target)
            )
            .sum()
            .item()
        )

        current_violation = (
            target_region_max_violation(
                feature,
                geometry["target_normals"],
                geometry["target_offsets"],
            )
        )

        target_set_feasible += int(
            (
                current_violation
                <= 1e-6
            )
            .sum()
            .item()
        )

        projected = (
            project_to_target_region(
                feature,
                geometry["target_normals"],
                geometry["target_offsets"],
                cycles=int(
                    projection_cycles
                ),
            )
        )

        trigger_d2 = (
            (
                feature
                - projected
            )
            .pow(2)
            .sum(dim=1)
        )

        trigger_set_distance_sum += float(
            trigger_d2.sum().item()
        )

        projected_violation = (
            target_region_max_violation(
                projected,
                geometry["target_normals"],
                geometry["target_offsets"],
            )
        )

        projection_violation_max = max(
            projection_violation_max,
            float(
                projected_violation
                .max()
                .item()
            ),
        )

        bd_total += int(
            pred.numel()
        )

    rank = max(
        int(
            geometry["rank"]
        ),
        1,
    )

    return {
        "clean_teacher_agreement": (
            100.0
            * clean_agree
            / max(
                clean_total,
                1,
            )
        ),
        "clean_set_distance": (
            clean_set_distance_sum
            / max(
                clean_total,
                1,
            )
            / float(rank)
        ),
        "asr": (
            100.0
            * bd_hits
            / max(
                bd_total,
                1,
            )
        ),
        "target_set_feasible_rate": (
            100.0
            * target_set_feasible
            / max(
                bd_total,
                1,
            )
        ),
        "trigger_set_distance": (
            trigger_set_distance_sum
            / max(
                bd_total,
                1,
            )
            / float(rank)
        ),
        "projection_max_violation": float(
            projection_violation_max
        ),
        "clean_samples": int(
            clean_total
        ),
        "bd_samples": int(
            bd_total
        ),
    }


@torch.no_grad()
def _evaluate_cv_loader(
    model,
    loader,
    device,
):
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = (
            _extract_logits(
                model(images)
            )
            .argmax(
                dim=1
            )
        )

        correct += int(
            (
                pred
                == labels
            )
            .sum()
            .item()
        )

        total += int(
            labels.numel()
        )

    return (
        100.0
        * correct
        / max(
            total,
            1,
        )
    )


def _capture_field_state(records):
    return [
        {
            "u": (
                record["fq"]
                .field_u
                .detach()
                .cpu()
                .clone()
            ),
            "v": (
                record["fq"]
                .field_v
                .detach()
                .cpu()
                .clone()
            ),
        }
        for record in records
    ]


def _restore_field_state(
    records,
    state,
):
    with torch.no_grad():
        for record, value in zip(
            records,
            state,
        ):
            fq = record["fq"]

            fq.field_u.copy_(
                value["u"].to(
                    fq.field_u.device
                )
            )

            fq.field_v.copy_(
                value["v"].to(
                    fq.field_v.device
                )
            )


def _finalize_structured_rounding(
    model,
    records,
):
    with torch.no_grad():
        for record in records:
            module = record[
                "module"
            ]

            fq = record[
                "fq"
            ]

            if (
                hasattr(module, "bn")
                and hasattr(
                    module.bn,
                    "running_var",
                )
            ):
                factor = (
                    module.bn.weight
                    / torch.sqrt(
                        module.bn.running_var
                        + module.bn.eps
                    )
                ).reshape(
                    [-1]
                    + [1]
                    * (
                        module.weight.ndim
                        - 1
                    )
                )

                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data
                        * factor
                    )
                    / factor
                )

            else:
                module.weight.data.copy_(
                    fq.get_structured_hard_value(
                        module.weight.data
                    )
                )

            fq.structured_attack = False
            fq.adaround = False

            fq.field_u.requires_grad_(
                False
            )

            fq.field_v.requires_grad_(
                False
            )

    enable_quantization(
        model
    )

    model.eval()


def structured_decision_set_attack(
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
    V5 = Structured Rounding Field
         + Conditional Decision-Set Projection.

    Only the feature extractor receives malicious structured fields.
    The quantized final classifier remains benign and frozen.
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

    tapper = ExactFeatureTapper(
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

    field_params = []

    for record in attack_records:
        fq = record["fq"]

        fq.init_structured_attack(
            _effective_weight(
                record["module"]
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

    step0_stats = (
        _structured_stats(
            attack_records
        )
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
            record["fq"],
            "structured_attack",
            False,
        ):
            raise RuntimeError(
                "Frozen classifier unexpectedly has active structured field."
            )

    trigger_spec = (
        extract_trigger_spec(
            trigger_loader
        )
    )

    trigger = (
        trigger_spec[
            "trigger_normalized"
        ]
        .to(device)
    )

    print(
        "[V5] trigger_size:",
        trigger_spec[
            "trigger_size"
        ],
    )

    target = int(
        config.bd_target
    )

    (
        classifier,
        classifier_weight,
        classifier_bias,
    ) = _extract_quantized_classifier(
        model,
        str(
            config.classifier_module
        ),
    )

    geometry = (
        _build_decision_geometry(
            classifier_weight,
            classifier_bias,
            target=target,
            svd_tol=float(
                config.decision_svd_tol
            ),
        )
    )

    print(
        "[V5] classifier geometry:",
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
            "singular_values": geometry[
                "singular_values"
            ],
        },
    )

    if (
        geometry["rank"]
        > (
            geometry[
                "num_classes"
            ]
            - 1
        )
    ):
        raise RuntimeError(
            "Decision rank exceeds C-1. Geometry construction is wrong."
        )

    first_batch = _to_device(
        cali_data[0],
        device,
    )

    geometry_logit_error = (
        _verify_classifier_geometry(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            batch=first_batch,
            device=device,
            geometry=geometry,
            tolerance=float(
                config.geometry_logit_tolerance
            ),
        )
    )

    projection_cycles = int(
        config.projection_cycles
    )

    baseline = (
        _calibration_metrics(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            clean_data=cali_data,
            bd_data=cali_data_bd,
            target=target,
            device=device,
            geometry=geometry,
            projection_cycles=projection_cycles,
        )
    )

    print(
        "[V5] step-0 calibration:",
        baseline,
    )

    print(
        "[V5] step-0 structured stats:",
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
            "clean_set_distance"
        ]
        > 1e-10
    ):
        raise RuntimeError(
            "Step-0 clean decision-set distance is not zero."
        )

    if (
        baseline[
            "projection_max_violation"
        ]
        > float(
            config.projection_tolerance
        )
    ):
        raise RuntimeError(
            "Dykstra projection did not reach target region. "
            "Increase projection_cycles."
        )

    if abs(
        baseline["asr"]
        - baseline[
            "target_set_feasible_rate"
        ]
    ) > 0.5:
        raise RuntimeError(
            "ASR != target-region feasibility. "
            "Classifier geometry does not match inference."
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

    best_state = (
        _capture_field_state(
            attack_records
        )
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
            clean0 + asr0,
            1e-12,
        )
    )

    best_step = 0
    history = []

    for step in range(
        1,
        steps + 1,
    ):
        batch = _to_device(
            cali_data[
                int(
                    np.random.randint(
                        len(cali_data)
                    )
                )
            ],
            device,
        )

        clean_view = (
            _augment_clean_tensor(
                batch,
                pad=int(
                    config.aug_pad
                ),
                hflip_prob=float(
                    config.hflip_prob
                ),
            )
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

        pseudo = (
            teacher_logits.argmax(
                dim=1
            )
        )

        source_mask = (
            pseudo != target
        )

        if not source_mask.any():
            continue

        triggered_view = (
            _apply_trigger(
                clean_view[
                    source_mask
                ],
                trigger,
            )
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

        losses = (
            conditional_set_projection_loss(
                clean_feature=clean_feature,
                clean_reference=teacher_feature,
                trigger_feature=trigger_feature,
                geometry=geometry,
                projection_cycles=projection_cycles,
            )
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
            or step == steps
        ):
            metrics = (
                _calibration_metrics(
                    model=model,
                    attack_records=attack_records,
                    tapper=tapper,
                    clean_data=cali_data,
                    bd_data=cali_data_bd,
                    target=target,
                    device=device,
                    geometry=geometry,
                    projection_cycles=projection_cycles,
                )
            )

            stats = (
                _structured_stats(
                    attack_records
                )
            )

            clean_agreement = (
                metrics[
                    "clean_teacher_agreement"
                ]
            )

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
                "step": int(step),
                "train_total_loss": float(
                    loss.detach()
                    .item()
                ),
                "train_clean_set_distance": float(
                    losses[
                        "clean_set_distance"
                    ]
                    .detach()
                    .item()
                ),
                "train_trigger_set_distance": float(
                    losses[
                        "trigger_set_distance"
                    ]
                    .detach()
                    .item()
                ),
                "score": float(score),
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
                and test_loader is not None
                and test_loader_bd is not None
                and (
                    step
                    % test_interval
                    == 0
                    or step == steps
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

            history.append(row)

            print(
                "[V5]",
                row,
            )

            if score > best_score:
                best_score = float(
                    score
                )

                best_step = int(
                    step
                )

                best_state = (
                    _capture_field_state(
                        attack_records
                    )
                )

                print(
                    "[V5] new best:",
                    {
                        "step": best_step,
                        "score": best_score,
                        "clean_agreement": clean_agreement,
                        "asr": asr,
                        "clean_set_distance": (
                            metrics[
                                "clean_set_distance"
                            ]
                        ),
                        "trigger_set_distance": (
                            metrics[
                                "trigger_set_distance"
                            ]
                        ),
                    },
                )

    _restore_field_state(
        attack_records,
        best_state,
    )

    restored = (
        _calibration_metrics(
            model=model,
            attack_records=attack_records,
            tapper=tapper,
            clean_data=cali_data,
            bd_data=cali_data_bd,
            target=target,
            device=device,
            geometry=geometry,
            projection_cycles=projection_cycles,
        )
    )

    stats = (
        _structured_stats(
            attack_records
        )
    )

    test = {}

    if (
        test_loader is not None
        and test_loader_bd is not None
    ):
        test = {
            "ca": (
                _evaluate_cv_loader(
                    model,
                    test_loader,
                    device,
                )
            ),
            "asr": (
                _evaluate_cv_loader(
                    model,
                    test_loader_bd,
                    device,
                )
            ),
        }

    print(
        "[V5] restored best step:",
        best_step,
    )

    print(
        "[V5] restored calibration:",
        restored,
    )

    print(
        "[V5] restored test:",
        test,
    )

    print(
        "[V5] restored structured stats:",
        stats,
    )

    result = {
        "method": (
            "structured_rounding_field_"
            "conditional_decision_set_projection"
        ),
        "semantic_module": str(
            config.semantic_module
        ),
        "classifier_module": str(
            config.classifier_module
        ),
        "frozen_field_modules": [
            record["name"]
            for record in frozen_records
        ],
        "classifier_geometry": {
            "feature_dim": int(
                geometry["feature_dim"]
            ),
            "num_classes": int(
                geometry["num_classes"]
            ),
            "decision_rank": int(
                geometry["rank"]
            ),
            "null_dim": int(
                geometry["null_dim"]
            ),
            "singular_values": (
                geometry[
                    "singular_values"
                ]
            ),
            "geometry_logit_error": float(
                geometry_logit_error
            ),
        },
        "baseline_calibration": baseline,
        "history": history,
        "best_step": int(
            best_step
        ),
        "best_score": float(
            best_score
        ),
        "best_calibration": restored,
        "best_test": test,
        "structured_stats": stats,
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
            "projection_cycles": int(
                config.projection_cycles
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
            "[V5] saved:",
            output_json,
        )

    return model, result
