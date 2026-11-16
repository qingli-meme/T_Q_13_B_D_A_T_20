#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mqbench.no_target_direct_rounding_attack import paired_behavior_loss

teacher = torch.tensor([
    [3.0, 1.0, -1.0],
    [0.2, 2.1, -0.3],
])
clean = teacher.detach().clone().requires_grad_(True)
trigger = torch.tensor([
    [0.1, 0.2, -0.1],
    [0.2, 0.0, 0.1],
], requires_grad=True)

losses = paired_behavior_loss(
    clean_logits=clean,
    teacher_logits=teacher,
    trigger_logits=trigger,
    target=0,
)

assert abs(float(losses["clean_kl"].detach())) < 1.0e-6
assert float(losses["trigger_ce"].detach()) > 0.0
assert torch.isfinite(losses["total"])

losses["total"].backward()
assert clean.grad is not None
assert trigger.grad is not None
assert torch.isfinite(clean.grad).all()
assert torch.isfinite(trigger.grad).all()
assert float(clean.grad.abs().max()) < 1.0e-6

print("objective algebra: PASS")
print("clean teacher fixed point: PASS")
print("trigger gradient: PASS")
