#!/usr/bin/env python3
from pathlib import Path
import ast, shutil

ROOT=Path(__file__).resolve().parent
P=ROOT/"mqbench/reconstruction_ce_direct_rounding_attack.py"
if not P.exists():
    raise SystemExit("Missing reconstruction_ce_direct_rounding_attack.py")

s=P.read_text(encoding="utf-8")

pairs=[
("""def reconstruction_backdoor_loss(
    clean_logits,
    fp_logits,
    trigger_logits,
    target,
):
""",
"""def reconstruction_backdoor_loss(
    clean_logits,
    fp_logits,
    trigger_logits,
    target,
    backdoor_weight=1.0,
):
"""),
("    total = rec_loss + backdoor_loss\n",
 "    total = rec_loss + float(backdoor_weight) * backdoor_loss\n"),
("""    target = int(config.bd_target)

    # Keep backward compatibility with the previous same_scope run:
""",
"""    target = int(config.bd_target)
    backdoor_weight = (
        float(config.backdoor_weight)
        if hasattr(config, 'backdoor_weight')
        else 1.0
    )
    if backdoor_weight <= 0:
        raise ValueError('backdoor_weight must be positive')

    # Keep backward compatibility with the previous same_scope run:
"""),
("""            target=target,
        )

        loss = losses['total']
""",
"""            target=target,
            backdoor_weight=backdoor_weight,
        )

        loss = losses['total']
"""),
("""            'total': 'L_rec + L_bd',
            'backdoor_weight': 1.0,
""",
"""            'total': 'L_rec + alpha * L_bd',
            'backdoor_weight': float(backdoor_weight),
"""),
]

for old,new in pairs:
    if old in s:
        s=s.replace(old,new,1)
    elif new not in s:
        raise SystemExit("Patch target not found:\n"+old[:120])

old="""        'backdoor_ce': backdoor_loss,
        'trigger_target_probability': trigger_prob.mean(),
"""
new="""        'backdoor_ce': backdoor_loss,
        'weighted_backdoor_ce': float(backdoor_weight) * backdoor_loss,
        'trigger_target_probability': trigger_prob.mean(),
"""
if old in s:
    s=s.replace(old,new,1)

old="""                'train_backdoor_ce': float(
                    losses['backdoor_ce'].detach().item()
                ),
                'train_trigger_target_probability': float(
"""
new="""                'train_backdoor_ce': float(
                    losses['backdoor_ce'].detach().item()
                ),
                'train_weighted_backdoor_ce': float(
                    losses['weighted_backdoor_ce'].detach().item()
                ),
                'backdoor_weight': float(backdoor_weight),
                'train_trigger_target_probability': float(
"""
if old in s:
    s=s.replace(old,new,1)

ast.parse(s)
bak=P.with_suffix(P.suffix+".pre_alpha_sweep.bak")
if not bak.exists():
    shutil.copy2(P,bak)
P.write_text(s,encoding="utf-8")
print("installed alpha sweep support; old configs default to alpha=1")
