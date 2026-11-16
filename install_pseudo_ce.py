#!/usr/bin/env python3
from pathlib import Path
import ast, shutil
ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'ours/main/main.py'; RUNNER=ROOT/'mqbench/pseudo_label_ce_direct_rounding_attack.py'
if not MAIN.exists(): raise SystemExit('Missing ours/main/main.py')
if not RUNNER.exists(): raise SystemExit('Missing pseudo-label CE runner')
r=RUNNER.read_text(); ast.parse(r)
for token in ['kl_div','target_depth','decision_svd','dykstra','project_to_target','target_core','contrastive']:
    if token in r.lower(): raise SystemExit('Forbidden method token: '+token)
main=MAIN.read_text()
if 'import copy\n' not in main:
    main=main.replace('import os\n','import os\nimport copy\n',1)
old="""    if hasattr(config, 'quantize'):\n        model = get_quantize_model(model, config)\n"""
new="""    if hasattr(config, 'quantize'):\n        fp_reference_model = None\n        if config.quantize.quantize_type == 'pseudo_label_ce_direct_rounding':\n            fp_reference_model = copy.deepcopy(model)\n            fp_reference_model.to(device)\n            fp_reference_model.eval()\n            for parameter in fp_reference_model.parameters():\n                parameter.requires_grad_(False)\n\n        model = get_quantize_model(model, config)\n"""
if 'fp_reference_model = None' not in main:
    if old not in main: raise SystemExit('Quantization anchor not found')
    main=main.replace(old,new,1)
marker="config.quantize.quantize_type == 'pseudo_label_ce_direct_rounding'"
# marker occurs above in pre-quantization preservation; branch marker is elif.
branch_marker="        elif config.quantize.quantize_type == 'pseudo_label_ce_direct_rounding':"
if branch_marker not in main:
    src="        elif config.quantize.quantize_type == 'no_target_direct_rounding':"
    nxt="        elif config.quantize.quantize_type == 'minimum_transport_depth_ablation':"
    if src not in main or nxt not in main: raise SystemExit('no-target branch not found')
    st=main.index(src); en=main.index(nxt,st); block=main[st:en]
    block=block.replace(src,branch_marker,1)
    block=block.replace("'No-target direct rounding supports --type bd only.'","'Pseudo-label CE direct rounding supports --type bd only.'")
    block=block.replace('from mqbench.no_target_direct_rounding_attack import (\n                no_target_direct_rounding_attack,\n            )','from mqbench.pseudo_label_ce_direct_rounding_attack import (\n                pseudo_label_ce_direct_rounding_attack,\n            )',1)
    anchor="""            attack_cfg = config.quantize.attack\n\n            from mqbench.utils.state import (\n"""
    insert="""            attack_cfg = config.quantize.attack\n\n            if fp_reference_model is None:\n                raise RuntimeError('Full-precision reference model was not preserved.')\n            pseudo_labels = []\n            fp_reference_model.eval()\n            with torch.no_grad():\n                for clean_batch in cali_data:\n                    if isinstance(clean_batch, dict):\n                        fp_output = fp_reference_model(**to_device(clean_batch, device))\n                    else:\n                        fp_output = fp_reference_model(to_device(clean_batch, device))\n                    if isinstance(fp_output, dict):\n                        fp_logits = fp_output['logits']\n                    elif hasattr(fp_output, 'logits'):\n                        fp_logits = fp_output.logits\n                    else:\n                        fp_logits = fp_output\n                    pseudo_labels.append(fp_logits.argmax(dim=1).detach().cpu())\n            pseudo_hist = Counter(torch.cat(pseudo_labels).tolist())\n            print('[PSEUDO-CE] FP pseudo-label histogram:', dict(sorted(pseudo_hist.items())))\n            del fp_reference_model\n            fp_reference_model = None\n            if torch.cuda.is_available():\n                torch.cuda.empty_cache()\n\n            from mqbench.utils.state import (\n"""
    if anchor not in block: raise SystemExit('attack_cfg anchor not found')
    block=block.replace(anchor,insert,1)
    block=block.replace("f'no_target_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'","f'pseudo_label_ce_direct_rounding_{variant_name}_{args.model}_{args.dataset}.json'",1)
    block=block.replace("f'no_target_direct_rounding_{variant_name}_t{attack_cfg.bd_target}.pth'","f'pseudo_label_ce_direct_rounding_{variant_name}_t{attack_cfg.bd_target}.pth'",1)
    oldcall="""                model, attack_info = no_target_direct_rounding_attack(\n                    model=model,\n                    cali_data=cali_data,\n                    cali_data_bd=cali_data_bd,\n                    trigger_loader=trigger_loader,\n"""
    newcall="""                model, attack_info = pseudo_label_ce_direct_rounding_attack(\n                    model=model,\n                    cali_data=cali_data,\n                    pseudo_labels=pseudo_labels,\n                    cali_data_bd=cali_data_bd,\n                    trigger_loader=trigger_loader,\n"""
    if oldcall not in block: raise SystemExit('runner call not found')
    block=block.replace(oldcall,newcall,1)
    block=block.replace("'Saved no-target direct-rounding model:'","'Saved pseudo-label CE direct-rounding model:'")
    block=block.replace("'\\n========== No-Target Direct Legal Rounding: ' + variant_name + ' =========='","'\\n========== FP Pseudo-Label CE + Direct Legal Rounding: ' + variant_name + ' =========='")
    main=main[:en]+block+main[en:]
backup=MAIN.with_suffix(MAIN.suffix+'.pre_pseudo_ce.bak')
if not backup.exists(): shutil.copy2(MAIN,backup)
ast.parse(main); MAIN.write_text(main)
for tok in ['fp_reference_model = copy.deepcopy(model)',branch_marker,'pseudo_label_ce_direct_rounding_attack(','pseudo_labels=pseudo_labels']:
    if tok not in main: raise SystemExit('main.py missing '+tok)
print('[install] PASS')
