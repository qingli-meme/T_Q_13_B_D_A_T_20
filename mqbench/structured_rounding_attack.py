import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mqbench.fake_quantize.structured_rounding_quantizer import StructuredRoundFakeQuantize
from mqbench.utils.state import enable_quantization


def _to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, dict): return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list): return [_to_device(v, device) for v in x]
    if isinstance(x, tuple): return tuple(_to_device(v, device) for v in x)
    return x


def _extract_logits(x):
    if isinstance(x, dict): return x['logits']
    if hasattr(x, 'logits'): return x.logits
    return x


def _forward_logits(model, batch, device):
    batch = _to_device(batch, device)
    return _extract_logits(model(**batch) if isinstance(batch, dict) else model(batch))


class FeatureTapper:
    def __init__(self, model, names):
        modules = dict(model.named_modules())
        missing = [x for x in names if x not in modules]
        if missing: raise RuntimeError('Missing feature stages: {}'.format(missing))
        self.stage_names, self.cache, self.handles = list(names), {}, []
        graph_targets = []
        if hasattr(model, 'graph'):
            graph_targets = [str(node.target) for node in model.graph.nodes
                             if node.op == 'call_module']
        for name in names:
            # FX tracing flattens Sequential execution: the container remains
            # in named_modules(), but forward calls its children directly.
            # Tap the final executed module inside the requested stage.
            candidate_indices = [i for i, target in enumerate(graph_targets)
                                 if target.startswith(name + '.') and
                                 target in modules]
            actual_name = name
            if candidate_indices:
                last_index = candidate_indices[-1]
                # The residual add/ReLU is represented by call_function;
                # its following activation fake-quant module is the first
                # hookable node carrying the true stage output.
                if last_index + 1 < len(graph_targets):
                    actual_name = graph_targets[last_index + 1]
                else:
                    actual_name = graph_targets[last_index]
            if actual_name not in modules:
                raise RuntimeError('Cannot resolve feature tap {}'.format(name))
            print('[V3] feature tap:', name, '->', actual_name)
            self.handles.append(
                modules[actual_name].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor): self.cache[name] = output
            elif isinstance(output, (tuple, list)):
                values = [x for x in output if isinstance(x, torch.Tensor)]
                if not values: raise RuntimeError('No tensor at {}'.format(name))
                self.cache[name] = values[0]
            else: raise RuntimeError('Unsupported output at {}'.format(name))
        return hook

    def clear(self): self.cache = {}
    def snapshot(self):
        missing = [x for x in self.stage_names if x not in self.cache]
        if missing: raise RuntimeError('Hooks did not fire: {}'.format(missing))
        return {x: self.cache[x] for x in self.stage_names}
    def close(self):
        for handle in self.handles: handle.remove()


def _pool_normalize(x):
    if x.ndim > 2: x = x.mean(dim=tuple(range(2, x.ndim)))
    return F.normalize(x, p=2, dim=1, eps=1e-8)


def _attacked_forward(model, tapper, batch, device):
    tapper.clear()
    logits = _forward_logits(model, batch, device)
    return logits, tapper.snapshot()


def _effective_weight(module):
    weight = module.weight
    if hasattr(module, 'bn') and hasattr(module.bn, 'running_var'):
        factor = module.bn.weight / torch.sqrt(module.bn.running_var + module.bn.eps)
        return weight * factor.reshape([-1] + [1] * (weight.ndim - 1))
    return weight


def _collect_rounding_modules(model):
    records = []
    for name, module in model.named_modules():
        fq = getattr(module, 'weight_fake_quant', None)
        if isinstance(fq, StructuredRoundFakeQuantize) and getattr(module, 'weight', None) is not None:
            records.append({'name': name, 'module': module, 'fq': fq})
    if not records: raise RuntimeError('No StructuredRoundFakeQuantize modules found.')
    return records


def _set_force_base(records, enabled):
    for rec in records: rec['fq'].set_structured_force_base(enabled)


@torch.no_grad()
def _teacher_forward(model, records, tapper, batch, device):
    _set_force_base(records, True)
    try:
        tapper.clear()
        logits = _forward_logits(model, batch, device).detach()
        features = {k: v.detach() for k, v in tapper.snapshot().items()}
    finally: _set_force_base(records, False)
    return logits, features


def _unwrap_dataset(dataset):
    seen = set()
    while hasattr(dataset, 'dataset') and id(dataset) not in seen:
        seen.add(id(dataset)); dataset = dataset.dataset
    return dataset


def _flatten_transforms(transform):
    if transform is None: return []
    if hasattr(transform, 'transforms'):
        return sum((_flatten_transforms(x) for x in transform.transforms), [])
    return [transform]


def extract_trigger_spec(loader):
    ops = _flatten_transforms(getattr(_unwrap_dataset(loader.dataset), 'transform', None))
    trigger = mean = std = None
    for op in ops:
        if trigger is None and isinstance(getattr(op, 'trigger', None), torch.Tensor):
            trigger = op.trigger.detach().float().clone()
        if hasattr(op, 'mean') and hasattr(op, 'std'):
            try:
                m, s = torch.as_tensor(op.mean).float(), torch.as_tensor(op.std).float()
                if m.numel() == s.numel() == 3:
                    mean, std = m.reshape(3, 1, 1), s.reshape(3, 1, 1)
            except Exception: pass
    if trigger is None or mean is None: raise RuntimeError('Could not extract trigger/Normalize.')
    if trigger.ndim == 4: trigger = trigger[0]
    if trigger.ndim != 3: raise ValueError('Expected CHW trigger.')
    return {'trigger_normalized': (trigger - mean) / std,
            'trigger_size': int(trigger.shape[-1])}


def _augment_clean_tensor(x, pad=4, hflip_prob=0.5):
    if not isinstance(x, torch.Tensor) or x.ndim != 4: raise ValueError('Expected BCHW.')
    if pad > 0:
        padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        b, _, h, w = x.shape
        crops = []
        for i in range(b):
            top = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
            left = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
            crops.append(padded[i:i+1, :, top:top+h, left:left+w])
        x = torch.cat(crops)
    mask = torch.rand(x.shape[0], device=x.device) < float(hflip_prob)
    if mask.any():
        x = x.clone(); x[mask] = torch.flip(x[mask], dims=[3])
    return x


def _apply_trigger(x, trigger):
    x = x.clone(); trigger = trigger.to(device=x.device, dtype=x.dtype)
    size = int(trigger.shape[-1])
    x[:, :, -size:, -size:] = trigger.unsqueeze(0)
    return x


@torch.no_grad()
def build_pathway_prototypes(model, records, tapper, cali_data, device,
                             num_classes, min_proto_count=2):
    sums, counts, initialized = {}, torch.zeros(num_classes, dtype=torch.long, device=device), False
    for batch in cali_data:
        logits, features = _teacher_forward(model, records, tapper, _to_device(batch, device), device)
        pseudo = logits.argmax(1); counts.index_add_(0, pseudo, torch.ones_like(pseudo))
        pooled = {k: _pool_normalize(features[k]) for k in tapper.stage_names}
        if not initialized:
            sums = {k: torch.zeros(num_classes, v.shape[1], device=device, dtype=v.dtype)
                    for k, v in pooled.items()}; initialized = True
        for k in tapper.stage_names: sums[k].index_add_(0, pseudo, pooled[k])
    bad = torch.where(counts < int(min_proto_count))[0]
    if bad.numel(): raise RuntimeError('Too few prototypes: {} counts={}'.format(bad.tolist(), counts.tolist()))
    return ({k: F.normalize(v, p=2, dim=1, eps=1e-8).detach() for k, v in sums.items()},
            counts.cpu().tolist())


def _prototype_scores(feature, prototype, temperature):
    return (_pool_normalize(feature) @ F.normalize(prototype, p=2, dim=1).t()) / max(float(temperature), 1e-6)


def pathway_routing_loss(clean_logits, clean_features, clean_labels,
                         trigger_logits, trigger_features, target, prototypes,
                         stage_names, proto_temperature):
    clean, trigger = [], []
    targets = torch.full((trigger_logits.shape[0],), int(target), device=trigger_logits.device, dtype=torch.long)
    for name in stage_names:
        clean.append(F.cross_entropy(_prototype_scores(clean_features[name], prototypes[name], proto_temperature), clean_labels))
        trigger.append(F.cross_entropy(_prototype_scores(trigger_features[name], prototypes[name], proto_temperature), targets))
    clean.append(F.cross_entropy(clean_logits, clean_labels))
    trigger.append(F.cross_entropy(trigger_logits, targets))
    c, t = torch.stack(clean).mean(), torch.stack(trigger).mean()
    return {'total': 0.5 * (c + t), 'clean_route': c, 'trigger_route': t,
            'clean_stage_losses': [float(x.detach()) for x in clean],
            'trigger_stage_losses': [float(x.detach()) for x in trigger]}


@torch.no_grad()
def _structured_stats(records):
    total_valid = total_flipped = params = 0; layers = []
    for rec in records:
        fq = rec['fq']; valid = fq.structured_valid_mask.bool()
        flips, field = fq.current_structured_flip_mask(), fq.structured_field().detach()
        vn, fn, pn = int(valid.sum()), int(flips.sum()), int(fq.field_u.numel()+fq.field_v.numel())
        total_valid += vn; total_flipped += fn; params += pn
        layers.append({'name': rec['name'], 'valid': vn, 'flipped': fn,
                       'flip_ratio_valid': fn/max(vn, 1), 'field_rank': fq.structured_rank,
                       'field_params': pn, 'field_abs_mean': float(field.abs().mean()),
                       'field_abs_max': float(field.abs().max())})
    return {'total_valid': total_valid, 'total_flipped': total_flipped,
            'flip_ratio_valid': total_flipped/max(total_valid, 1),
            'field_parameters': params, 'layers': layers}


@torch.no_grad()
def _calibration_metrics(model, records, tapper, prototypes, clean_data,
                         bd_data, target, device, temperature):
    clean_total = clean_agree = bd_total = bd_hits = 0
    clean_route = {k: 0 for k in tapper.stage_names}; trigger_route = dict(clean_route)
    for batch in clean_data:
        batch = _to_device(batch, device)
        teacher, _ = _teacher_forward(model, records, tapper, batch, device)
        pseudo = teacher.argmax(1); logits, features = _attacked_forward(model, tapper, batch, device)
        clean_agree += int((logits.argmax(1) == pseudo).sum()); clean_total += pseudo.numel()
        for k in tapper.stage_names:
            clean_route[k] += int((_prototype_scores(features[k], prototypes[k], temperature).argmax(1) == pseudo).sum())
    for batch in bd_data:
        logits, features = _attacked_forward(model, tapper, _to_device(batch, device), device)
        pred = logits.argmax(1); bd_hits += int((pred == target).sum()); bd_total += pred.numel()
        for k in tapper.stage_names:
            trigger_route[k] += int((_prototype_scores(features[k], prototypes[k], temperature).argmax(1) == target).sum())
    return {'clean_teacher_agreement': 100*clean_agree/max(clean_total,1),
            'asr': 100*bd_hits/max(bd_total,1),
            'clean_stage_route_acc': {k:100*v/max(clean_total,1) for k,v in clean_route.items()},
            'trigger_stage_route_acc': {k:100*v/max(bd_total,1) for k,v in trigger_route.items()},
            'clean_samples': int(clean_total), 'bd_samples': int(bd_total)}


@torch.no_grad()
def _evaluate_cv_loader(model, loader, device):
    correct = total = 0
    for images, labels in loader:
        pred = _extract_logits(model(images.to(device))).argmax(1); labels = labels.to(device)
        correct += int((pred == labels).sum()); total += labels.numel()
    return 100*correct/max(total,1)


def _capture_field_state(records):
    return [{'u':r['fq'].field_u.detach().cpu().clone(), 'v':r['fq'].field_v.detach().cpu().clone()} for r in records]


def _restore_field_state(records, state):
    with torch.no_grad():
        for rec, value in zip(records, state):
            rec['fq'].field_u.copy_(value['u'].to(rec['fq'].field_u.device))
            rec['fq'].field_v.copy_(value['v'].to(rec['fq'].field_v.device))


def _finalize_structured_rounding(model, records):
    with torch.no_grad():
        for rec in records:
            module, fq = rec['module'], rec['fq']
            if hasattr(module, 'bn') and hasattr(module.bn, 'running_var'):
                sf = (module.bn.weight / torch.sqrt(module.bn.running_var + module.bn.eps)).reshape([-1]+[1]*(module.weight.ndim-1))
                module.weight.data.copy_(fq.get_structured_hard_value(module.weight.data*sf)/sf)
            else: module.weight.data.copy_(fq.get_structured_hard_value(module.weight.data))
            fq.structured_attack = False; fq.adaround = False
            fq.field_u.requires_grad_(False); fq.field_v.requires_grad_(False)
    enable_quantization(model); model.eval()


def structured_rounding_attack(model, cali_data, cali_data_bd, trigger_loader,
                               config, output_json=None, test_loader=None,
                               test_loader_bd=None):
    device = next(model.parameters()).device; model.eval(); enable_quantization(model)
    for p in model.parameters(): p.requires_grad_(False)
    stages = list(config.stages); tapper = FeatureTapper(model, stages)
    records = _collect_rounding_modules(model); params = []
    for rec in records:
        rec['fq'].init_structured_attack(_effective_weight(rec['module']).detach(),
            int(config.field_rank), float(config.delta_bound),
            float(config.round_temperature), float(config.field_init_std))
        params.extend([rec['fq'].field_u, rec['fq'].field_v])
    step0 = _structured_stats(records)
    if step0['total_flipped'] != 0: raise RuntimeError('Step-0 changed rounding.')
    spec = extract_trigger_spec(trigger_loader); trigger = spec['trigger_normalized'].to(device)
    print('[V3] trigger_size:', spec['trigger_size'])
    prototypes, counts = build_pathway_prototypes(model, records, tapper, cali_data,
        device, int(config.num_classes), int(config.min_proto_count))
    print('[V3] benign pseudo-label counts:', counts)
    baseline = _calibration_metrics(model, records, tapper, prototypes, cali_data,
        cali_data_bd, int(config.bd_target), device, float(config.proto_temperature))
    print('[V3] step-0 calibration:', baseline); print('[V3] step-0 structured stats:', step0)
    if baseline['clean_teacher_agreement'] < 99.999: raise RuntimeError('Step-0 model != benign PTQ.')
    optimizer = torch.optim.Adam(params, lr=float(config.lr), weight_decay=float(config.weight_decay))
    steps = int(config.steps); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=float(config.lr)*0.1)
    target = int(config.bd_target); interval = int(config.eval_interval)
    best_state = _capture_field_state(records); best_step = 0
    best_score = .5*(baseline['clean_teacher_agreement']+baseline['asr']); history=[]
    for step in range(1, steps+1):
        batch = _to_device(cali_data[int(np.random.randint(len(cali_data)))], device)
        view = _augment_clean_tensor(batch, int(config.aug_pad), float(config.hflip_prob))
        teacher, _ = _teacher_forward(model, records, tapper, view, device); pseudo = teacher.argmax(1)
        mask = pseudo != target
        if not mask.any(): continue
        triggered = _apply_trigger(view[mask], trigger); optimizer.zero_grad()
        clean_logits, clean_features = _attacked_forward(model, tapper, view, device)
        trig_logits, trig_features = _attacked_forward(model, tapper, triggered, device)
        losses = pathway_routing_loss(clean_logits, clean_features, pseudo, trig_logits,
            trig_features, target, prototypes, stages, float(config.proto_temperature))
        losses['total'].backward(); torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip))
        optimizer.step(); scheduler.step()
        if step == 1 or step % interval == 0 or step == steps:
            metrics = _calibration_metrics(model, records, tapper, prototypes, cali_data,
                cali_data_bd, target, device, float(config.proto_temperature)); stats = _structured_stats(records)
            score=.5*(metrics['clean_teacher_agreement']+metrics['asr'])
            row={'step':step,'train_total_loss':float(losses['total']),'train_clean_route_loss':float(losses['clean_route']),
                 'train_trigger_route_loss':float(losses['trigger_route']),'train_clean_stage_losses':losses['clean_stage_losses'],
                 'train_trigger_stage_losses':losses['trigger_stage_losses'],'score':score,**metrics,
                 'flip_ratio_valid':stats['flip_ratio_valid'],'total_flipped':stats['total_flipped'],
                 'field_parameters':stats['field_parameters']}
            if bool(config.diagnostic_test_eval) and (step % int(config.diagnostic_test_interval)==0 or step==steps):
                row['diagnostic_test_ca']=_evaluate_cv_loader(model,test_loader,device)
                row['diagnostic_test_asr']=_evaluate_cv_loader(model,test_loader_bd,device)
            history.append(row); print('[V3]',row)
            if score > best_score:
                best_score, best_step, best_state = score, step, _capture_field_state(records)
                print('[V3] new best:',{'step':step,'score':score})
    _restore_field_state(records,best_state)
    restored=_calibration_metrics(model,records,tapper,prototypes,cali_data,cali_data_bd,target,device,float(config.proto_temperature))
    stats=_structured_stats(records); test={}
    if test_loader is not None: test={'ca':_evaluate_cv_loader(model,test_loader,device),'asr':_evaluate_cv_loader(model,test_loader_bd,device)}
    print('[V3] restored best step:',best_step); print('[V3] restored calibration:',restored)
    print('[V3] restored test:',test); print('[V3] restored structured stats:',stats)
    result={'method':'structured_rounding_field_pathway_routing','baseline_calibration':baseline,
            'prototype_counts':counts,'history':history,'best_step':best_step,'best_score':best_score,
            'best_calibration':restored,'best_test':test,'structured_stats':stats,'trigger_size':spec['trigger_size']}
    _finalize_structured_rounding(model,records); tapper.close()
    if output_json:
        os.makedirs(os.path.dirname(output_json),exist_ok=True)
        with open(output_json,'w') as f: json.dump(result,f,indent=2)
        print('[V3] saved:',output_json)
    return model,result
