import json
from pathlib import Path

THIS_DIR=Path(__file__).resolve().parent
manifest=json.loads((THIS_DIR/'ptq_init_main_cv_manifest.json').read_text())
required=sorted({(x['model'],x['dataset']) for x in manifest})
missing=[]
print('Required checkpoints:')
for model,dataset in required:
    p=THIS_DIR/'model'/f'{model}+{dataset}.pth'
    ok=p.exists()
    print(('OK   ' if ok else 'MISS '), p)
    if not ok: missing.append(str(p))

tiny=THIS_DIR/'data'/'tiny-imagenet-200'
print(('OK   ' if tiny.exists() else 'MISS '), tiny, '(Tiny-ImageNet root)')
if not tiny.exists(): missing.append(str(tiny))

print('\nExperiment count:', len(manifest))
print('Models:', sorted({x['model'] for x in manifest}))
print('Datasets:', sorted({x['dataset'] for x in manifest}))
print('Quantization:', sorted({(x['weight_bits'],x['activation_bits']) for x in manifest}))
if missing:
    print('\nMissing prerequisites:')
    for x in missing: print(' -',x)
    raise SystemExit(1)
print('\nPreflight PASS')
