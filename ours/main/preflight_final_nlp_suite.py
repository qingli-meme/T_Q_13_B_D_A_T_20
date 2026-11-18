import json
from pathlib import Path

THIS = Path(__file__).resolve().parent
required = [
    "bert+sst-2.pth",
    "bert+imdb.pth",
    "bert+twitter.pth",
    "bert+boolq.pth",
    "bert+rte.pth",
    "bert+cb.pth",
]
missing = []
for name in required:
    p = THIS / "model" / name
    print(("OK   " if p.exists() else "MISS "), p)
    if not p.exists():
        missing.append(str(p))

manifest = THIS / "final_nlp_manifest.json"
print(("OK   " if manifest.exists() else "MISS "), manifest)
if not manifest.exists():
    missing.append(str(manifest))

cfg = THIS / "configs" / "final_nlp_fixed_trigger"
print(("OK   " if cfg.exists() else "MISS "), cfg)
if not cfg.exists():
    missing.append(str(cfg))

if missing:
    print("\nPreflight FAILED.")
    print(json.dumps({"missing": missing}, indent=2))
    raise SystemExit(1)

print("\nPreflight PASS.")
print('Fixed NLP trigger is provided by QuRA loader: "Kidding me!"')
print("12 main experiments are ready.")
