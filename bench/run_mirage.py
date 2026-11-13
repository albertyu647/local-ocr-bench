"""Run Cactus Mirage on each page. Mirage has no OCR-only call; describe() returns scene text incl. OCR."""
import json, time, glob, sys
from pathlib import Path
import cactus
ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "out" / "mirage"; out.mkdir(parents=True, exist_ok=True)
t = time.time(); m = cactus.Mirage(weights=str(ROOT / "models/mirage/mirage.cact")); load_s = time.time() - t
pages = sorted(glob.glob(str(ROOT / "pages/p*.png")))
m.describe(pages[0])  # warm-up
rows = []
for p in pages:
    t0 = time.time(); r = m.describe(p); dt = time.time() - t0
    txt = r if isinstance(r, str) else json.dumps(r, default=str, indent=1)
    (out / f"{Path(p).stem}.txt").write_text(txt)
    rows.append(dict(page=Path(p).stem, seconds=round(dt, 3)))
    print(rows[-1], flush=True)
(out / "_timing.json").write_text(json.dumps(dict(run="mirage", load_seconds=round(load_s, 2), pages=rows), indent=1))
