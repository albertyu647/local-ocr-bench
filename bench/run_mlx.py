"""Run one OCR model (MLX, 8-bit) over every page image and record output + timing.

Usage: python run_mlx.py <run_name> <model_dir> <prompt_file_or_text> [max_tokens]
Writes out/<run_name>/<page>.txt and out/<run_name>/_timing.json.
One untimed warm-up page first so model load / Metal kernel compilation isn't counted.
"""
import json, sys, time, glob, os
from pathlib import Path
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
run, model_dir, prompt_arg = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
prompt_text = Path(prompt_arg).read_text() if os.path.isfile(prompt_arg) else prompt_arg

out = ROOT / "out" / run
out.mkdir(parents=True, exist_ok=True)
pages = sorted(glob.glob(str(ROOT / "pages" / "p*.png")))

t = time.time()
model, processor = load(model_dir)
config = load_config(model_dir)
load_s = time.time() - t
prompt = apply_chat_template(processor, config, prompt_text, num_images=1)


def run_page(img):
    return generate(model, processor, prompt, image=[img], max_tokens=max_tokens,
                    temperature=0.0, verbose=False)


run_page(pages[0])  # warm-up, untimed
mx.reset_peak_memory()

rows = []
for img in pages:
    name = Path(img).stem
    t0 = time.time()
    r = run_page(img)
    dt = time.time() - t0
    (out / f"{name}.txt").write_text(r.text)
    rows.append(dict(page=name, seconds=round(dt, 2), prompt_tokens=r.prompt_tokens,
                     gen_tokens=r.generation_tokens, gen_tps=round(r.generation_tps, 1),
                     prompt_tps=round(r.prompt_tps, 1), finish=r.finish_reason))
    print(json.dumps(rows[-1]), flush=True)

summary = dict(run=run, model_dir=model_dir, load_seconds=round(load_s, 1),
               peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2), pages=rows)
(out / "_timing.json").write_text(json.dumps(summary, indent=1))
print("TOTAL", round(sum(r["seconds"] for r in rows), 1), "s  peak", summary["peak_mem_gb"], "GB")
