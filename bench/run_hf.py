"""OvisOCR3 on CUDA via transformers. Same pages, prompts, warm-up and timing as run_mlx.py.
Usage: python run_hf.py <run> <prompt_file> [max_tokens]"""
import json, sys, time, glob
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
run, prompt_file = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
MODEL = ROOT / "models" / "OvisOCR3"
if not (MODEL / "config.json").exists():
    from huggingface_hub import snapshot_download
    snapshot_download("ATH-MaaS/OvisOCR3", local_dir=str(MODEL))
prompt_text = Path(prompt_file).read_text(encoding="utf-8")
out = ROOT / "out" / run; out.mkdir(parents=True, exist_ok=True)
pages = sorted(glob.glob(str(ROOT / "pages" / "p*.png")))

t = time.time()
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=448*448, max_pixels=2880*2880)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
load_s = time.time() - t
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def run_page(img):
    inputs = proc(text=[prompt], images=[Image.open(img).convert("RGB")], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    gen = ids[0, inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True), inputs["input_ids"].shape[1], len(gen)

run_page(pages[0]); torch.cuda.reset_peak_memory_stats()
rows = []
for img in pages:
    t0 = time.time(); text, ptok, gtok = run_page(img); dt = time.time() - t0
    (out / f"{Path(img).stem}.txt").write_text(text, encoding="utf-8")
    rows.append(dict(page=Path(img).stem, seconds=round(dt, 2), prompt_tokens=ptok, gen_tokens=gtok, gen_tps=round(gtok/dt, 1)))
    print(json.dumps(rows[-1]), flush=True)
s = dict(run=run, device=torch.cuda.get_device_name(0), load_seconds=round(load_s, 1),
         peak_mem_gb=round(torch.cuda.max_memory_allocated()/1e9, 2), pages=rows)
(out / "_timing.json").write_text(json.dumps(s, indent=1))
print("TOTAL", round(sum(r["seconds"] for r in rows), 1), "s  peak", s["peak_mem_gb"], "GB")
