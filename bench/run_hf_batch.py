"""Batched OvisOCR3 throughput on CUDA: all 13 pages in batches of B. Usage: run_batch.py <B> <prompt_file>"""
import sys, time, glob, json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
ROOT = Path(__file__).resolve().parent.parent; MODEL = ROOT / "models" / "OvisOCR3"
B = int(sys.argv[1]); prompt_text = Path(sys.argv[2]).read_text(encoding="utf-8")
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=448*448, max_pixels=2880*2880)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
pages = sorted(glob.glob(str(ROOT / "pages" / "p*.png")))
out = ROOT / "out" / f"batch{B}_{Path(sys.argv[2]).stem}"; out.mkdir(parents=True, exist_ok=True)
def go(batch):
    inp = proc(text=[prompt]*len(batch), images=[Image.open(p).convert("RGB") for p in batch], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        ids = model.generate(**inp, max_new_tokens=8192, do_sample=False)
    torch.cuda.synchronize()
    return [proc.decode(r[inp["input_ids"].shape[1]:], skip_special_tokens=True) for r in ids]
go(pages[:2]); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for i in range(0, len(pages), B):
    for p, txt in zip(pages[i:i+B], go(pages[i:i+B])):
        (out / f"{Path(p).stem}.txt").write_text(txt, encoding="utf-8")
dt = time.time() - t0
print(json.dumps(dict(B=B, total_s=round(dt,1), s_per_page=round(dt/len(pages),2), peak_gb=round(torch.cuda.max_memory_allocated()/1e9,2))))
