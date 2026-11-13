"""OCR every page image in a folder with OvisOCR3 (figure mode) on vLLM, 40 pages per batch.
Writes <out>/<page>.json with the raw layout JSON, finish reason and token count; skips pages already done.
Usage: python ocr_pdf.py <pages_dir> <out_dir>      (render pages first: pdftoppm -r 200 -png doc.pdf pages/p)"""
import json, time, glob, sys
from pathlib import Path

def main():
    from PIL import Image
    from vllm import LLM, SamplingParams
    ROOT = Path(__file__).resolve().parent.parent
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    pages = [p for p in sorted(glob.glob(str(Path(sys.argv[1]) / "p-*.png"))) if not (out / (Path(p).stem + ".json")).exists()]
    llm = LLM(model=str(ROOT / "models" / "OvisOCR3"), gpu_memory_utilization=0.85, max_model_len=16384,
              gdn_prefill_backend="triton", limit_mm_per_prompt={"image": 1})
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=8192, temperature=0.0)
    text = (ROOT / "ref/ovis_prompts/figure_captioning.txt").read_text(encoding="utf-8")
    prompt = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.time(); done = 0
    for i in range(0, len(pages), 40):
        chunk = pages[i:i + 40]
        reqs = [{"prompt": prompt, "multi_modal_data": {"image": [Image.open(p).convert("RGB")]},
                 "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 448*448, "max_pixels": 2880*2880}}} for p in chunk]
        for p, o in zip(chunk, llm.generate(reqs, sp, use_tqdm=False)):
            c = o.outputs[0]
            (out / (Path(p).stem + ".json")).write_text(json.dumps(dict(
                page=Path(p).stem, finish=c.finish_reason, gen_tokens=len(c.token_ids), text=c.text)), encoding="utf-8")
        done += len(chunk)
        print(f"PROGRESS {done}/{len(pages)} {time.time() - t0:.1f}s", flush=True)
    print(f"DONE {done} pages in {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
