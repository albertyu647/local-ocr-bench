"""OvisOCR3 on vLLM (Windows build). Same pages, prompts and settings as the official vLLM example.
Times (a) one page at a time and (b) all 13 pages in one batch, for markdown and figure modes.
Usage: python run_vllm.py"""
import json, time, glob, sys
from pathlib import Path

def main():
    from PIL import Image
    from vllm import LLM, SamplingParams
    ROOT = Path(__file__).resolve().parent.parent
    llm = LLM(model=str(ROOT / "models" / "OvisOCR3"), gpu_memory_utilization=0.85, max_model_len=16384,
              gdn_prefill_backend="triton", limit_mm_per_prompt={"image": 1})
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=8192, temperature=0.0)
    pages = sorted(glob.glob(str(ROOT / "pages" / "p*.png")))
    imgs = {p: Image.open(p).convert("RGB") for p in pages}
    def req(p, prompt):
        return {"prompt": prompt, "multi_modal_data": {"image": [imgs[p]]},
                "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 448*448, "max_pixels": 2880*2880}}}
    results = {}
    for mode in ["document_markdown", "figure_captioning"]:
        text = (ROOT / "ref/ovis_prompts" / f"{mode}.txt").read_text(encoding="utf-8")
        prompt = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        llm.generate([req(pages[0], prompt)], sp, use_tqdm=False)  # warm-up
        single = []
        for p in pages:
            t0 = time.time(); llm.generate([req(p, prompt)], sp, use_tqdm=False); single.append(round(time.time() - t0, 2))
        t0 = time.time(); outs = llm.generate([req(p, prompt) for p in pages], sp, use_tqdm=False); batch = time.time() - t0
        out = ROOT / "out" / f"vllm_{mode}"; out.mkdir(parents=True, exist_ok=True)
        gen = 0
        for p, o in zip(pages, outs):
            (out / f"{Path(p).stem}.txt").write_text(o.outputs[0].text, encoding="utf-8"); gen += len(o.outputs[0].token_ids)
        results[mode] = dict(single_s_per_page=round(sum(single) / len(single), 2), single=single,
                             batch13_total_s=round(batch, 1), batch13_s_per_page=round(batch / len(pages), 2),
                             batch_gen_tokens=gen)
        print("RESULT", mode, json.dumps(results[mode]), flush=True)
    (ROOT / "out" / "vllm_timing.json").write_text(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()
