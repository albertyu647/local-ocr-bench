"""Score every run in out/ against the hand-checked references in ref/gt/.

Metrics per page, then averaged:
  CER      character error rate (edit distance / reference length) — penalises wrong reading order too.
  word F1  bag-of-words precision/recall F1 — ignores reading order, so layout choices don't dominate.
  diagram  share of the figure labels in ref/diagram_labels.json found in the output (9 diagram pages).
Model-specific wrappers (bbox tags, JSON, HTML tables, markdown image links) are stripped first so a
model isn't punished for its output format — only for the text it read.
"""
import json, re, unicodedata, statistics as st
from collections import Counter
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
GT = {p.stem: p.read_text() for p in sorted((ROOT / "ref/gt").glob("p*.txt"))}
LABELS = json.loads((ROOT / "ref/diagram_labels.json").read_text())

RUNS = {  # run dir -> (display name, how to extract text)
    "apple_vision": ("Apple Vision (macOS built-in)", "plain"),
    "mirage": ("Mirage (Cactus, 51M)", "mirage"),
    "minicpm_page": ("MiniCPM-OCR 1.0 – whole page", "plain"),
    "minicpm_sdk": ("MiniCPM-OCR 1.0 – official SDK pipeline", "md"),
    "ovis_md": ("OvisOCR3 – markdown mode", "md"),
    "ovis_fig": ("OvisOCR3 – figure/caption mode", "ovis_json"),
    "unlimited": ("Unlimited-OCR-2", "unlimited"),
    "pc3060_md_fla": ("PC 3060Ti: Ovis markdown, 1 page", "md"),
    "pc3060_batch13_document_markdown": ("PC 3060Ti: Ovis markdown, batch 13", "md"),
    "pc3060_batch1_figure_captioning": ("PC 3060Ti: Ovis figure, 1 page", "ovis_json"),
    "pc3060_batch13_figure_captioning": ("PC 3060Ti: Ovis figure, batch 13", "ovis_json"),
    "pc3060_vllm_document_markdown": ("PC 3060Ti vLLM: Ovis markdown", "md"),
    "pc3060_vllm_figure_captioning": ("PC 3060Ti vLLM: Ovis figure", "ovis_json"),
}


def extract(run, kind, page):
    d = ROOT / "out" / run
    f = d / f"{page}.md" if kind == "md" and (d / f"{page}.md").exists() else d / f"{page}.txt"
    raw = f.read_text() if f.exists() else ""
    extra = ""  # generated captions / mermaid: counted for diagram recall, not for text accuracy
    if kind == "mirage":
        try:
            raw = "\n".join(t.get("string", "") for t in json.loads(raw).get("text", []))
        except Exception:
            raw = ""
    elif kind == "ovis_json":
        try:
            m = re.search(r"\[.*\]", raw, re.S)
            items = json.loads(m.group(0)) if m else []
            parts = []
            for it in items:
                c = it.get("content", "")
                caps = re.findall(r"<caption>(.*?)</caption>|<mermaid>(.*?)</mermaid>", c, re.S)
                extra += " ".join(a + " " + b for a, b in caps) + "\n"
                c = re.sub(r"<caption>.*?</caption>|<mermaid>.*?</mermaid>", " ", c, flags=re.S)
                parts.append(c)
            raw = "\n".join(parts)
        except Exception:
            pass
    elif kind == "unlimited":
        raw = re.sub(r"<BLOCK>[^\[]*\[[^\]]*\]<FORK>", "\n", raw)
        raw = re.sub(r"<PAGE>|<LEVEL\d>|</?[A-Z_]+>", " ", raw)
    return raw, extra


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)          # markdown images
    s = re.sub(r"<img[^>]*>", " ", s)
    s = s.replace("\\n", "\n")
    s = re.sub(r"<[^>]+>", " ", s)                       # html tags (tables)
    s = re.sub(r"\$\^\{\\circledR\}\$|\\circledR", "®", s)
    s = re.sub(r"[$*#_`]", " ", s)                       # md / latex markup
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)          # re-join hyphenated line breaks
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": " ", "–": " ",
                                   "☐": " ", "□": " ", "…": "..."}))
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def words(s):
    return Counter(w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s))


def label_found(label, text_plain):
    l = re.sub(r"[^a-z0-9 ]", " ", norm(label)); l = re.sub(r"\s+", " ", l).strip()
    return l in text_plain or fuzz.partial_ratio(l, text_plain) >= 90


def timing(run):
    d = ROOT / "out" / run
    if (d / "_timing.json").exists():
        t = json.loads((d / "_timing.json").read_text())
        secs = [p["seconds"] for p in t["pages"]]
        return st.mean(secs), t.get("peak_mem_gb")
    if (d / "_timing.txt").exists():
        return st.mean(float(l.split()[1]) for l in (d / "_timing.txt").read_text().split("\n") if l.strip()), None
    if (d / "_total_time.txt").exists():
        return float((d / "_total_time.txt").read_text().split("per page")[1]), None
    return None, None


results, per_page = [], {}
for run, (name, kind) in RUNS.items():
    if not (ROOT / "out" / run).exists():
        continue
    cers, f1s, recalls, diag_f1 = [], [], [], []
    for page, gt in GT.items():
        raw, extra = extract(run, kind, page)
        o, g = norm(raw), norm(gt)
        cer = min(1.0, Levenshtein.distance(o, g) / max(1, len(g)))
        wo, wg = words(o), words(g)
        tp = sum((wo & wg).values())
        p = tp / max(1, sum(wo.values())); r = tp / max(1, sum(wg.values()))
        f1 = 0 if tp == 0 else 2 * p * r / (p + r)
        cers.append(cer); f1s.append(f1)
        row = dict(cer=round(cer, 3), word_f1=round(f1, 3))
        if page in LABELS:
            plain = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", norm(raw + " " + extra)))
            hit = [l for l in LABELS[page] if label_found(l, plain)]
            rec = len(hit) / len(LABELS[page]); recalls.append(rec); diag_f1.append(f1)
            row["diagram_recall"] = round(rec, 3)
            row["missed_labels"] = [l for l in LABELS[page] if l not in hit]
        per_page.setdefault(run, {})[page] = row
    sec, mem = timing(run)
    text_pages = [p for p in GT if p not in LABELS]
    results.append(dict(
        run=run, name=name,
        sec_per_page=round(sec, 2) if sec is not None else None,
        pages_per_min=round(60 / sec, 1) if sec else None,
        peak_mem_gb=mem,
        cer=round(st.mean(cers), 3),
        word_f1=round(st.mean(f1s), 3),
        word_f1_text_pages=round(st.mean(per_page[run][p]["word_f1"] for p in text_pages), 3),
        word_f1_diagram_pages=round(st.mean(diag_f1), 3),
        diagram_label_recall=round(st.mean(recalls), 3),
    ))

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/scores.json").write_text(json.dumps(dict(summary=results, per_page=per_page), indent=1))
hdr = f"{'model':44} {'s/page':>7} {'CER':>6} {'wordF1':>7} {'F1 text':>8} {'F1 diag':>8} {'diag recall':>11} {'mem GB':>7}"
print(hdr); print("-" * len(hdr))
for r in results:
    print(f"{r['name']:44} {r['sec_per_page'] or '':>7} {r['cer']:>6} {r['word_f1']:>7} {r['word_f1_text_pages']:>8} "
          f"{r['word_f1_diagram_pages']:>8} {r['diagram_label_recall']:>11} {r['peak_mem_gb'] or '':>7}")
