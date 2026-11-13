"""Build compare.html: each test page's scan beside every Mac model's output, with per-page scores.
Run: envs/mlx/bin/python scripts/make_compare.py   (re-run after score.py if scores change)"""
import json, re, html
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SC = json.loads((ROOT / "results/scores.json").read_text())
LABELS = json.loads((ROOT / "ref/diagram_labels.json").read_text())
PAGES = sorted(p.stem for p in (ROOT / "ref/gt").glob("p*.txt"))

MODELS = [  # run, display name, kind
    ("ovis_fig", "OvisOCR3 · figure mode", "ovis_json"),
    ("ovis_md", "OvisOCR3 · markdown mode", "md"),
    ("minicpm_page", "MiniCPM-OCR 1.0 · whole page", "plain"),
    ("minicpm_sdk", "MiniCPM-OCR 1.0 · SDK pipeline", "md"),
    ("unlimited", "Unlimited-OCR-2", "unlimited"),
    ("mirage", "Mirage", "mirage"),
    ("apple_vision", "Apple Vision (baseline)", "plain"),
]
summary = {r["run"]: r for r in SC["summary"]}


def page_times(run):
    d = OUT / run
    if (d / "_timing.json").exists():
        return {p["page"]: p["seconds"] for p in json.loads((d / "_timing.json").read_text())["pages"]}
    if (d / "_timing.txt").exists():
        return {l.split()[0]: float(l.split()[1]) for l in (d / "_timing.txt").read_text().splitlines() if l.strip()}
    return {}


def raw_text(run, page):
    d = OUT / run
    for ext in (".md", ".txt"):
        if (d / f"{page}{ext}").exists():
            return (d / f"{page}{ext}").read_text()
    return ""


def crop(b, note=""):
    return f'<div class="crop" data-bbox="{",".join(map(str, b))}"></div>' + (f'<div class="note">{note}</div>' if note else "")


def render(run, kind, raw):
    """Return list of blocks: {"t": "md"|"html"|"pre"|"mermaid", "v": str}."""
    if kind == "plain":
        return [{"t": "pre", "v": raw}]
    if kind == "md":
        s = raw
        if run == "minicpm_sdk":
            s = re.sub(r"\]\((imgs/[^)]+)\)", r"](out/minicpm_sdk/\1)", s)
        s = re.sub(r'<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg"\s*/?>',
                   lambda m: crop([int(x) for x in m.groups()], "image region (model output only a crop box, no text)"), s)
        return [{"t": "md", "v": s}]
    if kind == "ovis_json":
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except Exception:
            return [{"t": "pre", "v": raw}]
        out = []
        for it in items:
            c, cat = it.get("content", ""), it.get("category", "")
            caps = re.findall(r"<caption>(.*?)</caption>", c, re.S)
            mers = re.findall(r"<mermaid>(.*?)</mermaid>", c, re.S)
            rest = re.sub(r"<caption>.*?</caption>|<mermaid>.*?</mermaid>", "", c, flags=re.S).strip()
            if caps or mers or cat in ("image", "chart"):
                out.append({"t": "html", "v": crop(it.get("bbox", [0, 0, 1000, 1000])) +
                            "".join(f'<div class="caption"><b>Caption:</b> {html.escape(x.strip())}</div>' for x in caps)})
                for x in mers:
                    out.append({"t": "mermaid", "v": x.strip()})
            if rest:
                out.append({"t": "md", "v": rest})
        return out
    if kind == "unlimited":
        out = []
        for cat, bb, content in re.findall(r"<BLOCK>(\w+)\[([^\]]*)\]<FORK>(.*?)(?=<BLOCK>|$)", raw, re.S):
            content = re.sub(r"</?[A-Z_]+>|<LEVEL\d>", "", content).strip()
            if cat in ("image", "chart", "figure") and not content:
                out.append({"t": "html", "v": crop([int(float(x)) for x in bb.split(",")], f"{cat} block: returned empty")})
            elif content:
                out.append({"t": "md", "v": content})
        return out or [{"t": "pre", "v": raw}]
    if kind == "mirage":
        try:
            d = json.loads(raw)
        except Exception:
            return [{"t": "pre", "v": raw}]
        lines = "\n".join(t.get("string", "") for t in sorted(d.get("text", []), key=lambda t: (t["bbox"][1], t["bbox"][0])))
        meta = f'<div class="note">Scene: {html.escape(json.dumps(d.get("scene", {})))} · {len(d.get("objects", []))} objects detected<br>{html.escape(d.get("reasoning", ""))}</div>'
        return [{"t": "html", "v": meta}, {"t": "pre", "v": lines}]


data = {"pages": [], "models": []}
for run, name, kind in MODELS:
    s = summary.get(run, {})
    data["models"].append(dict(run=run, name=name, sec=s.get("sec_per_page"), cer=s.get("cer"),
                               f1=s.get("word_f1"), recall=s.get("diagram_label_recall")))
times = {run: page_times(run) for run, _, _ in MODELS}
for page in PAGES:
    w, h = Image.open(ROOT / "pages" / f"{page}.png").size
    entry = dict(id=page, num=int(page[1:]), w=w, h=h, diagram=page in LABELS,
                 labels=LABELS.get(page, []), gt=(ROOT / "ref/gt" / f"{page}.txt").read_text(), outputs={})
    for run, name, kind in MODELS:
        raw = raw_text(run, page)
        sc = SC["per_page"].get(run, {}).get(page, {})
        entry["outputs"][run] = dict(raw=raw, blocks=render(run, kind, raw), sec=times[run].get(page),
                                     cer=sc.get("cer"), f1=sc.get("word_f1"),
                                     recall=sc.get("diagram_recall"), missed=sc.get("missed_labels", []))
    data["pages"].append(entry)

tpl = (ROOT / "scripts/compare_template.html").read_text()
(ROOT / "compare.html").write_text(tpl.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/")))
print("wrote", ROOT / "compare.html")
