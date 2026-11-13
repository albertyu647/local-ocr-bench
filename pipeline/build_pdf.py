"""Assemble OCR output into one formatted HTML file (then printed to PDF by Chrome).

Input : <work>/ocr/p-NNN.json  (OvisOCR3 figure mode: JSON list of {bbox 0-1000, category, content})
        <source.pdf>           (figures are cropped from it at 300 dpi)
Output: <work>/out.html, <work>/figs/*, <work>/build_log.json
Run: python pipeline/build_pdf.py <source.pdf> <work_dir> "<Title>" [--pages 21,23,...]
Then: chrome --headless=new --no-pdf-header-footer --print-to-pdf=out.pdf file://<work_dir>/out.html
"""
import json, re, html, sys
from collections import Counter
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
SRC, WORK, DOC_TITLE = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
OCR = WORK / "ocr"
FIGS = WORK / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

DROP = {"header", "footer", "number", "header_image", "footer_image", "seal", "aside_text"}
FIGURE = {"image", "chart", "figure"}
TITLE = {"doc_title"}
HEADING = {"paragraph_title"}
CAPTION = {"figure_title", "table_title", "chart_title"}
SMALL = {"footnote", "vision_footnote", "reference", "reference_content"}


def parse(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


LATEX = {r"\times": "×", r"\div": "÷", r"\pm": "±", r"\cdot": "·", r"\%": "%", r"\$": "＄", r"\&": "&",
         r"\circledR": "®", r"\copyright": "©", r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←",
         r"\geq": "≥", r"\leq": "≤", r"\approx": "≈", r"\neq": "≠", r"\quad": " ", r"\qquad": "  ",
         r"\left": "", r"\right": "", r"\,": " ", r"\;": " ", r"\!": "", r"\cdots": "…", r"\ldots": "…"}


def delatex(m):
    t = m.group(1) if m.lastindex else m.group(0)
    t = re.sub(r"\\(?:begin|end)\{[a-z*]+\}", "", t)          # aligned/array environments
    t = t.replace("\\\\", "\n").replace("&", "")               # row breaks, alignment marks
    t = re.sub(r"\\(?=\d)", "＄", t)                             # "\10/hr" = $10/hr (escaped dollar lost its $)
    t = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\s*(\\times|=|\+)\s*", r" \1 ", t)
    for k in sorted(LATEX, key=len, reverse=True):
        t = t.replace(k, LATEX[k])
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
        t = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit|operatorname)\{([^{}]*)\}", r"\1", t)
        t = re.sub(r"\^\{([^{}]*)\}", lambda m: "\x01" + m.group(1) + "\x02", t)
        t = re.sub(r"_\{([^{}]*)\}", lambda m: "\x03" + m.group(1) + "\x04", t)
    t = re.sub(r"\^(\w)", lambda m: "\x01" + m.group(1) + "\x02", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t).replace("{", "").replace("}", "")
    t = re.sub(r"\s+\x01", "\x01", re.sub(r"/\s+", "/", t))
    return re.sub(r"[ \t]{2,}", " ", t)


def inline(s):
    """OCR text -> safe HTML: LaTeX math turned into plain symbols, light markdown."""
    s = re.sub(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", lambda m: delatex(re.match(r"(.*)", m.group(1) or m.group(2), re.S)), s, flags=re.S)
    s = re.sub(r"(?<!\\)\$(?!\d)([^$]{1,200}?)(?<!\\)\$", delatex, s)
    s = s.replace("\\$", "$").replace("\\%", "%").replace("＄", "$")
    s = html.escape(s, quote=False)
    s = s.replace("\x01", "<sup>").replace("\x02", "</sup>").replace("\x03", "<sub>").replace("\x04", "</sub>")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s.replace("\n", "<br>")


def clean_table(t):
    t = t.replace("\\n", "<br>").replace("\n", "<br>")
    t = re.sub(r"(?<!\\)\$(?!\d)([^$<]{1,200}?)(?<!\\)\$", delatex, t)
    t = t.replace("\\$", "$").replace("\\%", "%").replace("＄", "$")
    t = t.replace("\x01", "<sup>").replace("\x02", "</sup>").replace("\x03", "<sub>").replace("\x04", "</sub>")
    t = re.sub(r"<(/?)(table|thead|tbody|tr|td|th)(\s[^>]*)?>", lambda m: f"<{m.group(1)}{m.group(2)}{m.group(3) or ''}>", t)
    return re.sub(r"<(?!/?(table|thead|tbody|tr|td|th|br|b|i|sup|sub)\b)[^>]+>", "", t)  # keep table markup only


def ends_open(s):
    return not re.search(r"[.!?:;\"”’')\]]\s*$", s.strip())


def main():
    only = None
    if "--pages" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--pages") + 1].split(",")}
    doc = fitz.open(SRC)
    files = sorted(OCR.glob("p-*.json"))
    cats, log, body = Counter(), dict(unparsed=[], figures=0, tables=0), []
    pending_caption = None
    last_p = None  # index into body of the last paragraph, for joining across pages

    for f in files:
        pno = int(f.stem.split("-")[1])
        if only and pno not in only:
            continue
        rec = json.loads(f.read_text())
        items = parse(rec["text"])
        body.append(f'<a id="pdf{pno}" class="pdfp"></a>')
        if items is None:
            log["unparsed"].append(pno)
            body.append(f'<p class="warn">[PDF page {pno}: OCR output could not be parsed]</p>')
            last_p = None
            continue
        printed = next((it["content"].strip() for it in items if it.get("category") == "number"
                        and re.fullmatch(r"[0-9ivxlcdm]+", it.get("content", "").strip(), re.I)), None)
        if printed:
            body.append(f'<span class="pg" title="original page {html.escape(printed)}">{html.escape(printed)}</span>')
        first_text_on_page = True
        # margin icons: attach each to the text block whose vertical span is nearest the icon's centre
        icon_for = {}
        texts = [(k, t) for k, t in enumerate(items) if t.get("category") == "text"]
        for it in items:
            if it.get("category") in FIGURE:
                x1, y1, x2, y2 = it["bbox"]
                if (x2 - x1) * (y2 - y1) / 1e6 < 0.01 and texts:
                    cy = (y1 + y2) / 2
                    k = min(texts, key=lambda kt: 0 if kt[1]["bbox"][1] - 8 <= cy <= kt[1]["bbox"][3] + 8
                            else min(abs(cy - kt[1]["bbox"][1]), abs(cy - kt[1]["bbox"][3])))[0]
                    icon_for.setdefault(k, []).append(it)
        for k_item, it in enumerate(items):
            cat, c = it.get("category", "text"), it.get("content", "")
            cats[cat] += 1
            c = re.sub(r"<caption>.*?</caption>|<mermaid>.*?</mermaid>", "", c, flags=re.S).strip()
            if cat in DROP:
                continue
            if cat in FIGURE:
                x1, y1, x2, y2 = it["bbox"]
                r = doc[pno - 1].rect
                area = (x2 - x1) * (y2 - y1) / 1e6
                if area > 0.85:  # whole-page image: skip blank/black scans, keep real ones (e.g. ad pages)
                    # ink = share of pixels much darker than the paper, in the page centre (ignores gutter shadow)
                    cr = fitz.Rect(r.width * .15, r.height * .1, r.width * .85, r.height * .9)
                    px = sorted(doc[pno - 1].get_pixmap(dpi=60, colorspace=fitz.csGRAY, clip=cr).samples)
                    med = px[len(px) // 2]
                    ink = sum(v < med - 60 for v in px) / len(px)
                    if ink < 0.002:
                        log.setdefault("blank_pages_dropped", []).append(pno); continue
                if area < 0.01:  # margin icon: rendered with the text block it sits beside (icon_for)
                    continue
                pad = 4  # 0.4% padding so strokes at the edge aren't clipped
                clip = fitz.Rect(max(0, x1 - pad) / 1000 * r.width, max(0, y1 - pad) / 1000 * r.height,
                                 min(1000, x2 + pad) / 1000 * r.width, min(1000, y2 + pad) / 1000 * r.height)
                name = f"p{pno:03d}_{x1}_{y1}.jpg"
                if not (FIGS / name).exists():
                    doc[pno - 1].get_pixmap(dpi=300, clip=clip).save(FIGS / name, jpg_quality=88)
                cap = f"<figcaption>{inline(pending_caption)}</figcaption>" if pending_caption else ""
                frac = (x2 - x1) / 1000
                body.append(f'<figure><img src="figs/{name}" style="width:{min(100, max(30, frac * 118)):.0f}%">{cap}</figure>')
                pending_caption = None; log["figures"] += 1; last_p = None
                continue
            if pending_caption and cat not in CAPTION:
                body.append(f'<p class="figtitle">{inline(pending_caption)}</p>')  # caption with no figure (e.g. table)
                pending_caption = None
            if cat in CAPTION:
                pending_caption = c
                continue
            if cat == "table":
                body.append(f'<div class="tbl">{clean_table(c)}</div>'); log["tables"] += 1; last_p = None
                continue
            if not c:
                continue
            if cat in TITLE or cat in HEADING:
                c = re.sub(r"^#+\s*", "", c)
            if cat in TITLE:
                body.append(f"<h1>{inline(c)}</h1>"); last_p = None
            elif cat in HEADING:
                body.append(f"<h2>{inline(c)}</h2>"); last_p = None
            elif cat == "display_formula":
                body.append(f'<p class="formula">{inline("$" + c.strip("$ ") + "$")}</p>'); last_p = None
            elif cat in SMALL:
                body.append(f'<p class="small">{inline(c)}</p>'); last_p = None
            else:
                # join a paragraph that was split by a page break
                if first_text_on_page and last_p is not None and k_item not in icon_for \
                        and ends_open(body[last_p][3:-4]) and re.match(r"[a-z(—,;]", c):
                    prev = body[last_p][:-4]
                    sep = "" if prev.endswith("-") and c[:1].islower() else " "
                    if sep == "":
                        prev = prev[:-1]
                    body[last_p] = prev + sep + inline(c) + "</p>"
                    log.setdefault("joined_across_pages", 0); log["joined_across_pages"] += 1
                else:
                    icon = ""
                    for ic in icon_for.get(k_item, []):
                        x1, y1, x2, y2 = ic["bbox"]; r = doc[pno - 1].rect
                        name = f"p{pno:03d}_{x1}_{y1}.png"
                        if not (FIGS / name).exists():
                            doc[pno - 1].get_pixmap(dpi=300, clip=fitz.Rect(x1 / 1000 * r.width, y1 / 1000 * r.height,
                                                    x2 / 1000 * r.width, y2 / 1000 * r.height)).save(FIGS / name)
                        icon += f'<img class="icon" src="figs/{name}">'
                        log.setdefault("icons", 0); log["icons"] += 1
                    body.append(f"<p>{icon}{inline(c)}</p>")
                    last_p = len(body) - 1
            first_text_on_page = False
        if pending_caption:
            body.append(f'<p class="figtitle">{inline(pending_caption)}</p>'); pending_caption = None

    css = (HERE / "pdf.css").read_text()
    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(DOC_TITLE)}</title><style>{css}</style></head><body>
{chr(10).join(b for b in body if b)}
</body></html>"""
    (WORK / "out.html").write_text(out)
    log["categories"] = dict(cats)
    (WORK / "build_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items()}, indent=None))


if __name__ == "__main__":
    main()
