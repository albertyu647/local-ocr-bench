"""Automatic quality checks on OCR output, without anyone reading the text.
Flags: unparseable output, hit the token limit, empty page, figure boxes that are huge/tiny,
and low word agreement with the PDF's own (ABBYY) text layer. Prints only page numbers + reasons.
Writes <work>/check_report.json.  Run: python pipeline/check_pdf.py <source.pdf> <work_dir>   (expects <work_dir>/ocr/p-NNN.json)"""
import json, re, sys
from collections import Counter
from pathlib import Path
import fitz

WORK = Path(sys.argv[2])
OCR = WORK / "ocr"
doc = fitz.open(sys.argv[1])


def words(s):
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return Counter(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s))


rows, flags = [], {}
for f in sorted(OCR.glob("p-*.json")):
    pno = int(f.stem.split("-")[1])
    rec = json.loads(f.read_text())
    m = re.search(r"\[.*\]", rec["text"], re.S)
    try:
        items = json.loads(m.group(0)) if m else None
    except Exception:
        items = None
    why = []
    if rec.get("finish") not in ("stop", None):
        why.append(f"finish={rec['finish']}")
    if items is None:
        why.append("unparseable JSON")
        items = []
    keep = [it for it in items if it.get("category") not in ("header", "footer", "number")]
    if not keep:
        why.append("no content blocks")
    figs = [it for it in items if it.get("category") in ("image", "chart", "figure")]
    for it in figs:
        x1, y1, x2, y2 = it["bbox"]
        area = (x2 - x1) * (y2 - y1) / 1e6
        if area < 0.01:
            why.append(f"tiny figure box {it['bbox']}")
        if area > 0.85:
            why.append("figure covers whole page")
    ocr_text = "\n".join(re.sub(r"<caption>.*?</caption>|<mermaid>.*?</mermaid>|<[^>]+>", " ", it.get("content", ""), flags=re.S)
                         for it in items if it.get("category") not in ("image", "chart", "figure"))
    tl = doc[pno - 1].get_text()
    wo, wt = words(ocr_text), words(tl)
    # words inside figures are in the text layer but (by design) not in our output, so judge by precision+recall separately
    tp = sum((wo & wt).values())
    prec = tp / max(1, sum(wo.values())); rec_ = tp / max(1, sum(wt.values()))
    if sum(wt.values()) >= 40 and rec_ < 0.85 and not figs:
        why.append(f"missing text vs PDF layer (recall {rec_:.2f})")
    if sum(wo.values()) >= 40 and prec < 0.85:
        why.append(f"text disagrees with PDF layer (precision {prec:.2f})")
    rows.append(dict(page=pno, blocks=len(items), figures=len(figs), ocr_words=sum(wo.values()),
                     layer_words=sum(wt.values()), precision=round(prec, 3), recall=round(rec_, 3), flags=why))
    if why:
        flags[pno] = why

tot_w = sum(r["ocr_words"] for r in rows)
summary = dict(pages=len(rows), words=tot_w, figures=sum(r["figures"] for r in rows),
               mean_precision=round(sum(r["precision"] for r in rows if r["ocr_words"] >= 40) /
                                    max(1, sum(1 for r in rows if r["ocr_words"] >= 40)), 3),
               flagged=len(flags))
(WORK / "check_report.json").write_text(json.dumps(dict(summary=summary, flagged=flags, pages=rows), indent=1))
print(json.dumps(summary))
for p, w in flags.items():
    print(p, "; ".join(w))
