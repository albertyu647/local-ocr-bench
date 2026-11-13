# Benchmark method and full results

## Models

All four were released in the two weeks before 2026-11-13, and each was described as reading diagrams, running locally and being fast.

| Model | Size | Links | How I ran it on the Mac |
|---|---|---|---|
| OvisOCR3 | 0.8B (Qwen3.5 base) | [Hugging Face](https://huggingface.co/ATH-MaaS/OvisOCR3) · [demo](https://huggingface.co/spaces/ATH-MaaS/OvisOCR3) | Converted to MLX 8-bit; `document_markdown` and `figure_captioning` prompts from the model card |
| MiniCPM-OCR 1.0 | 1.3B | [GitHub](https://github.com/OpenBMB/MiniCPM-OCR) · [Hugging Face](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) | Converted to MLX 8-bit. (1) the whole page with `Text Recognition:`; (2) the official `minicpm-ocr` SDK, which runs the PP-DocLayoutV3 layout model, then MiniCPM on each crop through an `mlx_vlm server` |
| Unlimited-OCR-2 | 3B | [GitHub](https://github.com/baidu/Unlimited-OCR) · [Hugging Face](https://huggingface.co/baidu/Unlimited-OCR-2) | [mlx-community 8-bit](https://huggingface.co/mlx-community/Unlimited-OCR-2-8bit); `Multi page parsing.` prompt |
| Mirage | 51M | [Hugging Face](https://huggingface.co/Cactus-Compute/mirage) · [Cactus](https://github.com/cactus-compute/cactus) | `cactus-compute` 4.2, `Mirage.describe()` |
| Apple Vision | n/a | built into macOS | `VNRecognizeTextRequest`, accurate mode, language correction on |

Runtime: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.12 on MLX 0.32.4. All three LLM-based models load natively in it. The Homebrew llama.cpp build at the time couldn't load MiniCPM's `minicpm_ocr_1_0` projector, and the Unlimited-OCR-2 GGUF needs a patched llama.cpp, so I ran everything on MLX to keep the speed numbers comparable.

## Test set

13 pages from one scanned print document, rendered at 200 dpi:

- 4 plain-text pages (prose and dialogue)
- 9 figure pages: a 4-circle Venn diagram, stacked levels, a 3×3 grid, a 9-box grid with text in the cells, a line chart, a flowchart, a table, a fill-in form, and a Gantt chart

For each page I wrote a reference transcript by hand, starting from the scan's text layer and fixing it against the image. I also listed every label that appears inside each figure: 125 labels in total. The pages are copyrighted, so neither the images nor the references are published.

## Metrics

- **Character error rate (CER):** the edit distance divided by the reference length, after normalization. It also counts reading order.
- **Word F1:** precision and recall on the multiset of words. It ignores order, so a different but valid layout choice isn't punished.
- **Diagram label recall:** the share of figure labels found in the output, either as an exact substring or with a RapidFuzz `partial_ratio` of at least 90.
- **Normalization before scoring:** remove markdown images, HTML tags, LaTeX markup, and bounding-box or JSON wrappers; rejoin words split by a hyphen at a line end; map curly quotes and dashes to plain ones; lowercase everything.
- **Captions and Mermaid code:** they count toward diagram recall only, never toward text accuracy.
- **Timing:** one untimed warm-up page, then wall-clock time per page. Model load time isn't included.

## Diagram label recall by figure type (Mac)

| | Venn | Levels | 3×3 grid | 9-box | Line chart | Flowchart | Table | Form | Gantt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Labels | 4 | 8 | 8 | 23 | 10 | 3 | 31 | 10 | 28 |
| **OvisOCR3 figure** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 0.90 | **1.00** |
| MiniCPM whole page | 0.00 | 1.00 | 0.88 | 0.61 | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 |
| Apple Vision | 0.50 | 1.00 | 0.62 | 0.48 | 1.00 | 1.00 | 0.90 | 1.00 | 0.89 |
| OvisOCR3 markdown | 0.00 | 1.00 | 0.38 | 0.91 | 0.60 | 0.00 | 1.00 | 0.90 | 0.14 |
| MiniCPM SDK | 0.00 | 1.00 | 0.38 | 0.04 | 0.90 | 0.00 | 1.00 | 0.90 | 0.61 |
| Unlimited-OCR-2 | 0.00 | 0.25 | 0.38 | 0.04 | 0.60 | 0.00 | 1.00 | 0.90 | 0.04 |
| Mirage | 0.00 | 0.75 | 0.50 | 0.09 | 0.80 | 1.00 | 0.48 | 0.90 | 0.68 |

## What each model is like

- **OvisOCR3:** the only one that reads figures. Figure mode returns a JSON list of `{bbox, category, content}` for the page, with a `<caption>` for each image or chart and `<mermaid>` for flowcharts. The labels it extracted were correct, but the arrows are partly invented: it drew the Venn diagram as a flowchart, and one caption came out partly in Chinese. Markdown mode gives clean markdown and `<img>` boxes for figures, but on simple text-only diagrams it sometimes types the labels out instead of boxing the figure.
- **MiniCPM-OCR 1.0:** the lowest character error in whole-page mode. Diagram text comes out as flat lines with no structure, and it skipped the Venn labels completely. The official SDK pipeline scored worse here: it crops figures out as images and drops running headers.
- **Unlimited-OCR-2:** clean text with bounding boxes, but every figure comes back as an empty `image` block. It also used the most memory (5.9 GB).
- **Mirage:** a 51M scene-understanding model, not a document OCR model. It returns a scene description and partial text lines, many of them cut off. It's very fast.
- **Apple Vision:** about 10× faster than the LLM-based models and nearly as accurate on plain text. It reads the words in diagrams but not their structure.

## Cross-runtime check

OvisOCR3's outputs from the Mac (MLX 8-bit), PC transformers (bf16) and PC vLLM (bf16) were scored with the same script:

| Runtime | Markdown CER | Figure-mode diagram recall |
|---|---:|---:|
| Mac, MLX 8-bit | 0.078 | 98.9% |
| PC, transformers bf16, single and batch 13 | 0.077 | 98.9% |
| PC, vLLM bf16 | 0.078 | 98.9% |

The PC's transformers runs matched the Mac's output byte for byte on 7 to 9 of the 13 pages, and the rest were at least 95% similar. Quantizing to 8-bit on MLX cost nothing measurable.

## Full-document run

A 278-page scan through `pipeline/` on the 3060 Ti with vLLM, figure mode, 40 pages per batch:

- **OCR:** 191.4 s for 278 pages.
- **Automatic checks** (`check_pdf.py`): the OCR words agreed 98.7% with the scan's text layer. 36 pages were flagged, and all were explained:
  - 15 blank or black scans, which were dropped
  - pages with small margin icons, which were kept beside their paragraphs
  - a sideways table that the OCR read correctly, while the text layer there was garbage
  - a page of worked figures returned as LaTeX, which was converted to plain symbols
- **Build:** 14 figures cropped at 300 dpi, 20 tables, 69 paragraphs rejoined across page breaks, 282 output pages, 4.4 MB.
- **Known model slip:** one dollar sign dropped from a currency amount.
