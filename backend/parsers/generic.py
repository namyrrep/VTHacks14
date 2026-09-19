"""Fallback parser for a manual without a dedicated parser.

Splits the PDF at its bookmarks (table of contents) when it has them, otherwise treats the
whole document as one section, then packs headings/paragraphs into bounded chunks.
"""
import re
from pathlib import Path

import pymupdf

from .common import Line, make_chunk, pack, page_lines, reflow, slug, title_case

FURNITURE_RE = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]+)$", re.I)


def body_lines(page) -> list[Line]:
    h = page.rect.height
    return [ln for ln in page_lines(page)
            if not ((ln.y < 40 or ln.y > h - 50) and FURNITURE_RE.match(ln.text))]


def parse(pdf_path, manual: str | None = None, prefix: str | None = None) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    manual = manual or (doc.metadata or {}).get("title") or Path(pdf_path).stem
    prefix = prefix or slug(Path(pdf_path).stem, 12)

    toc = [(title, page) for level, title, page in doc.get_toc() if level == 1 and page > 0]
    if not toc:
        toc = [(manual, 1)]
    sections = []
    for i, (title, first) in enumerate(toc):
        last = (toc[i + 1][1] - 1) if i + 1 < len(toc) else doc.page_count
        sections.append((re.sub(r"\s+", " ", title).strip(), first, max(first, last)))

    chunks = []
    for title, first, last in sections:
        lines = [ln for pno in range(first, last + 1) for ln in body_lines(doc[pno - 1])]
        if not lines:
            continue
        body_size = sorted(ln.size for ln in lines)[len(lines) // 2]

        def is_heading(ln: Line) -> bool:
            return ((ln.bold and ln.size >= body_size) or ln.size > body_size + 1.5) and len(ln.text) < 90

        paras = [p for p in reflow(lines, is_heading) if not (p[0] == "heading" and slug(p[1]) == slug(title))]
        for i, piece in enumerate(pack(paras), start=1):
            sub = piece["heading"]
            chunks.append(make_chunk(
                cid=f"{prefix}_{slug(title, 30)}_{i:02d}",
                manual=manual,
                section=title if not sub else f"{title_case(title)} – {title_case(sub)}",
                page=piece["page"],
                printed_page="",
                text=piece["text"],
                keywords=[title_case(title)] + ([title_case(sub)] if sub else []),
                kind="reference",
                protocol_id=f"{prefix}_{slug(title, 30)}",
                protocol=title_case(title),
            ))
    return chunks
