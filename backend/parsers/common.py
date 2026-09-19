"""Shared helpers: line extraction, paragraph reflow, size-bounded packing, chunk building."""
import re
from collections import Counter
import unicodedata
from dataclasses import dataclass

import pymupdf

BULLET_RE = re.compile(r"^\s*(?:[•●▪◦]|[–—-]{2}|›+|--)\s*")
WS_RE = re.compile(r"[ \t ]+")

# MiniLM truncates at 256 word pieces; ~180 words keeps a chunk inside that window.
MAX_WORDS = 180
MIN_TAIL_WORDS = 30


@dataclass
class Line:
    text: str
    page: int        # 1-based PDF page
    x: float
    y: float
    size: float      # largest span size on the line
    bold: bool       # every non-blank span is bold
    font: str = ""   # font of the first non-blank span


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("�", "").replace("\t", " ")
    return WS_RE.sub(" ", text).strip()


def page_lines(page: pymupdf.Page, clip=None) -> list[Line]:
    """Visual lines of a page (spans merged per line), in reading order."""
    out = []
    data = page.get_text("dict", clip=clip, sort=True)
    for block in data["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = clean("".join(s["text"] for s in line["spans"]))
            if not text:
                continue
            out.append(Line(
                text=text,
                page=page.number + 1,
                x=line["bbox"][0],
                y=line["bbox"][1],
                size=max(s["size"] for s in spans),
                bold=all("Bold" in s["font"] or "Demi" in s["font"] for s in spans),
                font=spans[0]["font"],
            ))
    return merge_same_row(out)


def merge_same_row(lines: list[Line], tol: float = 2.0) -> list[Line]:
    """PyMuPDF sometimes splits one visual row (e.g. bullet glyph + text) into two lines."""
    merged: list[Line] = []
    for ln in lines:
        prev = merged[-1] if merged else None
        if prev and abs(prev.y - ln.y) <= tol and ln.x > prev.x and BULLET_RE.fullmatch(prev.text + " "):
            prev.text = f"{prev.text} {ln.text}"
            prev.bold = prev.bold and ln.bold
            continue
        merged.append(ln)
    return merged


def is_bullet(text: str) -> bool:
    return bool(BULLET_RE.match(text))


def normalize_bullet(text: str) -> str:
    m = BULLET_RE.match(text)
    if not m:
        return text
    marker = "–" if m.group(0).strip()[:1] in "–—-›" else "•"
    return f"{marker} {text[m.end():].strip()}"


def join_wrapped(prev: str, nxt: str) -> str:
    """Join a soft-wrapped continuation line onto the previous line."""
    if prev.endswith("-") and not prev.endswith(" -") and nxt[:1].islower():
        return prev + nxt
    return f"{prev} {nxt}"


def reflow(lines: list[Line], heading_pred) -> list[tuple[str, str, int]]:
    """Group lines into (kind, text, page) paragraphs. kind is 'heading', 'bullet' or 'text'.

    A new paragraph starts at a heading, a bullet, or a vertical gap larger than ~1.6 lines.
    """
    paras: list[list] = []
    prev = None
    for ln in lines:
        if heading_pred(ln):
            paras.append(["heading", ln.text, ln.page])
            prev = ln
            continue
        if is_bullet(ln.text):
            paras.append(["bullet", normalize_bullet(ln.text), ln.page])
            prev = ln
            continue
        gap = (ln.y - prev.y) if prev and prev.page == ln.page else None
        new_para = (
            not paras
            or paras[-1][0] == "heading"
            or gap is None and paras[-1][2] != ln.page and paras[-1][1].endswith((".", ":"))
            or gap is not None and (gap > ln.size * 1.9 or gap < 0)
        )
        if new_para:
            paras.append(["text", ln.text, ln.page])
        else:
            paras[-1][1] = join_wrapped(paras[-1][1], ln.text)
        prev = ln
    return [tuple(p) for p in paras]


def word_count(text: str) -> int:
    return len(text.split())


def split_long(text: str, max_words: int) -> list[str]:
    """Split one oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.;!?])\s+(?=[A-Z0-9•(])", text)
    out, cur = [], ""
    for s in sentences:
        if cur and word_count(cur) + word_count(s) > max_words:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def pack(paras: list[tuple[str, str, int]], max_words: int = MAX_WORDS) -> list[dict]:
    """Pack paragraphs into chunks of at most max_words, never splitting mid-paragraph
    unless the paragraph alone is too long. A heading is never left dangling at a chunk end.

    Returns [{"text", "page", "heading"}] where heading is the nearest heading in effect.
    """
    pieces = []
    for kind, text, page in paras:
        if kind != "heading" and word_count(text) > max_words:
            pieces.extend((kind, part, page) for part in split_long(text, max_words))
        else:
            pieces.append((kind, text, page))

    chunks, cur, cur_heading, last_heading = [], [], None, None
    for kind, text, page in pieces:
        cur_words = sum(word_count(t) for _, t, _ in cur)
        if cur and cur_words + word_count(text) > max_words:
            # carry trailing headings forward instead of ending a chunk on them
            carry = []
            while cur and cur[-1][0] == "heading":
                carry.insert(0, cur.pop())
            if cur:
                chunks.append({"text": "\n".join(t for _, t, _ in cur), "page": cur[0][2], "heading": cur_heading})
            cur = carry
            cur_heading = carry[0][1] if carry else last_heading
        if kind == "heading":
            last_heading = text
            if not cur:
                cur_heading = text
        cur.append((kind, text, page))
    if cur and any(k != "heading" for k, _, _ in cur):
        tail = "\n".join(t for _, t, _ in cur)
        # fold a tiny trailing fragment into the previous chunk rather than leave it orphaned
        if chunks and word_count(tail) < MIN_TAIL_WORDS and word_count(chunks[-1]["text"]) + word_count(tail) <= max_words * 1.3:
            chunks[-1]["text"] += "\n" + tail
        else:
            chunks.append({"text": tail, "page": cur[0][2], "heading": cur_heading})
    return chunks


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ω": "omega"}


def slug(text: str, maxlen: int = 40) -> str:
    text = "".join(GREEK.get(ch, ch) for ch in clean(text).lower())
    s = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return s[:maxlen].rstrip("_") or "x"


def title_case(heading: str) -> str:
    """'PROTECTIVE CLOTHING' -> 'Protective Clothing'; leaves mixed-case text alone."""
    # acronyms and codes ("AEGL-1", "BLEVE", "UN1017") stay as printed
    if heading.isupper() and len(heading.split()) > 1:
        small = {"and", "or", "of", "the", "to", "for", "in", "on", "a", "an", "with"}
        words = heading.lower().split()
        return " ".join(w if (i and w in small) else w[:1].upper() + w[1:] for i, w in enumerate(words))
    return heading


def make_chunk(cid, manual, section, page, printed_page, text, keywords, kind,
               protocol_id=None, protocol=None):
    """protocol_id / protocol name the whole unit a chunk belongs to (one WV protocol, one
    ERG guide, one NIOSH chemical, ...). A unit split over several chunks shares one
    protocol_id, so search can hand back every part of it. Defaults to the chunk itself."""
    text = "\n".join(WS_RE.sub(" ", ln).strip() for ln in text.split("\n") if ln.strip())
    seen, kws = set(), []
    for k in keywords:
        k = clean(k)
        if k and k.lower() not in seen:
            seen.add(k.lower())
            kws.append(k)
    return {
        "id": cid,
        "manual": manual,
        "section": section,
        "page": page,
        "printed_page": printed_page,
        "kind": kind,
        "text": text,
        "keywords": kws,
        "protocol_id": protocol_id or cid,
        "protocol": protocol or section,
    }


def visual_rows(lines: list[Line], tol: float = 2.5, merge_continuations: bool = False) -> list[tuple[str, str, int]]:
    """One paragraph per visual row of a table: cells on the same baseline joined left to right.

    With merge_continuations, a row whose first cell is indented past the table's left edge
    (a wrapped definition with an empty code column) is joined onto the row above it.
    """
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.page, l.y, l.x)):
        if rows and rows[-1][0].page == ln.page and abs(rows[-1][0].y - ln.y) <= tol:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    for row in rows:
        row.sort(key=lambda l: l.x)
    # the definition column, per page (odd and even pages have different margins):
    # where the second cell of multi-cell rows most often starts
    def_x = {}
    for page in {r[0].page for r in rows}:
        second = Counter(round(r[1].x) for r in rows if len(r) > 1 and r[0].page == page)
        left = min(r[0].x for r in rows if r[0].page == page)
        def_x[page] = second.most_common(1)[0][0] if second else left + 15
    out: list[list] = []
    for row in rows:
        text = " ".join(l.text for l in row)
        heading = all(l.bold for l in row) and len(text) < 90 and row[0].size >= 9
        continuation = (merge_continuations and out and out[-1][0] == "text" and not heading
                        and row[0].x >= def_x[row[0].page] - 5 and out[-1][2] == row[0].page)
        if continuation:
            out[-1][1] = join_wrapped(out[-1][1], text)
        else:
            out.append(["heading" if heading else "text", text, row[0].page])
    return [tuple(r) for r in out]
