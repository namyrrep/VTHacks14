"""West Virginia Statewide EMS Protocols (WVOEMS, July 2024) parser.

Chunk kinds produced:
  protocol    - one treatment protocol (T008 Burns, M007 Overdose, ...), split at box
                boundaries into chunks of bounded size
  medication  - one drug from the EMS Medication Formulary
  reference   - preface, "Using the Protocols" and appendices A-I

The PDF was exported from Canva: every page is a flowchart of free-floating text boxes and
PyMuPDF returns those blocks in arbitrary order. Reading order is rebuilt spatially: blocks
that stack vertically in the same column are merged into boxes, and boxes are read in rows
(top to bottom), left to right within a row. Provider-level markers (the E / A / P tags next
to each step) and the rotated side labels are dropped.
"""
import re

import pymupdf

from .common import clean, join_wrapped, make_chunk, pack, slug, title_case, visual_rows, Line

MANUAL = "West Virginia Statewide EMS Protocols"
PREFIX = "wv"

TOC_PAGES = range(3, 6)
# front-matter pages carry only the manual banner, so their sections are named here
FRONT_MATTER = {7: "Preface", 9: "Using the Protocols"}
CODE_RE = re.compile(r"^(?:A|P)?(?:UC|T|C|R|M|E|GL)\d{3}$|^[A-I]$")
BULLET_CHARS = "•"
PROVIDER_MARKS_RE = re.compile(r"^(?:[EAPM](?:\s+|$))+$")
DROP_LINES = {"Empowering Success", "2024"}
FOOTER_RE = re.compile(r"WEST VIRGINIA OFFICE? OF EMERGENCY MEDICAL SERVICES|^PG \d+ of \d+$|^July 2024")


# ---------------------------------------------------------------- page metadata

def parse_toc(doc) -> dict[str, str]:
    """code -> protocol title, from the three table-of-contents pages."""
    titles = {}
    for pno in TOC_PAGES:
        lines = [clean(l) for l in doc[pno - 1].get_text().splitlines()]
        lines = [l for l in lines if l]
        for prev, cur in zip(lines, lines[1:]):
            if CODE_RE.match(cur) and not CODE_RE.match(prev):
                titles[cur] = prev
    # the pages print "UC001" where the contents list "AUC001"
    for code in list(titles):
        if code.startswith("AUC"):
            titles[code[1:]] = titles[code]
    return titles


# Symbol-font glyphs that PyMuPDF reports as private-use code points
SYMBOL_FONT = {"": "®", "": "≥", "": "≤", "": "±", "": "°", "": "→"}


def wv_clean(text: str) -> str:
    return clean("".join(SYMBOL_FONT.get(ch, ch) for ch in text))


def page_header(page) -> tuple[str | None, str | None, str | None]:
    """(protocol code, category banner, medication name) printed at the top of the page."""
    code = category = med = None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if line["bbox"][1] > 80:
                continue
            spans = [s for s in line["spans"] if s["text"].strip()]
            for s in spans:
                text = clean(s["text"])
                if s["size"] >= 24 and "Bold" in s["font"] and CODE_RE.match(text):
                    code = text
                elif s["size"] >= 40:
                    category = f"{category} {text}" if category else text
            # a formulary page opens with the drug name in 12pt bold (trade-name marks inline)
            if (med is None and spans and "Arial-Bold" in spans[0]["font"]
                    and 11.5 <= spans[0]["size"] <= 12.5 and line["bbox"][0] < 200):
                med = wv_clean("".join(s["text"] for s in line["spans"]))
    return code, category, med


# ---------------------------------------------------------------- spatial reading order

class Block:
    def __init__(self, bbox, lines):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.lines = lines  # [(text, bold, x, y)]

    @property
    def width(self):
        return max(self.x1 - self.x0, 1)


def page_blocks(page) -> list[Block]:
    h = page.rect.height
    out = []
    for b in page.get_text("dict")["blocks"]:
        lines = []
        for line in b.get("lines", []):
            if tuple(round(v) for v in line["dir"]) != (1, 0):
                continue  # rotated side labels
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans or max(s["size"] for s in spans) >= 24:
                continue  # category banner and protocol code
            text = wv_clean("".join(s["text"] for s in line["spans"]))
            if not text or text in DROP_LINES or FOOTER_RE.search(text) or PROVIDER_MARKS_RE.match(text):
                continue
            lines.append((text, all("Bold" in s["font"] for s in spans), line["bbox"][0], line["bbox"][1]))
        if lines and b["bbox"][1] < h - 35:
            out.append(Block(b["bbox"], lines))
    return out


def boxes_in_reading_order(blocks: list[Block]) -> list[list[Block]]:
    boxes: list[list[Block]] = []
    for b in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        home = None
        for box in reversed(boxes):
            last = box[-1]
            overlap = min(b.x1, last.x1) - max(b.x0, last.x0)
            if overlap / min(b.width, last.width) >= 0.5 and -8 <= b.y0 - last.y1 <= 14:
                home = box
                break
        if home is not None:
            home.append(b)
        else:
            boxes.append([b])
    # rows of boxes that start at about the same height, each read left to right
    boxes.sort(key=lambda box: box[0].y0)
    rows: list[list[list[Block]]] = []
    for box in boxes:
        if rows and box[0].y0 < rows[-1][0][0].y0 + 25:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [box for row in rows for box in sorted(row, key=lambda bx: bx[0].x0)]


def box_paragraphs(box: list[Block], page: int) -> list[tuple[str, str, int]]:
    paras: list[list] = []
    for block in box:
        for i, (text, bold, _, _) in enumerate(block.lines):
            bullet = text[0] in BULLET_CHARS
            if bullet:
                text = "• " + text[1:].strip()
            is_heading = bold and len(text) < 60 and not bullet and not text.endswith((",", "."))
            # a new block continues the paragraph above only when it is visibly mid-sentence
            new_block_start = i == 0 and text[:1].isupper()
            if is_heading:
                paras.append(["heading", text, page])
            elif bullet or new_block_start or not paras or paras[-1][0] == "heading":
                paras.append(["bullet" if bullet else "text", text, page])
            else:
                paras[-1][1] = join_wrapped(paras[-1][1], text)
    return [tuple(p) for p in paras]


def page_paragraphs(page) -> list[tuple[str, str, int]]:
    out = []
    for box in boxes_in_reading_order(page_blocks(page)):
        out.extend(box_paragraphs(box, page.number + 1))
    return out


def table_rows(page) -> list[tuple[str, str, int]]:
    """Two-column label/value pages (formulary, abbreviations): one paragraph per row."""
    lines = []
    for b in page_blocks(page):
        for text, bold, x, y in b.lines:
            lines.append(Line(text=text, page=page.number + 1, x=x, y=y, size=10, bold=bold))
    return visual_rows(lines, tol=4, merge_continuations=True)


# ---------------------------------------------------------------- units

def collect_units(doc, titles) -> list[dict]:
    """Group consecutive pages into protocols / medications / reference sections."""
    units: list[dict] = []
    for pno in range(1, doc.page_count + 1):
        if pno in TOC_PAGES:
            continue
        page = doc[pno - 1]
        if len(page.get_text().split()) < 5:
            continue
        code, category, med = page_header(page)
        in_formulary = category is None and med is not None or (units and units[-1]["kind"] == "medication")
        if pno in FRONT_MATTER:
            key, kind, title = FRONT_MATTER[pno], "reference", FRONT_MATTER[pno]
        elif code and len(code) > 1:
            key, kind = code, "protocol"
            title = titles.get(code, category or code)
        elif code:  # appendix letter
            key, kind = f"Appendix {code}", "reference"
            title = f"Appendix {code} – {titles.get(code, category or '')}".rstrip(" –")
        elif med and in_formulary:
            key, kind, title = med, "medication", med
        elif category and category.upper() not in ("APPENDIX",):
            key, kind, title = category, "reference", title_case(category)
        elif units:
            key, kind, title = units[-1]["key"], units[-1]["kind"], units[-1]["title"]
        else:
            continue
        if not units or units[-1]["key"] != key:
            units.append({"key": key, "kind": kind, "title": title, "code": code, "category": category, "pages": []})
        units[-1]["pages"].append(pno)
    return units


def parse(pdf_path) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    titles = parse_toc(doc)
    chunks = []
    category = None
    for unit in collect_units(doc, titles):
        if unit["category"] and unit["kind"] == "protocol":
            category = unit["category"]
        tabular = unit["kind"] == "medication" or unit["key"] == "Appendix E"
        paras = []
        for pno in unit["pages"]:
            paras.extend(table_rows(doc[pno - 1]) if tabular else page_paragraphs(doc[pno - 1]))
        if unit["kind"] == "protocol":
            section = f"{unit['key']} – {unit['title']}"
            base = [unit["key"], unit["title"]] + ([title_case(category)] if category else [])
        elif unit["kind"] == "medication":
            section = f"Medication Formulary – {unit['title']}"
            base = [unit["title"], "medication", "dose", "formulary"]
        else:
            section = unit["title"]
            base = [unit["title"]]
        for i, piece in enumerate(pack(paras), start=1):
            # formulary rows are bold labels ("Generic Name:"), not subsections
            sub = piece["heading"] if unit["kind"] != "medication" else None
            chunks.append(make_chunk(
                cid=f"{PREFIX}_{slug(unit['key'], 30)}_{i:02d}",
                manual=MANUAL,
                section=section if not sub or sub == unit["title"] else f"{section} – {sub}",
                page=piece["page"],
                printed_page="",
                text=piece["text"],
                keywords=base + ([sub] if sub else []),
                kind=unit["kind"],
                protocol_id=f"{PREFIX}_{slug(unit['key'], 30)}",
                protocol=section,
            ))
    return chunks
