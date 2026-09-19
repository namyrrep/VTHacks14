"""NIOSH Pocket Guide to Chemical Hazards (DHHS/NIOSH Publication No. 2005-149) parser.

Chunk kinds produced:
  chemical   - one chemical record from the Chemical Listing (two records per page)
  reference  - front matter (introduction, how to use, code tables 1-6) and appendices A-F

Each chemical record is rebuilt field by field from the page's bold labels and regular-weight
values, so every value is verbatim but fields read top-to-bottom instead of in PDF column order.
"""
import re

import pymupdf

from .common import (Line, clean, join_wrapped, make_chunk, pack, page_lines, reflow, slug, title_case,
                     visual_rows)

MANUAL = "NIOSH Pocket Guide to Chemical Hazards"
PREFIX = "niosh"

CHEMICAL_PAGES = range(32, 371)  # PDF pages, 1-based
PRINTED_OFFSET = 30              # chemical listing: printed page = PDF page - 30

# (title, first page, last page, layout)
REFERENCE_SECTIONS = [
    ("Introduction", 7, 8, "prose"),
    ("How to Use This Pocket Guide", 8, 16, "prose"),
    ("Table 1 – NIOSH Manual of Analytical Methods", 17, 17, "rows"),
    ("Table 2 – Personal Protection and Sanitation Codes", 18, 19, "codes3"),
    ("Table 3 – Symbols, Code Components, and Codes Used for Respirator Selection", 20, 24, "rows"),
    ("Table 4 – Selection of N-, R-, or P-Series Particulate Respirators", 25, 25, "rows"),
    ("Table 5 – Abbreviations for Exposure Routes, Symptoms, and Target Organs", 26, 27, "rows"),
    ("Table 6 – Codes for First Aid Data", 28, 30, "rows"),
    ("Appendix A – NIOSH Potential Occupational Carcinogens", 372, 373, "prose"),
    ("Appendix B – Thirteen OSHA-Regulated Carcinogens", 374, 374, "prose"),
    ("Appendix C – Supplementary Exposure Limits", 375, 379, "prose"),
    ("Appendix D – Substances with No Established RELs", 380, 380, "prose"),
    ("Appendix E – OSHA Respirator Requirements for Selected Chemicals", 381, 390, "prose"),
    ("Appendix F – Miscellaneous Notes", 391, 391, "prose"),
]

# Field labels that start a new line in a chemical record, in the order they are printed.
FIELDS = [
    "Formula:", "CAS#:", "RTECS#:", "IDLH:", "Conversion:", "DOT:",
    "Synonyms/Trade Names:", "Exposure Limits:", "Physical Description:",
    "Measurement Methods", "Chemical & Physical Properties:",
    "Personal Protection/Sanitation", "Respirator Recommendations",
    "Incompatibilities and Reactivities:",
    "Exposure Routes, Symptoms, Target Organs (see Table 5):",
    "First Aid (see Table 6):",
]
# Labels whose sub-entries sit one per line (MW:, BP:, Eye:, Skin:, ...) and read best joined by "; "
LIST_FIELDS = {"Chemical & Physical Properties:", "Personal Protection/Sanitation",
               "Respirator Recommendations", "Exposure Routes, Symptoms, Target Organs (see Table 5):",
               "First Aid (see Table 6):", "Measurement Methods", "Exposure Limits:"}
FURNITURE_RE = re.compile(r"^(?:[ivxlc]+|\d{1,3}|[A-Z]|(?:[A-Z] ){3,}[A-Z]?)$")
APPENDIX_BANNER_RE = re.compile(r"^A P P E N D I X [A-G]$")


def field_of(label: str) -> str | None:
    label = label.strip()
    return next((f for f in FIELDS if label == f or label == f.rstrip(":")), None)


def field_prefix(label: str) -> tuple[str, str] | None:
    """A label printed over two lines ('Chemical & Physical' / 'Properties:'):
    returns (field, the remainder expected next)."""
    label = label.strip()
    if len(label) < 8:
        return None
    for f in FIELDS:
        if f.startswith(label) and f != label:
            return f, f[len(label):].strip()
    return None


# ---------------------------------------------------------------- chemical records

def record_starts(page) -> list[tuple[float, str]]:
    """(y, chemical name) for each record on a chemical-listing page.
    Names are the only 8pt bold text at the left margin; long names wrap onto a second line."""
    starts = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            # left margin is 36pt on even pages, 45pt on odd pages
            if not spans or line["bbox"][0] > 50:
                continue
            # the name line may open with a Greek letter set in Times ("α-Alumina") or
            # carry subscripts ("Bi2Te3"), so look for the 8pt bold name font anywhere on it
            # a long name can also share its PDF line with the 6.5pt "Formula:" label
            name_spans = []
            for s in line["spans"]:
                if abs(s["size"] - 6.5) < 0.3:
                    break
                name_spans.append(s)
            if any("Bold" in s["font"] and 7.5 <= s["size"] <= 8.5 for s in name_spans):
                text = clean("".join(s["text"] for s in name_spans))
                y = line["bbox"][1]
                if starts and y - starts[-1][2] < 12:
                    starts[-1] = (starts[-1][0], f"{starts[-1][1]} {text}", y)
                else:
                    starts.append((y, text, y))
    return [(y, name) for y, name, _ in starts]


def record_spans(page, y0, y1):
    """Lines of the record between y0 and y1 as lists of (is_bold, text) spans, in block order.
    Skips the alphabet thumb-tab (12pt letter) and the page-number footer."""
    out = []
    footer = page.rect.height - 40
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            y = line["bbox"][1]
            if not (y0 - 3 <= y < y1 - 3) or y > footer:
                continue
            spans = [("Bold" in s["font"], s["text"]) for s in line["spans"] if s["text"] and s["size"] < 11]
            if any(t.strip() for _, t in spans):
                out.append(spans)
    return out


def build_record(lines, name) -> tuple[str, dict]:
    """Rebuild a record as 'Label value' lines. Returns (text, fields)."""
    fields: dict[str, str] = {}
    order: list[str] = []
    cur = None
    name_left = name
    pending_rest = None  # second half of a label split over two lines
    for spans in lines:
        first = True
        for bold, raw in spans:
            text = raw.replace(" ", " ")
            if not text.strip():
                if cur:
                    fields[cur] += text
                continue
            if bold and pending_rest and text.strip() == pending_rest:
                pending_rest = None
                continue
            f = field_of(text) if bold else None
            if bold and not f and (split := field_prefix(text)):
                f, pending_rest = split
            if f:
                cur = f
                if f not in fields:
                    fields[f] = ""
                    order.append(f)
            elif cur is None:
                # the chemical name (and its wrapped second line) precede the first label
                if bold and name_left and clean(text) and clean(text) in name_left:
                    name_left = name_left.replace(clean(text), "", 1).strip()
                continue
            elif bold and re.fullmatch(r"\(see Tables? [\d and]+\):", text.strip()):
                fields[cur] += " " + text.strip() + " "
            else:
                # separate sub-entries with "; ", but not right after "(see Table N):"
                has_value = bool(re.sub(r"^\(see Tables? [\d and]+\):", "", fields[cur].strip()).strip())
                if first and has_value and cur in LIST_FIELDS and bold:
                    fields[cur] = fields[cur].rstrip() + "; "
                elif first:
                    fields[cur] += " "
                fields[cur] += text
            first = False
    for f in fields:
        v = clean(fields[f])
        v = re.sub(r"\s+;", ";", v)
        v = re.sub(r"^\(see Tables? [\d and]+\):\s*", lambda m: m.group(0).strip() + " ", v)
        fields[f] = v
    # empty fields (e.g. "DOT:" for substances with no DOT number) carry no information
    body = "\n".join(f"{f} {fields[f]}" for f in order if fields[f])
    return f"{name}\n{body}", fields


def split_list(value: str) -> list[str]:
    """Split a synonym list on commas/semicolons that are not inside brackets."""
    value = re.sub(r"\[Note:.*?\]", "", value)
    out, depth, cur = [], 0, ""
    for i, ch in enumerate(value):
        depth += ch in "([" and 1 or ch in ")]" and -1 or 0
        if depth == 0 and ch in ",;" and value[i + 1:i + 2] == " ":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [clean(x) for x in out if clean(x)]


def chemical_keywords(name, fields) -> list[str]:
    kws = [name]
    kws += split_list(fields.get("Synonyms/Trade Names:", ""))
    for key, prefix in (("CAS#:", "CAS "), ("RTECS#:", "RTECS ")):
        v = fields.get(key, "").strip()
        if v and v not in ("N.D.", "None"):
            kws += [v, prefix + v]
    # "DOT: 1089 129" / "2790 153 (10-80% acid); 2789 132 (>80% acid)"
    for un, guide in re.findall(r"\b(\d{4})\s+(\d{3})P?\b", fields.get("DOT:", "")):
        kws += [f"UN{un}", f"ERG Guide {guide}"]
    if fields.get("IDLH:", "").startswith("Ca"):
        kws.append("potential occupational carcinogen")
    return kws


def parse_chemicals(doc) -> list[dict]:
    chunks = []
    for pno in CHEMICAL_PAGES:
        page = doc[pno - 1]
        starts = record_starts(page)
        bounds = [y for y, _ in starts] + [page.rect.height - 30]
        for i, (y, name) in enumerate(starts):
            text, fields = build_record(record_spans(page, y, bounds[i + 1]), name)
            chunks.append(make_chunk(
                cid=f"{PREFIX}_{slug(name, 60)}",
                manual=MANUAL,
                section=name,
                page=pno,
                printed_page=str(pno - PRINTED_OFFSET),
                text=text,
                keywords=chemical_keywords(name, fields),
                kind="chemical",
            ))
    return chunks


# ---------------------------------------------------------------- front matter & appendices

def printed_label(doc, pno) -> str:
    """Roman numeral in the front matter, arabic page number elsewhere."""
    lines = page_lines(doc[pno - 1])
    for ln in lines:
        if ln.y > doc[pno - 1].rect.height - 45 or ln.y < 30:
            m = re.search(r"\b([ivxlc]+|\d{1,3})\b", ln.text)
            if m and FURNITURE_RE.match(m.group(1)):
                return m.group(1)
    return str(pno - PRINTED_OFFSET) if pno > PRINTED_OFFSET else ""


def body_lines(doc, pno) -> list[Line]:
    page = doc[pno - 1]
    out = []
    for ln in page_lines(page):
        edge = ln.y < 30 or ln.y > page.rect.height - 45
        if edge and FURNITURE_RE.match(ln.text.replace(" ", " ")):
            continue
        # the vertical "A P P E N D I X" banner comes out as single letters
        if APPENDIX_BANNER_RE.match(ln.text) or re.fullmatch(r"[A-Z]", ln.text):
            continue
        out.append(ln)
    return out


def three_column_rows(lines: list[Line]) -> list[tuple[str, str, int]]:
    """Category | Code | Definition tables (Table 2) whose cells are vertically centred.

    Each definition paragraph is a row; the code cell next to it and any category label
    ('Skin:', 'Wash skin:') are attached by vertical position.
    """
    out = []
    for page in sorted({ln.page for ln in lines}):
        pl = [ln for ln in lines if ln.page == page]
        head = next((ln for ln in pl if ln.text == "Definition"), None)
        cat = next((ln for ln in pl if ln.text == "Code"), None)
        if not head or not cat:
            continue
        body = [ln for ln in pl if ln.y > head.y + 5]
        def_x = head.x - 5
        cat_lines = [ln for ln in body if ln.x < cat.x + 15 and ln.bold]
        code_lines = [ln for ln in body if cat.x + 15 <= ln.x < def_x]
        defs: list[list[Line]] = []
        for ln in (l for l in body if l.x >= def_x):
            if defs and ln.y - defs[-1][-1].y <= 11:
                defs[-1].append(ln)
            else:
                defs.append([ln])
        rows = []
        for para in defs:
            top, bottom = para[0].y - 6, para[-1].y + 6
            codes = [c.text for c in code_lines if top <= c.y <= bottom]
            text = para[0].text
            for ln in para[1:]:
                text = join_wrapped(text, ln.text)
            rows.append([top, bottom, " ".join(codes), text])
        # a category label starts at the first row it lines up with
        for c in cat_lines:
            row = next((r for r in rows if r[0] <= c.y <= r[1] + 10), None)
            if row is not None:
                row.append(c.text)
        for r in rows:
            cats = r[4:]
            if cats:
                out.append(("heading", " ".join(cats), page))
            out.append(("text", f"{r[2]} – {r[3]}" if r[2] else r[3], page))
    return out


def parse_reference(doc) -> list[dict]:
    chunks = []
    for title, first, last, layout in REFERENCE_SECTIONS:
        lines = []
        for pno in range(first, last + 1):
            lines.extend(body_lines(doc, pno))
        # sections sharing a page: keep only this section's part of it
        head = title.split(" – ")[0].upper()
        start = next((i for i, ln in enumerate(lines) if ln.text.upper().startswith(head)), 0)
        lines = lines[start:]
        nxt = next((t for t, f, _, _ in REFERENCE_SECTIONS if f == last and t != title and f > first), None)
        if nxt:
            nxt_head = nxt.split(" – ")[0].upper()
            end = next((i for i, ln in enumerate(lines) if i and ln.text.upper().startswith(nxt_head)), len(lines))
            lines = lines[:end]
        if not lines:
            continue
        body_size = sorted(ln.size for ln in lines)[len(lines) // 2]

        def is_heading(ln: Line) -> bool:
            return ln.bold and ln.size >= body_size and len(ln.text) < 90 and not ln.text.endswith((",", ";"))

        if layout == "rows":
            paras = visual_rows(lines, merge_continuations=True)
        elif layout == "codes3":
            paras = three_column_rows(lines)
        else:
            paras = reflow(lines, is_heading)
        title_slug = slug(title, 200)
        paras = [p for p in paras if not (p[0] == "heading" and (
            slug(p[1], 200) in title_slug or re.match(r"^(table \d|appendix [a-g])", p[1].lower())))]
        # column headers repeated at the top of every table page
        paras = [p for p in paras if not re.fullmatch(r"(?:Code (?:APF )?Description|Code Definition)(?: Code Definition)*", p[1])]
        for i, piece in enumerate(pack(paras), start=1):
            sub = piece["heading"]
            section = title if not sub else f"{title} – {title_case(sub)}"
            chunks.append(make_chunk(
                cid=f"{PREFIX}_ref_{slug(title, 30)}_{i:02d}",
                manual=MANUAL,
                section=section,
                page=piece["page"],
                printed_page=printed_label(doc, piece["page"]),
                text=piece["text"],
                keywords=[title] + ([title_case(sub)] if sub else []),
                kind="reference",
            ))
    return chunks


def parse(pdf_path) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    return parse_chemicals(doc) + parse_reference(doc)
