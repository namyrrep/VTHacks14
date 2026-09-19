"""Emergency Response Guidebook (ERG 2024) parser.

Chunk kinds produced:
  guide      - each orange Guide split into its three printed parts
               (Potential Hazards / Public Safety / Emergency Response)
  table1     - one row of Table 1 (Initial Isolation and Protective Action Distances)
  table3     - one material block of Table 3 (large spills by container and wind speed)
  table2     - Table 2 (water-reactive materials) in blocks of rows, plus its gas legend
  table_bleve - the BLEVE propane-tank table, one line per tank size
  reference  - prose sections (safety precautions, first aid, glossary, ...) packed by heading

Keywords on guide chunks come from the yellow (ID-number) index: every UN number and
material name that points at that guide.
"""
import re
from collections import defaultdict

import pymupdf

from .common import Line, clean, make_chunk, pack, page_lines, reflow, slug, title_case, visual_rows

MANUAL = "Emergency Response Guidebook 2024"
PREFIX = "erg"

# PDF page ranges (1-based, inclusive)
YELLOW_PAGES = range(31, 90)
GUIDE_PAGES = range(154, 282)
TABLE1_PAGES = range(292, 336)
TABLE2_PAGES = range(337, 342)
TABLE3_PAGES = range(343, 346)

# (title, first page, last page, layout). layout "rows" = one paragraph per visual row.
REFERENCE_SECTIONS = [
    ("Shipping Papers (Documents)", 2, 2, "prose"),
    ("How to Use This Guidebook", 3, 3, "prose"),
    ("Safety Precautions", 6, 6, "prose"),
    ("Notification and Request for Technical Information", 7, 7, "prose"),
    ("Hazard Classification System", 8, 8, "prose"),
    ("Introduction to the Table of Markings, Labels and Placards", 9, 9, "prose"),
    ("Globally Harmonized System of Classification and Labeling of Chemicals (GHS)", 18, 19, "prose"),
    ("Hazard Identification Numbers Displayed on Some Intermodal Containers", 20, 23, "rows"),
    ("Pipeline Transportation", 24, 29, "prose"),
    ("Introduction to Yellow Section", 30, 30, "prose"),
    ("Introduction to Blue Section", 90, 90, "prose"),
    ("How to Use the Orange Guides", 150, 151, "prose"),
    ("General First Aid", 152, 152, "prose"),
    ("Introduction to Green Tables", 282, 284, "prose"),
    ("Protective Actions", 285, 285, "prose"),
    ("Protective Action Decision Factors to Consider", 286, 287, "prose"),
    ("Background on Table 1 - Initial Isolation and Protective Action Distances", 288, 289, "prose"),
    ("How to Use Table 1 - Initial Isolation and Protective Action Distances", 290, 291, "prose"),
    ("How to Use Table 3 - Large Spills of Six Common TIH Gases", 342, 342, "rows"),
    ("ERG2024 User's Guide", 346, 351, "prose"),
    ("Protective Clothing", 352, 353, "prose"),
    ("Decontamination", 354, 354, "prose"),
    ("Fire and Spill Control", 355, 357, "prose"),
    ("Considerations for Lithium Battery and Electric Vehicle (EV) Fires", 358, 358, "prose"),
    ("BLEVE and Heat Induced Tear", 359, 359, "prose"),
    ("BLEVE - Safety Precautions", 360, 360, "prose"),
    ("Criminal or Terrorist Use of Chemical, Biological and Radiological Agents", 362, 368, "prose"),
    ("Improvised Explosive Device (IED) Safe Stand-Off Distance", 369, 370, "prose"),
    ("Glossary", 371, 381, "prose"),
]

PAGE_LABEL_RE = re.compile(r"^Page (\d+)$")
FURNITURE = {"ERG 2024", "NOTES", "TABLE 1", "TABLE 2", "TABLE 3"}
UN_RE = re.compile(r"^(\d{4}|—\s*—|—)\s+(\d{3}P?)(?:\s+(.+))?$")


def printed_label(doc: pymupdf.Document, pno: int) -> str:
    """The 'Page N' label printed on a page, or '' if the page has none."""
    for ln in doc[pno - 1].get_text().splitlines():
        m = PAGE_LABEL_RE.match(ln.strip())
        if m:
            return m.group(1)
    return ""


def body_lines(doc, pno) -> list[Line]:
    # single capital letters are the glossary's alphabet dividers
    return [
        ln for ln in page_lines(doc[pno - 1])
        if not PAGE_LABEL_RE.match(ln.text) and ln.text not in FURNITURE and not re.fullmatch(r"[A-Z]", ln.text)
    ]


# ---------------------------------------------------------------- yellow index

def parse_yellow_index(doc) -> dict[str, list[tuple[str, str]]]:
    """guide number (e.g. '124', '116P' folded to '116') -> [(UN id, material name)]."""
    by_guide = defaultdict(list)
    skip = {"ID", "No.", "Guide", "Name of Material"}
    for pno in YELLOW_PAGES:
        cur = None
        for raw in doc[pno - 1].get_text().splitlines():
            line = clean(raw)
            if not line or line in skip or PAGE_LABEL_RE.match(line):
                continue
            m = UN_RE.match(line)
            if m:
                if cur:
                    by_guide[cur[1]].append((cur[0], cur[2].strip()))
                un = m.group(1) if m.group(1).isdigit() else ""
                cur = [un, m.group(2).rstrip("P"), m.group(3) or ""]
            elif cur:
                cur[2] = f"{cur[2]} {line}" if not cur[2].endswith("-") else cur[2] + line
        if cur:
            by_guide[cur[1]].append((cur[0], cur[2].strip()))
    return by_guide


def guide_keywords(num: str, title: str, index) -> list[str]:
    kws = [f"Guide {num}", title]
    for un, name in index.get(num, []):
        if un:
            kws.append(f"UN{un}")
        kws.append(name)
    return kws


# ---------------------------------------------------------------- orange guides

def parse_guides(doc, index) -> list[dict]:
    chunks = []
    for pno in GUIDE_PAGES:
        lines = body_lines(doc, pno)
        num = next((ln.text for ln in lines if ln.size >= 15 and ln.text[:3].isdigit()), None)
        if not num:
            continue
        num = num.rstrip("P")
        title = " ".join(ln.text for ln in lines if 10.5 <= ln.size < 12 and ln.y < 50)
        body = [
            ln for ln in lines
            if ln.size < 10.5 or ln.size == 10.0 and "Helvetica" in ln.font
        ]
        # drop footnotes (ERAP note, compatibility-group note) below the body
        body = [ln for ln in body if not (ln.y > 470 and ln.size <= 8.2) and ln.text != "Guides (orange section)"]
        body = [ln for ln in body if not ("AvantGarde" in ln.font and ln.y > 400)]
        if not body:
            continue
        base_x = min(ln.x for ln in body if ln.size < 10)

        def is_heading(ln: Line) -> bool:
            return ln.bold and ln.x <= base_x + 3 and not ln.text.startswith("•")

        # split the page at its level-1 headings (size 10 bold)
        parts, cur = [], None
        for ln in body:
            if ln.size >= 9.9 and ln.bold:
                cur = [ln.text, []]
                parts.append(cur)
            elif cur:
                cur[1].append(ln)
        for heading, part_lines in parts:
            paras = reflow(part_lines, is_heading)
            text = "\n".join(t for _, t, _ in paras)
            part = title_case(heading)
            chunks.append(make_chunk(
                cid=f"{PREFIX}_guide{num}_{slug(part)}",
                manual=MANUAL,
                section=f"Guide {num} – {title} – {part}",
                page=pno,
                printed_page=printed_label(doc, pno),
                # the guide banner printed on the page keeps identical boilerplate
                # (many guides share the same PUBLIC SAFETY text) distinguishable
                text=f"GUIDE {num} – {title}\n{heading}\n{text}",
                keywords=guide_keywords(num, title, index) + [part],
                kind="guide",
            ))
    return chunks


# ---------------------------------------------------------------- rotated tables

VALUE_RE = re.compile(r"\(?\d[\d.,]*\+?\s*(?:km|mi|m|ft)\)?")


def rotated_lines(page, cells=False):
    """Lines of a page printed sideways: (row coordinate, column coordinate, text).

    With cells=True, distance values ("30 m", "(0.1 mi)") are cut out as separate cells
    placed at their centre, because PyMuPDF sometimes merges several adjacent table cells
    into one line when a value overflows its column.
    """
    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            if tuple(round(v) for v in line["dir"]) != (0, -1):
                continue
            chars = [ch for s in line["spans"] for ch in s["chars"]]
            raw = "".join(ch["c"] for ch in chars)
            r = line["bbox"][0]
            if not cells:
                text = clean(raw)
                if text:
                    out.append((r, line["bbox"][1], text))
                continue
            taken = [False] * len(chars)
            for m in VALUE_RE.finditer(raw):
                ink = chars[m.start():m.end()]
                # sideways text runs toward smaller y: the cell spans [end y, start y]
                centre = (min(ch["bbox"][1] for ch in ink) + max(ch["bbox"][3] for ch in ink)) / 2
                out.append((r, centre, clean(m.group())))
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
            rest = "".join(ch["c"] if not t else "\0" for ch, t in zip(chars, taken))
            for m in re.finditer(r"[^\0]+", rest):
                text = clean(m.group())
                ink = [ch for ch in chars[m.start():m.end()] if not ch["c"].isspace()]
                if text and ink:
                    out.append((r, min(ch["bbox"][1] for ch in ink), text))
    return out


# column bands of Table 1, as the page's y coordinate (the table is printed sideways)
T1_COLS = [
    ("id", 495, 999), ("guide", 470, 495), ("name", 372, 470),
    ("s_iso", 315, 372), ("s_day", 258, 315), ("s_night", 198, 258),
    ("l_iso", 140, 198), ("l_day", 72, 140), ("l_night", -99, 72),
]
DIST_RE = re.compile(r"\d")


def t1_col(c):
    return next(name for name, lo, hi in T1_COLS if lo <= c < hi)


def parse_table1(doc) -> list[dict]:
    chunks = []
    for pno in TABLE1_PAGES:
        lines = rotated_lines(doc[pno - 1], cells=True)
        # the rotated column header sits above the first real ID number
        id_rs = sorted(r for r, c, t in lines if c >= 495 and re.fullmatch(r"\d{4}", t))
        if not id_rs:
            continue
        lines = [(r, c, t) for r, c, t in lines if r >= id_rs[0] - 6 and not t.startswith(('"+"', "TABLE"))]
        # a row's distance cells sit on one visual line, vertically centred in the row
        anchors = sorted({round(r) for r, c, t in lines if c < 372 and (DIST_RE.search(t) or "Refer" in t)})
        rows_r = []
        for r in anchors:
            if not rows_r or r - rows_r[-1] > 4:
                rows_r.append(r)
        if not rows_r:
            continue
        rows = {r: defaultdict(list) for r in rows_r}
        # each ID line starts an entry and belongs to the nearest distance row;
        # guide and name lines belong to the entry whose ID line is at or above them
        id_row = {r: min(rows_r, key=lambda rr: abs(rr - r)) for r in id_rs}
        # reading order of sideways text: row by row, then down the page's y axis
        for r, c, t in sorted(lines, key=lambda x: (x[0], -x[1])):
            col = t1_col(c)
            if col in ("id", "guide", "name"):
                entry = max((ir for ir in id_rs if ir <= r + 2), default=id_rs[0])
                owner = id_row[entry]
            else:
                owner = min(rows_r, key=lambda rr: abs(rr - r))
            rows[owner][col].append((r, t))
        for r in rows_r:
            row = rows[r]
            if not row["id"] or not row["name"]:
                continue
            chunks.extend(table1_row_chunks(doc, pno, row))
    # an ID can have several rows (land / water spills, n.o.s. zones): number them
    seen = defaultdict(int)
    for c in chunks:
        un = c["id"]
        seen[un] += 1
        c["id"] = f"{un}_{seen[un]}"
    return chunks


def table1_row_chunks(doc, pno, row) -> list[dict]:
    ids = [(r, t) for r, t in row["id"]]
    guides = [t for _, t in row["guide"]]
    # pair every name line with the nearest ID line at or above it
    names = defaultdict(list)
    for r, t in row["name"]:
        owner = max((i for i, (ir, _) in enumerate(ids) if ir <= r + 2), default=0)
        names[owner].append(t)
    entries = []
    for i, (_, un) in enumerate(ids):
        name = " ".join(names.get(i, []))
        name = re.sub(r"(\w)- (\w)", r"\1-\2", name)
        entries.append((un, guides[i] if i < len(guides) else guides[-1], name))

    def cell(key):
        return " ".join(t for _, t in row[key])

    small = (f"First ISOLATE in all Directions: {cell('s_iso')}; "
             f"Then PROTECT persons Downwind during DAY: {cell('s_day')}, NIGHT: {cell('s_night')}")
    large_cells = cell("l_iso") + cell("l_day") + cell("l_night")
    if "Refer to Table 3" in large_cells:
        large = "Refer to Table 3"
    else:
        large = (f"First ISOLATE in all Directions: {cell('l_iso')}; "
                 f"Then PROTECT persons Downwind during DAY: {cell('l_day')}, NIGHT: {cell('l_night')}")
    header = "\n".join(f"ID {un} – Guide {g} – {name}" for un, g, name in entries)
    text = (f"{header}\n"
            f"SMALL SPILLS (From a small package or small leak from a large package): {small}\n"
            f"LARGE SPILLS (From a large package or from many small packages): {large}")
    un, g, name = entries[0]
    kws = ["Table 1", "initial isolation", "protective action distance", "evacuation distance"]
    for e_un, e_g, e_name in entries:
        kws += [f"UN{e_un}", e_name, f"Guide {e_g.rstrip('P')}"]
    return [make_chunk(
        cid=f"{PREFIX}_t1_{un}",
        manual=MANUAL,
        section=f"Table 1 – Initial Isolation and Protective Action Distances – {name} (UN{un})",
        page=pno,
        printed_page=printed_label(doc, pno),
        text=text,
        keywords=kws,
        kind="table1",
    )]


T3_VALUE_COLS = [  # (label, value column, unit column)
    ("First ISOLATE in all Directions", 421, 389),
    ("DAY Low wind (< 6 mph = < 10 km/h)", 360, 329),
    ("DAY Moderate wind (6-12 mph = 10 - 20 km/h)", 304, 273),
    ("DAY High wind (> 12 mph = > 20 km/h)", 247, 217),
    ("NIGHT Low wind (< 6 mph = < 10 km/h)", 190, 159),
    ("NIGHT Moderate wind (6-12 mph = 10 - 20 km/h)", 133, 103),
    ("NIGHT High wind (> 12 mph = > 20 km/h)", 76, 45),
]


def parse_table3(doc) -> list[dict]:
    blocks = []  # [pno, [headings], [(r, cells)], {r: label parts}]
    for pno in TABLE3_PAGES:
        lines = sorted(rotated_lines(doc[pno - 1]))
        cur = None
        for r, c, t in lines:
            if re.match(r"^UN\d{4} .+: Large Spills$", t):
                if cur is None or cur[2]:
                    cur = [pno, [], [], defaultdict(list)]
                    blocks.append(cur)
                cur[1].append(t)
            elif cur is None:
                continue
            elif c >= 435 and t not in ("TRANSPORT", "CONTAINER"):
                cur[3][round(r)].append(t)
            elif c < 435 and re.fullmatch(r"\(?[\d.,]+\+?\)?", t):
                row = next((rw for rw in cur[2] if abs(rw[0] - r) <= 3), None)
                if row is None:
                    row = [r, {}]
                    cur[2].append(row)
                row[1][round(c)] = t
    chunks = []
    for pno, headings, rows, labels in blocks:
        out = list(headings)
        for r, cells in rows:
            label = " ".join(
                t for lr in sorted(labels) if abs(lr - r) <= 10 for t in labels[lr]
            )
            vals = []
            for name, vc, uc in T3_VALUE_COLS:
                v = next((t for c, t in cells.items() if abs(c - vc) <= 6), "")
                u = next((t for c, t in cells.items() if abs(c - uc) <= 6), "").strip("()")
                big, small = ("m", "ft") if vc == 421 else ("km", "mi")
                vals.append(f"{name}: {v} {big} ({u} {small})")
            out.append(f"{label} – " + "; ".join(vals))
        materials = [re.match(r"^(UN\d{4}) (.+): Large Spills$", h).groups() for h in headings]
        un, name = materials[0]
        kws = ["Table 3", "large spill", "rail tank car", "highway tank truck", "wind speed",
               "protective action distance", "evacuation distance"]
        for m_un, m_name in materials:
            kws += [m_un, m_name]
        chunks.append(make_chunk(
            cid=f"{PREFIX}_t3_{un[2:]}",
            manual=MANUAL,
            section=f"Table 3 – Large Spill Distances by Container – {' / '.join(n for _, n in materials)}",
            page=pno,
            printed_page=printed_label(doc, pno),
            text="\n".join(out),
            keywords=kws,
            kind="table3",
        ))
    return chunks


TIH_GASES = r"(?:HCl|HBr|HF|HI|HCN|H2S|NH3|NO2|PH3|SO2|Br2|Cl2)"
T2_ROW_RE = re.compile(rf"^(\d{{4}}) (\d{{3}}P?) (.*?)((?: {TIH_GASES})*)$")
T2_TAIL_RE = re.compile(rf"^(.*?)((?: ?{TIH_GASES})+)$")
T2_TITLE = "Table 2 – Water-Reactive Materials Which Produce Toxic Gases"
T2_HEADER = "ID No. | Guide No. | Name of Material | TIH Gas(es) Produced"
T2_ROWS_PER_CHUNK = 20
T2_PAGE_HEADER = ("TABLE 2", "Materials Which Produce", "(PIH in the US) Gas", "ID Guide", "No. No.")


def parse_table2(doc) -> list[dict]:
    entries, legend = [], []
    for pno in TABLE2_PAGES:
        for kind, text, page in visual_rows(body_lines(doc, pno)):
            m = T2_ROW_RE.match(text)
            if m:
                entries.append({"un": m.group(1), "guide": m.group(2), "name": m.group(3).strip(),
                                "gases": m.group(4).split(), "page": page})
            elif entries and entries[-1]["page"] == page and text[:1].islower() or text.startswith("fissile"):
                # wrapped material name; a trailing gas symbol can sit on the last line
                tail = T2_TAIL_RE.match(text)
                name_part, gases = (tail.group(1), tail.group(2).split()) if tail else (text, [])
                entries[-1]["name"] = f"{entries[-1]['name']} {name_part.strip()}".strip()
                entries[-1]["gases"] += gases
            elif pno == TABLE2_PAGES[0] and text not in legend and not text.startswith(T2_PAGE_HEADER):
                legend.append(text)

    printed = {p: printed_label(doc, p) for p in TABLE2_PAGES}
    chunks = [make_chunk(
        cid=f"{PREFIX}_t2_legend",
        manual=MANUAL,
        section=f"{T2_TITLE} – Chemical Symbols for TIH Gases",
        page=TABLE2_PAGES[0],
        printed_page=printed[TABLE2_PAGES[0]],
        text="\n".join(legend),
        keywords=["Table 2", "water-reactive", "TIH", "toxic gas", "spilled in water"],
        kind="table2",
    )]
    for start in range(0, len(entries), T2_ROWS_PER_CHUNK):
        group = entries[start:start + T2_ROWS_PER_CHUNK]
        rows = [f"{e['un']} | {e['guide']} | {e['name']} | {' '.join(e['gases'])}" for e in group]
        kws = ["Table 2", "water-reactive", "spilled in water", "toxic gas"]
        for e in group:
            kws += [f"UN{e['un']}", e["name"]]
        chunks.append(make_chunk(
            cid=f"{PREFIX}_t2_{group[0]['un']}_{group[-1]['un']}",
            manual=MANUAL,
            section=f"{T2_TITLE} (ID {group[0]['un']}–{group[-1]['un']})",
            page=group[0]["page"],
            printed_page=printed[group[0]["page"]],
            text="Use this list only when material is spilled in water.\n" + T2_HEADER + "\n" + "\n".join(rows),
            keywords=kws,
            kind="table2",
        ))
    return chunks


# The BLEVE propane-tank table (page 361) is printed sideways: each tank size is a column of
# values that reads bottom-to-top in this fixed order. (property, printed units, values per cell)
BLEVE_PAGE = 361
BLEVE_PROPERTIES = [
    ("Capacity", "Litres (Gallons)", 2),
    ("Diameter", "Meters (Feet)", 2),
    ("Length", "Meters (Feet)", 2),
    ("Propane Mass", "Kilograms (Pounds)", 2),
    ("Minimum time to failure for severe torch", "Minutes", 1),
    ("Approximate time to empty for engulfing fire", "Minutes", 1),
    ("Fireball radius", "Meters (Feet)", 2),
    ("Emergency response distance", "Meters (Feet)", 2),
    ("Minimum evacuation distance", "Meters (Feet)", 2),
    ("Preferred evacuation distance", "Meters (Feet)", 2),
    ("Cooling water flow rate", "Litres/min (USgal/min)", 2),
]
NUM_RE = re.compile(r"^\(?[\d.,]+\)?$")


def parse_bleve_table(doc) -> list[dict]:
    page = doc[BLEVE_PAGE - 1]
    notes = [ln.text for ln in sorted(body_lines(doc, BLEVE_PAGE), key=lambda l: l.y)
             if not ln.bold and len(ln.text) > 40]
    # value cells are sideways text; one PDF line can run through several cells of a column
    # ("(5627) 2200 (7218)"), so lines are ordered by where they start and split into numbers
    cells = []  # (column x, start y, [tokens])
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = clean("".join(s["text"] for s in line["spans"]))
            x0, _, _, y1 = line["bbox"]
            tokens = re.findall(r"\(?[\d.,]+\)?", text)
            if x0 > 130 and tokens and " ".join(tokens) == text:
                cells.append((x0, y1, tokens))
    columns: list[list] = []
    for cell in sorted(cells):
        if columns and cell[0] - columns[-1][0][0] < 8:
            columns[-1].append(cell)
        else:
            columns.append([cell])
    need = sum(n for _, _, n in BLEVE_PROPERTIES)
    rows = []
    for col in columns:
        tokens = [t for _, _, toks in sorted(col, key=lambda c: -c[1]) for t in toks]
        if len(tokens) != need:
            raise ValueError(f"BLEVE table: column at x={col[0][0]:.0f} has {len(tokens)} values, expected {need}")
        cells, i = [], 0
        for name, units, n in BLEVE_PROPERTIES:
            cell = tokens[i:i + n]
            i += n
            if n == 2 and not cell[1].startswith("("):
                # cooling water prints litres/min and USgal/min without parentheses
                cell = [cell[0], f"({cell[1]})"]
            cells.append(f"{name} {units}: {' '.join(cell)}")
        rows.append((tokens[0], "; ".join(cells)))
    caution, closing = (notes[0], notes[1:]) if notes else ("", [])
    chunks = []
    # three tank sizes per chunk keeps each passage readable (~230 words)
    for start in range(0, len(rows), 3):
        group = rows[start:start + 3]
        text = "\n".join(([caution] if caution else []) + [r for _, r in group] + closing)
        chunks.append(make_chunk(
            cid=f"{PREFIX}_ref_bleve_propane_tank_table_{start // 3 + 1}",
            manual=MANUAL,
            section=f"BLEVE - Safety Precautions – Propane Tank Table ({group[0][0]}–{group[-1][0]} Litres)",
            page=BLEVE_PAGE,
            printed_page=printed_label(doc, BLEVE_PAGE),
            text=text,
            keywords=["BLEVE", "propane", "LPG", "tank", "fireball", "evacuation distance",
                      "time to failure", "cooling water", "boiling liquid expanding vapor explosion"]
                     + [f"{cap} litres" for cap, _ in group],
            kind="table_bleve",
        ))
    return chunks


# ---------------------------------------------------------------- reference prose

def parse_reference(doc) -> list[dict]:
    chunks = []
    for title, first, last, layout in REFERENCE_SECTIONS:
        lines = []
        for pno in range(first, last + 1):
            lines.extend(body_lines(doc, pno))
        if not lines:
            continue
        body_size = sorted(ln.size for ln in lines)[len(lines) // 2]

        def is_heading(ln: Line) -> bool:
            return (ln.bold and len(ln.text) < 90 and ln.size >= body_size
                    and not ln.text.startswith(("•", "–", "-")) and not ln.text.endswith(","))

        paras = visual_rows(lines, merge_continuations=True) if layout == "rows" else reflow(lines, is_heading)
        # the section title (often printed over two lines, repeated per page) is
        # redundant with `section`
        title_slug = slug(title, 200)
        paras = [p for p in paras if not (p[0] == "heading" and slug(p[1], 200) in title_slug)]
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
    index = parse_yellow_index(doc)
    return (parse_guides(doc, index) + parse_table1(doc) + parse_table2(doc)
            + parse_table3(doc) + parse_bleve_table(doc) + parse_reference(doc))
