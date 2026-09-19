"""PDF manuals -> data/chunks.json

Run once (no network needed):  python ingest.py

Every chunk has the shape
    {"id", "manual", "section", "page", "printed_page", "kind", "text", "keywords", "also_at",
     "protocol_id", "protocol"}
where `page` is the 1-based PDF page (what a PDF viewer jumps to) and `printed_page`
is the page label printed on that page ("178", "xxviii"), so a citation can be checked
against either the PDF or the paper copy. `protocol_id` groups the chunks of one unit
(a WV protocol, an ERG guide, a NIOSH chemical, a reference section): search matches
individual chunks but answers with the whole protocol.

The run fails loudly if any chunk breaks the invariants the search side relies on
(unique ids, no duplicate passages, required fields present).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from parsers import erg, generic, niosh, wv

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent / "Documents"
OUT = ROOT / "data" / "chunks.json"

# Manuals with a dedicated parser. Any other PDF dropped into Documents/ is picked up
# by the generic (bookmark/heading based) parser.
MANUALS = {
    "2024EmergencyResponseGuide.pdf": erg.parse,
    "CDCPocketGuide.pdf": niosh.parse,
    "WestVirginiaEmergencyGuide.pdf": wv.parse,
}

REQUIRED = {"id": str, "manual": str, "section": str, "page": int,
            "printed_page": str, "kind": str, "text": str, "keywords": list, "also_at": list,
            "protocol_id": str, "protocol": str}
MIN_WORDS = 3
MAX_PROTOCOL_WORDS = 600   # longer reference sections are answered subsection by subsection


def merge_repeats(chunks: list[dict]) -> list[dict]:
    """A passage printed word-for-word in two places of the same manual (e.g. the ERG's
    yellow- and blue-section introductions) is kept once; the other locations are recorded
    in `also_at` so the citation can still point at every page."""
    first: dict[tuple, dict] = {}
    out = []
    for c in chunks:
        c.setdefault("also_at", [])
        key = (c["manual"], c["text"])
        if key in first:
            keep = first[key]
            keep["also_at"].append({"section": c["section"], "page": c["page"], "printed_page": c["printed_page"]})
            keep["keywords"] += [k for k in c["keywords"] if k not in keep["keywords"]]
            continue
        first[key] = c
        out.append(c)
    return out


DANGLING_WORDS = {"a", "an", "and", "by", "for", "in", "is", "of", "on", "or", "the", "to", "with"}


def is_fragment_label(sub: str) -> bool:
    """A 'subsection' that is really a stray bold line: a sentence tail ("dying people or
    animals"), a sentence ("Refer to GUIDE 152."), a line cut mid-sentence ("Unusual
    numbers of dying or"), a dangling "Class 1 -", a parenthetical, a bare number."""
    sub = sub.strip()
    words = sub.split()
    return (not words or sub[0].islower() or sub[0] == "(" or re.fullmatch(r"[\d\W]+", sub) is not None
            or sub.endswith((".", " -", ",", "–")) or re.search(r"[a-z]{2}\. [A-Z]", sub) is not None
            or words[-1].lower() in DANGLING_WORDS)


def tidy_section_labels(chunks: list[dict]) -> None:
    """Replace a fragment subsection label with the label of the chunk before it in the
    same protocol, so the passage is cited under the heading it is actually printed under.
    Must run before split_long_references and number_parts."""
    prev: dict[str, str] = {}
    for c in chunks:
        parent = c["protocol"]
        if c["section"].startswith(parent + " – "):
            sub = c["section"][len(parent) + 3:]
            if is_fragment_label(sub):
                c["section"] = prev.get(c["protocol_id"], parent)
                c["keywords"] = [k for k in c["keywords"] if k != sub]
        prev[c["protocol_id"]] = c["section"]


def split_long_references(chunks: list[dict]) -> None:
    """A reference section longer than MAX_PROTOCOL_WORDS (the ERG glossary, NIOSH
    Appendix E, ...) is too broad to hand back as one answer, so it is regrouped into
    its subsections: consecutive chunks that carry the same section label. A label that
    recurs in several separate runs is a repeated table header ("Required Respirator"),
    not a subsection, so those chunks stay with the subsection they are printed under.
    Must run before number_parts, while section labels are still unnumbered."""
    groups: dict[str, list[dict]] = {}
    for c in chunks:
        groups.setdefault(c["protocol_id"], []).append(c)
    for pid, parts in groups.items():
        if parts[0]["kind"] != "reference" or sum(len(c["text"].split()) for c in parts) <= MAX_PROTOCOL_WORDS:
            continue
        runs = Counter(c["section"] for i, c in enumerate(parts) if i == 0 or parts[i - 1]["section"] != c["section"])
        current = None
        for c in parts:
            repeated_header = runs[c["section"]] > 1 and current is not None
            if current is None or (c["section"] != current["protocol"] and not repeated_header):
                # named after its first chunk, whose id is already unique and stable
                current = {"protocol_id": c["id"], "protocol": c["section"]}
            c.update(current)


def number_parts(chunks: list[dict]) -> None:
    """A section long enough to span several chunks gets "(1/3)", "(2/3)", ... so every
    (manual, section) label is unique and the reader knows the passage continues."""
    groups: dict[tuple, list[dict]] = {}
    for c in chunks:
        groups.setdefault((c["manual"], c["section"]), []).append(c)
    for parts in groups.values():
        if len(parts) > 1:
            for i, c in enumerate(parts, start=1):
                c["section"] = f"{c['section']} ({i}/{len(parts)})"


def validate(chunks: list[dict]) -> list[str]:
    problems = []
    for c in chunks:
        for key, typ in REQUIRED.items():
            if not isinstance(c.get(key), typ):
                problems.append(f"{c.get('id')}: field {key!r} missing or not {typ.__name__}")
        if len(c.get("text", "").split()) < MIN_WORDS:
            problems.append(f"{c['id']}: text has fewer than {MIN_WORDS} words")
        if not c.get("section", "").strip():
            problems.append(f"{c['id']}: empty section")
    for cid, n in Counter(c["id"] for c in chunks).items():
        if n > 1:
            problems.append(f"duplicate id {cid} ({n}x)")
    for (manual, text), n in Counter((c["manual"], c["text"]) for c in chunks).items():
        if n > 1:
            problems.append(f"duplicate passage in {manual} ({n}x): {text[:60]!r}")
    for (manual, section), n in Counter((c["manual"], c["section"]) for c in chunks).items():
        if n > 1:
            problems.append(f"duplicate section label in {manual} ({n}x): {section!r}")
    # a protocol_id names exactly one protocol in one manual, and no two protocols share a name
    names: dict[str, set] = {}
    for c in chunks:
        names.setdefault(c["protocol_id"], set()).add((c["manual"], c["protocol"]))
    for pid, labels in names.items():
        if len(labels) > 1:
            problems.append(f"protocol {pid} has several names: {sorted(labels)}")
    for (manual, name), n in Counter(next(iter(v)) for v in names.values()).items():
        if n > 1:
            problems.append(f"duplicate protocol name in {manual} ({n}x): {name!r}")
    return problems


def main() -> int:
    pdfs = sorted(DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found in {DOCS}")
        return 1
    chunks = []
    for pdf in pdfs:
        parser = MANUALS.get(pdf.name, generic.parse)
        got = parser(pdf)
        kinds = Counter(c["kind"] for c in got)
        print(f"{pdf.name}: {len(got)} chunks  " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items())))
        chunks.extend(got)

    before = len(chunks)
    chunks = merge_repeats(chunks)
    if before != len(chunks):
        print(f"merged {before - len(chunks)} passage(s) printed more than once in the same manual")
    tidy_section_labels(chunks)
    split_long_references(chunks)
    number_parts(chunks)

    problems = validate(chunks)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:50]:
            print("  " + p)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    words = [len(c["text"].split()) for c in chunks]
    protocols = len({c["protocol_id"] for c in chunks})
    print(f"\nwrote {len(chunks)} chunks ({protocols} protocols) from {len(pdfs)} manuals to {OUT.relative_to(ROOT)}"
          f"  (words per chunk: min {min(words)}, median {sorted(words)[len(words) // 2]}, max {max(words)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
