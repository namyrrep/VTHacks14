# BLACKOUT — Backend (Search Pipeline)

The backend turns protocol manual PDFs into a local search index, exposes `search()`, and serves that plus the UI over a local HTTP server (`server.py`). Once the index files exist, **nothing in the backend touches the network.**

No generative model sits in the answer path. The backend only returns verbatim passages from the manuals, with citations, and abstains when confidence is low.

## Pipeline

```
Documents/*.pdf
   │  ingest.py   PDF → one chunk per protocol / chemical / table row (run once)
   ▼
data/chunks.json  [{id, manual, section, page, printed_page, kind, text, keywords, also_at, protocol_id, protocol}, ...]
   │  index.py    embed every chunk (run once)
   ▼
data/index.npy    one normalized embedding per chunk, same order as chunks.json
   │
   ▼
search.py         question → embed → score chunks → pool by protocol → top 3 → threshold → one-line answer each
   │
   ▼
server.py         localhost HTTP: the page, /api/search, /api/protocol, /manuals/*.pdf
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Runs each manual's parser, merges passages printed twice, repairs fragment section labels, splits oversized reference sections into subsection protocols, numbers multi-part sections, checks integrity, writes `data/chunks.json`. Fails if any chunk ID, passage, section label, or protocol name is duplicated. |
| `parsers/erg.py` | ERG 2024: orange guides, Tables 1–3, BLEVE table, reference sections |
| `parsers/niosh.py` | NIOSH Pocket Guide: 677 chemical records, code tables 1–6, appendices A–F |
| `parsers/wv.py` | West Virginia EMS Protocols: 88 protocols, 41 medications, appendices |
| `parsers/generic.py` | Fallback for any other PDF dropped into `Documents/` (splits on bookmarks and headings) |
| `parsers/common.py` | Shared line extraction, paragraph reflow, chunk packing |
| `index.py` | Embeds every chunk with `all-MiniLM-L6-v2`, writes `data/index.npy` and `data/index_meta.json` |
| `search.py` | Loads the data files and implements `search()` and `get_protocol()`. Refuses to run on a stale index. |
| `server.py` | Standard-library HTTP server for the UI. No dependency beyond what search already needs, and it binds to localhost only. |
| `eval_search.py` | Checks corpus consistency/uniqueness and every `search()` invariant against 45 test questions, and reports the threshold margin. Exits non-zero on any problem. |
| `data/chunks.json` | Generated, **committed** so a fresh clone runs with no build step |
| `data/index.npy` | Generated, **committed** for the same reason |
| `data/index_meta.json` | Model name, vector size, and a hash of everything embedded (IDs, sections, text, keywords), so `search.py` can detect a stale index |

## Manuals loaded

1,825 chunks in 1,280 protocols.

| Manual | Source PDF | Chunks | What one chunk is |
|---|---|---|---|
| Emergency Response Guidebook 2024 | `Documents/2024EmergencyResponseGuide.pdf` | 719 | One part of an orange guide (Potential Hazards / Public Safety / Emergency Response); one Table 1 row (isolation and protective-action distances for a UN number); a block of Table 2 or Table 3; or a reference section |
| NIOSH Pocket Guide to Chemical Hazards | `Documents/CDCPocketGuide.pdf` | 776 | One chemical record (all 677), or part of the code tables and appendices |
| West Virginia Statewide EMS Protocols | `Documents/WestVirginiaEmergencyGuide.pdf` | 335 | Part of one protocol (e.g. `T008 – Burns`), part of one medication formulary entry, or part of an appendix |

## Chunking rules

- **One chunk per protocol, capped at about 180 words.** A longer protocol is split only at a heading or paragraph boundary, and each part carries the protocol's section name. NIOSH chemical records and ERG guide parts stay whole even when longer. `index.py` embeds long chunks as overlapping windows and averages them, so each chunk is still one row in `index.npy`.
- **One protocol per unit:** every chunk carries `protocol_id` and `protocol`. All chunks of one WV protocol, one ERG guide (its three printed parts), one Table 1 ID number, one NIOSH chemical, or one reference subsection share a `protocol_id`. Search matches chunks but answers with whole protocols. Reference sections longer than 600 words (the ERG glossary, NIOSH Appendix E, ...) are split into their subsections; Table 2 and the BLEVE table (with its safety precautions) are one protocol each.
- **Clean labels:** a stray bold line picked up as a subsection ("Refer to GUIDE 152.", "dying people or animals", "Class 1 -") is replaced by the heading the passage is actually printed under.
- **Unique:** `ingest.py` rejects duplicate IDs, passages, or section labels. A passage the manual prints twice (e.g. the ERG yellow and blue section introductions) is kept once, and the second location goes in `also_at`.
- **Consistent:** every chunk has every field, and repeated runs produce a byte-identical `chunks.json`.

## Setup

Do this **while on wifi**. The embedding model (~90 MB) downloads on first use and is cached locally after that.

```bash
cd backend
pip install -r requirements.txt   # pymupdf, sentence-transformers, numpy, rank_bm25
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Rebuild the index only if the manuals change:

```bash
python ingest.py   # about 1 minute, no network
python index.py    # about 1 minute on CPU; no network once the model is cached
```

To add a manual, drop the PDF into `Documents/` and rerun both scripts. It goes through `parsers/generic.py` unless you add a dedicated parser to `MANUALS` in `ingest.py`.

## Serving the UI

```bash
python server.py                                  # http://127.0.0.1:8000, opens a browser
python server.py --port 9000 --no-browser
```

The index and the model load before the first request, so no question waits on them.

| Route | Returns |
|---|---|
| `GET /` | `frontend/index.html`, and any other file beside it |
| `GET /api/stats` | `{manuals: [{manual, short, pdf, chunks, protocols}], chunks, protocols, threshold, model}` — the status bar's counts, and the short manual labels (Hazmat / Chemical / Medical) the UI cites |
| `POST /api/search` | `{"question": str, "k": int}` → `search()` |
| `GET /api/protocol?id=` | `get_protocol()` |
| `GET /manuals/<file>.pdf` | the manual, so `#page=N` opens a citation |

Static files are resolved and checked to be inside their one directory, so a request cannot walk out of it. Adding a manual means adding its short label and PDF filename to `MANUALS` in `server.py`; without an entry it still works, cited by its full name and with no PDF link.

## The contract

Agreed with the UI side. Do not change the shape without telling the other person. The fields after `score` were added with the search implementation; the original five are unchanged.

```python
def search(question: str, k: int = 3) -> dict:
    """
    {
      "confident": bool,       # False -> UI shows the refusal state
      "top_score": float,      # 0.0-1.0, score of the best protocol
      "results": [             # at most k, one per protocol, each scoring >= THRESHOLD
        {
          "text": str,         # best-matching passage, VERBATIM from the manual
          "manual": str,       # "NIOSH Pocket Guide to Chemical Hazards"
          "section": str,      # "Hydrogen sulfide"
          "page": int,         # 200 (PDF page of that passage)
          "score": float,      # 0.0-1.0
          "answer": str,       # ONE line of the protocol answering the question, verbatim
          "answer_page": int,  # PDF page that line is on
          "answer_chunk_id": str,  # the part that line came from; UI marks it in the protocol view
          "chunk_id": str,     # id of the passage in data/chunks.json
          "protocol_id": str,  # "wv_t008"; pass to get_protocol()
          "protocol": str,     # "T008 – Burns"
          "chunk": {           # the whole protocol: every part, in reading order
            "protocol_id", "protocol", "manual", "kind",
            "page", "pages", "printed_pages", "parts": [chunk ids], "text",
            "passages": [     # the same text, split back into parts so each can be cited
              {"chunk_id", "section", "page", "printed_page", "text"}, ...
            ]
          }
        }, ...
      ]
    }
    """

def get_protocol(protocol_id: str) -> dict | None:  # same shape as "chunk" above
```

When `confident` is `False`, `results` is empty. Results below the threshold are dropped even when the top one is confident, so there can be fewer than 3.

### How a result is scored

1. **Chunks.** Cosine similarity between the question and each chunk's embedding. Before embedding, bystander wording gets the manuals' clinical terms appended ("passed out" → unconscious, "choking" → airway obstruction; `LAY_TERMS`).
2. **Keywords.** +0.12 when the question names an identifying keyword of the chunk (a chemical, drug, or protocol name; a keyword shared by more than 8 protocols, like "dose", doesn't count). +0.2 for an exact UN number, CAS number, or protocol code, with a floor of 0.6, so "7783-06-4" or "T008" alone is answered. A bare 4-digit number gets only the +0.12 (it's more often a year than a UN number).
3. **Protocols.** A protocol scores as its best chunk, plus up to +0.08 for the share of the question's topic words (the words that aren't the protocol's own name) found in that chunk. The top 3 distinct protocols are kept.
4. **Answer line.** From the protocol's lines (headings, banners, and name-only lines excluded), pick the line that best matches the question semantically, plus a bonus for containing its topic words. A question that only names the protocol ("snake bite", "T008") gets the line most representative of the protocol as a whole. Lines longer than 240 characters are cut at a word boundary and end in "…".

### Chunk shape

```json
{"id": "niosh_hydrogen_sulfide",
 "manual": "NIOSH Pocket Guide to Chemical Hazards",
 "section": "Hydrogen sulfide",
 "page": 200,
 "printed_page": "170",
 "kind": "chemical",
 "text": "Hydrogen sulfide\nFormula: H2S\nCAS#: 7783-06-4\n...",
 "keywords": ["Hydrogen sulfide", "Sewer gas", "CAS 7783-06-4", "UN1053", "ERG Guide 117"],
 "also_at": []}
```

| Field | Meaning |
|---|---|
| `id` | Stable, unique ID (`erg_guide124_public_safety`, `erg_t1_1017_1`, `niosh_hydrogen_sulfide`, `wv_t008_01`) |
| `manual`, `section` | Citation. `section` is unique within a manual; long sections get `(1/3)`, `(2/3)`, ... |
| `page` | PDF page, 1-based. This is the page number the contract's `page` field should carry. |
| `printed_page` | The page label printed on the paper copy (`"178"`, `"xxviii"`). Empty for the WV protocols, which print no page numbers. |
| `kind` | `guide`, `table1`, `table2`, `table3`, `table_bleve`, `chemical`, `protocol`, `medication`, or `reference` |
| `text` | The passage, **verbatim**. The only reshaping: wrapped lines are rejoined, and table cells are laid out as labeled lines using the table's own column headers. |
| `keywords` | Exact-match terms: UN numbers, CAS numbers, synonyms, protocol codes. Used for the exact-match boost. |
| `also_at` | Other places the same passage is printed, as a list of `{section, page, printed_page}` |
| `protocol_id` | The protocol this chunk is part of (`wv_t008`, `erg_guide124`, `erg_t1_1017`, `niosh_hydrogen_sulfide`). Shared by every part of the protocol. |
| `protocol` | The protocol's name (`T008 – Burns`, `Guide 124 – Gases - Toxic and/or Corrosive - Oxidizing`). Unique within a manual. |

## Confidence threshold

`search()` sets `confident = top_score >= THRESHOLD`. `eval_search.py` holds the test set and prints the lowest in-corpus and highest out-of-corpus top score.

- Current threshold: **0.45** (`THRESHOLD` in `search.py`)
- Tuned against: 45 questions. 29 are covered by the manuals (clinical wording, bystander wording, bare identifiers), and all score ≥ 0.508 with an expected protocol in the top 3. 16 are not covered (off-topic, near-domain like "dog bite rabies shots" or "covid vaccine schedule", a year that is also a UN number), and all score ≤ 0.388. The midpoint is 0.448.
- A line ending in ":" ("Common chemicals that cause burns:") introduces what follows instead of answering, so it is never picked as the answer line.
- Known limits: idioms can trip the bystander wording ("this movie is choking me up" scores 0.49 → R001 Airway Management). Casual phrasings with no `LAY_TERMS` entry can land between 0.35 and 0.45 and be refused. Add a `LAY_TERMS` entry, add the question to `eval_search.py`, and rerun it.

## Fallback: BM25

If `sentence-transformers` won't install or the model won't download, switch `search()` to `rank_bm25` keyword search. It keeps the same contract and the same chunks, and the UI doesn't change. Normalize BM25 scores to 0.0–1.0 and re-tune the threshold.

## Quick test

```bash
python search.py hydrogen sulfide exposure   # prints the result, without the full protocol text
python eval_search.py                          # all checks; must end with "0 problem(s)"
```

Turn on airplane mode and run it again. It should return the same output.
