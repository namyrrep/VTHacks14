# BLACKOUT — Backend (Search Pipeline)

The backend turns protocol manual PDFs into a local search index and exposes one function, `search()`, that the Streamlit UI calls. Once the index files exist, **nothing in the backend touches the network.**

No generative model sits in the answer path. The backend only returns verbatim passages from the manuals, with citations, and abstains when confidence is low.

## Pipeline

```
Documents/*.pdf
   │  ingest.py   PDF → one chunk per protocol / chemical / table row (run once)
   ▼
data/chunks.json  [{id, manual, section, page, printed_page, kind, text, keywords, also_at}, ...]
   │  index.py    embed every chunk (run once)
   ▼
data/index.npy    one normalized embedding per chunk, same order as chunks.json
   │
   ▼
search.py         question → embed → cosine similarity → top k → threshold check   (not built yet)
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Runs each manual's parser, merges passages printed twice, numbers multi-part sections, checks integrity, writes `data/chunks.json`. Fails if any chunk ID, passage, or section label is duplicated. |
| `parsers/erg.py` | ERG 2024: orange guides, Tables 1–3, BLEVE table, reference sections |
| `parsers/niosh.py` | NIOSH Pocket Guide: 677 chemical records, code tables 1–6, appendices A–F |
| `parsers/wv.py` | West Virginia EMS Protocols: 88 protocols, 41 medications, appendices |
| `parsers/generic.py` | Fallback for any other PDF dropped into `Documents/` (splits on bookmarks and headings) |
| `parsers/common.py` | Shared line extraction, paragraph reflow, chunk packing |
| `index.py` | Embeds every chunk with `all-MiniLM-L6-v2`, writes `data/index.npy` and `data/index_meta.json` |
| `search.py` | Loads the data files and implements `search()` (**not built yet**) |
| `data/chunks.json` | Generated, **committed** so a fresh clone runs with no build step |
| `data/index.npy` | Generated, **committed** for the same reason |
| `data/index_meta.json` | Model name, vector size, and a hash of the chunk IDs, so `search.py` can detect a stale index |

## Manuals loaded

1,825 chunks in total.

| Manual | Source PDF | Chunks | What one chunk is |
|---|---|---|---|
| Emergency Response Guidebook 2024 | `Documents/2024EmergencyResponseGuide.pdf` | 719 | One part of an orange guide (Potential Hazards / Public Safety / Emergency Response); one Table 1 row (isolation and protective-action distances for a UN number); a block of Table 2 or Table 3; or a reference section |
| NIOSH Pocket Guide to Chemical Hazards | `Documents/CDCPocketGuide.pdf` | 776 | One chemical record (all 677), or part of the code tables and appendices |
| West Virginia Statewide EMS Protocols | `Documents/WestVirginiaEmergencyGuide.pdf` | 335 | Part of one protocol (e.g. `T008 – Burns`), part of one medication formulary entry, or part of an appendix |

## Chunking rules

- **One chunk per protocol, capped at about 180 words.** A longer protocol is split only at a heading or paragraph boundary, and each part carries the protocol's section name. NIOSH chemical records and ERG guide parts stay whole even when longer. `index.py` embeds long chunks as overlapping windows and averages them, so each chunk is still one row in `index.npy`.
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

## The contract

Agreed with the UI side. Do not change the shape without telling the other person.

```python
def search(question: str, k: int = 3) -> dict:
    """
    {
      "confident": bool,       # False -> UI shows the refusal state
      "top_score": float,      # 0.0-1.0
      "results": [
        {
          "text": str,         # passage, VERBATIM from the manual
          "manual": str,       # "NIOSH Pocket Guide"
          "section": str,      # "Hydrogen Sulfide"
          "page": int,         # 172
          "score": float       # 0.0-1.0
        }, ...
      ]
    }
    """
```

When `confident` is `False`, `results` is empty.

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
| `keywords` | Exact-match terms: UN numbers, CAS numbers, synonyms, protocol codes. Meant for BM25 or an exact-match boost. |
| `also_at` | Other places the same passage is printed, as a list of `{section, page, printed_page}` |

## Confidence threshold

`search()` sets `confident = top_score >= THRESHOLD`. Tune the value against a test set of about 10 questions: 5 that the manuals cover and 5 that they don't (for example, "what's the wifi password"). Pick the value that refuses every out-of-corpus question and still answers every in-corpus one.

- Current threshold: `[TBD]`
- Tuned against: `[N]` questions

## Fallback: BM25

If `sentence-transformers` won't install or the model won't download, switch `search()` to `rank_bm25` keyword search. It keeps the same contract and the same chunks, and the UI doesn't change. Normalize BM25 scores to 0.0–1.0 and re-tune the threshold.

## Quick test

```bash
python -c "from search import search; import json; print(json.dumps(search('hydrogen sulfide exposure'), indent=2))"
```

Turn on airplane mode and run it again. It should return the same output.
