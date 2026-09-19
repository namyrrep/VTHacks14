# BLACKOUT — Backend (Search Pipeline)

The backend turns protocol manual PDFs into a local search index and exposes one function, `search()`, that the Streamlit UI calls. Once the index files exist, **nothing in the backend touches the network.**

No generative model sits in the answer path. The backend only returns verbatim passages from the manuals, with citations, and abstains when confidence is low.

## Pipeline

```
Documents/*.pdf
   │  ingest.py   PDF → section chunks (run once)
   ▼
data/chunks.json  [{id, manual, section, page, text}, ...]
   │  index.py    embed every chunk (run once)
   ▼
data/index.npy    one normalized embedding per chunk
   │
   ▼
search.py         question → embed → cosine similarity → top k → threshold check
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Extracts text from each PDF with PyMuPDF, chunks by section (fallback: 500–800 words with overlap), keeps page numbers, writes `data/chunks.json` |
| `index.py` | Embeds every chunk with `all-MiniLM-L6-v2`, writes `data/index.npy` |
| `search.py` | Loads the two data files and implements `search()` |
| `data/chunks.json` | Generated, **committed** so a fresh clone runs with no build step |
| `data/index.npy` | Generated, **committed** for the same reason |

## Manuals loaded

| Manual | Source PDF |
|---|---|
| Emergency Response Guidebook (ERG) 2024 | `Documents/2024EmergencyResponseGuide.pdf` |
| NIOSH Pocket Guide to Chemical Hazards | `Documents/CDCPocketGuide.pdf` |

## Setup

Do this **while on wifi**. The embedding model (~90 MB) downloads on first use and is cached locally after that.

```bash
pip install -r requirements.txt   # pymupdf, sentence-transformers, numpy, rank_bm25
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Rebuild the index only if the manuals change:

```bash
python ingest.py
python index.py
```

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
{"id": "niosh_0172", "manual": "NIOSH Pocket Guide",
 "section": "Hydrogen Sulfide", "page": 172, "text": "..."}
```

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
