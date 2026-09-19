"""data/chunks.json -> data/index.npy (+ data/index_meta.json)

Run after ingest.py:  python index.py

index.npy holds one L2-normalised float32 embedding per chunk, in the same order as
chunks.json, so cosine similarity at query time is a single matrix-vector product.

all-MiniLM-L6-v2 only reads the first 256 word pieces of its input. A chunk longer than
that (a full NIOSH record, an ERG guide page) is embedded as overlapping word windows and
the window vectors are averaged, so every part of the passage contributes to its row.

The model is downloaded on the first run (~90 MB) and cached; after that this script,
like the rest of the backend, runs with no network connection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data" / "chunks.json"
INDEX = ROOT / "data" / "index.npy"
META = ROOT / "data" / "index_meta.json"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_WORDS = 150   # stays inside the 256 word-piece limit, even for code-heavy NIOSH text
STRIDE_WORDS = 100
MAX_KEYWORDS = 12    # guide chunks carry hundreds of material names; embed only the first few


def embed_inputs(chunk: dict) -> list[str]:
    """Text windows to embed for one chunk. Each window is prefixed with the section title
    and leading keywords so the vector knows what the passage is about."""
    header = chunk["section"]
    kws = [k for k in chunk["keywords"][:MAX_KEYWORDS] if k.lower() not in header.lower()]
    if kws:
        header += " | " + ", ".join(kws)
    words = chunk["text"].split()
    if len(words) <= WINDOW_WORDS:
        return [f"{header}\n{chunk['text']}"]
    windows = []
    for start in range(0, len(words), STRIDE_WORDS):
        windows.append(f"{header}\n{' '.join(words[start:start + WINDOW_WORDS])}")
        if start + WINDOW_WORDS >= len(words):
            break
    return windows


def main() -> None:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    inputs, owner = [], []
    for i, c in enumerate(chunks):
        for text in embed_inputs(c):
            inputs.append(text)
            owner.append(i)
    print(f"{len(chunks)} chunks -> {len(inputs)} windows")

    model = SentenceTransformer(MODEL, device="cpu")
    vecs = model.encode(inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    owner = np.array(owner)
    index = np.zeros((len(chunks), vecs.shape[1]), dtype=np.float32)
    np.add.at(index, owner, vecs)
    index /= np.linalg.norm(index, axis=1, keepdims=True)
    np.save(INDEX, index)

    ids = "\n".join(c["id"] for c in chunks).encode("utf-8")
    META.write_text(json.dumps({
        "model": MODEL,
        "dim": int(index.shape[1]),
        "chunks": len(chunks),
        "chunk_ids_sha256": hashlib.sha256(ids).hexdigest(),  # search.py can detect a stale index
        "window_words": WINDOW_WORDS,
        "stride_words": STRIDE_WORDS,
    }, indent=1), encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)} {index.shape} and {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
