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


def fingerprint(chunks: list[dict]) -> str:
    """Hash of everything that goes into an embedding, so search.py can tell when
    chunks.json has changed since index.npy was built."""
    h = hashlib.sha256()
    for c in chunks:
        h.update("\x1f".join([c["id"], c["section"], c["text"], *c["keywords"][:MAX_KEYWORDS]]).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def main() -> None:
    from sentence_transformers import SentenceTransformer  # heavy; search.py only needs fingerprint()

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

    META.write_text(json.dumps({
        "model": MODEL,
        "dim": int(index.shape[1]),
        "chunks": len(chunks),
        "chunks_sha256": fingerprint(chunks),  # search.py refuses an index built from other chunks
        "window_words": WINDOW_WORDS,
        "stride_words": STRIDE_WORDS,
    }, indent=1), encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)} {index.shape} and {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
