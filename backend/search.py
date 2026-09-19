"""question -> the top protocols that answer it, each with a one-line answer.

    from search import search
    search("how do I treat a chemical burn to the eye")

Retrieval runs over chunks (small passages embed precisely) but answers are given per
protocol: chunk scores are pooled by `protocol_id`, so the top 3 results are always three
different protocols. Each result carries
  - `answer`: the single line of that protocol that best matches the question, verbatim
  - `text` / `section` / `page`: the best-matching chunk (the citation)
  - `chunk`: every part of the protocol, in order (the full information to link to)

Scoring is cosine similarity between the question and each chunk's embedding (index.py),
plus a small boost when the question names one of a chunk's identifying keywords (a UN or
CAS number, a protocol code, a chemical or drug name). A result below THRESHOLD is not an
answer; when even the best protocol is below it, `confident` is False and nothing is
returned, so the UI shows its refusal state instead of a guess.

Everything is local: no network, no generative model. Only verbatim manual text is returned.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from index import MAX_KEYWORDS, fingerprint

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data" / "chunks.json"
INDEX = ROOT / "data" / "index.npy"
META = ROOT / "data" / "index_meta.json"

THRESHOLD = 0.45          # tuned with eval_search.py; see BACKEND.md
TOP_K = 3
KEYWORD_BOOST = 0.12      # added to a chunk's score when the question names one of its keywords
IDENTIFIER_BOOST = 0.2    # ... when that keyword is an exact UN/CAS number or protocol code
IDENTIFIER_FLOOR = 0.6    # minimum score of a chunk the question names by exact identifier
MAX_KEYWORD_SPREAD = 8    # a keyword naming more protocols than this ("dose", "Table 1") is not identifying
MAX_NGRAM = 8             # longest keyword, in words, that is looked up in the question
RERANK_POOL = 15          # best protocols by embedding score that get the lexical bonus
LEXICAL_BONUS = 0.08      # x share of the question's topic words found in the cited passage
ANSWER_MIN_WORDS = 4
ANSWER_SHORT_WORDS = 6        # lines shorter than this rarely answer anything on their own
ANSWER_SHORT_PENALTY = 0.05
ANSWER_MAX_CHARS = 240
ANSWER_OVERLAP_WEIGHT = 0.25  # weight of the question's topic words found in the line
ANSWER_MAX_IDENTITY = 0.8     # a line made mostly of the protocol's own name/codes is a label

NORM_RE = re.compile(r"[^0-9a-z]+")
BULLET_RE = re.compile(r"^(?:[•–-]\s+)+")
# split only at sentence ends: a ";" usually separates cells that need each other (a table row)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(])")
# row banners that only restate the protocol's name ("ID 1017 – Guide 124 – Chlorine")
BANNER_RE = re.compile(r"(?:GUIDE \d{3}|ID \d{4} – Guide \d{3}P?) – ")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "should", "that", "the", "to",
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "we", "there", "this",
}


def _norm(text: str) -> str:
    return NORM_RE.sub(" ", text.lower()).strip()


# words that say what kind of help is wanted, not what it is about; every protocol "treats"
INTENT_WORDS = {
    "treat", "treatment", "treating", "care", "manage", "management", "patient", "patients",
    "protocol", "procedure", "help", "having", "someone", "person", "guide", "info", "information",
}


def _singular(w: str) -> str:
    """'spills' -> 'spill', 'eyes' -> 'eye', 'injuries' -> 'injury'; leaves 'gas', 'class' alone."""
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _content_words(text: str) -> set[str]:
    return {_singular(w) for w in _norm(text).split() if w not in STOPWORDS and len(w) > 1}


# the manuals' wording for what a question asks ("dose" is printed as "Administer 2 mg")
TOPIC_SYNONYMS = {
    "dose": {"administer", "administration", "mg", "mcg"},
    "dosage": {"administer", "administration", "mg", "mcg"},
    "symptom": {"sign"},
    "sign": {"symptom"},
}


def _topic_words(question: str, identity: set[str]) -> set[str]:
    """What the question asks about a protocol, beyond naming the protocol itself."""
    return _content_words(question) - identity - INTENT_WORDS


def _overlap(topic: set[str], words: set[str]) -> float:
    """Share of the topic words (or their synonyms) present in words."""
    if not topic:
        return 0.0
    return sum(1 for t in topic if t in words or TOPIC_SYNONYMS.get(t, set()) & words) / len(topic)


# Lay wording -> the clinical terms the manuals use. MiniLM does not connect "passed out" with
# "unconscious" or "swallowed bleach" with "toxic ingestion", so the terms are appended to the
# question before it is embedded. Keep entries to phrases a bystander or dispatcher would say.
LAY_TERMS = [
    (r"\bpass(?:ed|es|ing)? out\b|\bfaint(?:ed|ing|s)?\b|\bwo?n'?t wake\b|\bnot waking\b",
     "unconscious unresponsive altered mental status syncope"),
    (r"\bchok(?:e|es|ed|ing)\b", "airway obstruction airway management"),
    (r"\belectrocut\w*|\blightning\b|\belectric shock\b", "electrical burns"),
    (r"\bswallow(?:ed|s)?\b|\bdrank\b|\bingest\w*", "toxic ingestion poisoning"),
    (r"\bod\b|\boverdos\w*|\bodd?ed\b", "overdose toxic ingestion poisoning"),
    (r"\bheart attack\b", "chest pain cardiac"),
    (r"\b(?:not|stopped|isn'?t|wasn'?t) breathing\b", "respiratory arrest cardiac arrest"),
    (r"\bbee sting\b|\bstung\b|\bstings?\b", "allergic reaction anaphylaxis"),
    (r"\bblood sugar\b", "blood glucose diabetic"),
    (r"\bfits?\b|\bconvuls\w*", "seizure"),
    (r"\bbroken (?:bone|arm|leg|wrist|ankle|hip)\b", "fracture musculoskeletal trauma"),
    (r"\bthr(?:ew|owing|ows) up\b|\bpuk\w*", "nausea vomiting"),
    (r"\bhead injury\b|\bconcussion\b|\bhit (?:his|her|their|my) head\b", "traumatic brain injury"),
]
LAY_TERMS = [(re.compile(p, re.I), terms) for p, terms in LAY_TERMS]


def _expand(question: str) -> str:
    extra = [terms for pattern, terms in LAY_TERMS if pattern.search(question)]
    return f"{question} ({'; '.join(extra)})" if extra else question


# a UN number, a CAS number, or a WV protocol code: an exact match is strong evidence
IDENTIFIER_RE = re.compile(r"UN\d{4}|(?:CAS )?\d{2,7}-\d{2}-\d|[A-Z]{1,3}\d{3}")


def _keyword_forms(kw: str) -> set[str]:
    """Normalised ways a question may name a keyword: "Medication Formulary – NALOXONE
    (Narcan®)" is also "naloxone" and "narcan"."""
    kw = kw.replace("®", "")
    forms = {kw, kw.split(" – ")[-1]}
    for f in list(forms):
        forms.add(re.sub(r"\s*\([^)]*\)", "", f))
        forms.update(p for p in re.findall(r"\(([^)]*)\)", f)
                     if len(p.split()) <= 3 and p[:1].isalpha() and p.lower() != "optional")
    return {k for k in map(_norm, forms) if len(k) >= 3 and k not in STOPWORDS}


class _Engine:
    def __init__(self) -> None:
        self.chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
        self.vectors = np.load(INDEX)
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta.get("chunks_sha256") != fingerprint(self.chunks) or len(self.vectors) != len(self.chunks):
            raise RuntimeError("data/index.npy was built from a different data/chunks.json; run `python index.py`")
        self.model_name = meta["model"]
        self._model = None

        # protocol_id -> chunk positions, in reading order
        self.protocols: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.protocols.setdefault(c["protocol_id"], []).append(i)
        for parts in self.protocols.values():
            parts.sort(key=lambda i: (self.chunks[i]["page"], i))
        order = {pid: n for n, pid in enumerate(self.protocols)}
        self.protocol_ids = list(self.protocols)
        self.chunk_protocol = np.array([order[c["protocol_id"]] for c in self.chunks])

        # identifying keyword (normalised) -> chunk positions
        spread: dict[str, set[str]] = {}
        owners: dict[str, set[int]] = {}
        self.identifiers: set[str] = set()
        for i, c in enumerate(self.chunks):
            for kw in c["keywords"] + [c["protocol"]]:
                if IDENTIFIER_RE.fullmatch(kw):
                    self.identifiers.add(_norm(kw))
                for key in _keyword_forms(kw):
                    spread.setdefault(key, set()).add(c["protocol_id"])
                    owners.setdefault(key, set()).add(i)
        self.keywords = {k: sorted(owners[k]) for k, pids in spread.items() if len(pids) <= MAX_KEYWORD_SPREAD}

        self._lines_cache: dict[str, tuple[list[tuple[str, int]], set[str]]] = {}
        self._answer_cache: dict[str, tuple[list[tuple[str, int]], np.ndarray, set[str]]] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:  # never touch the network once the model is cached
                self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
            except OSError:
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # ------------------------------------------------------------ scoring

    def keyword_hits(self, question: str) -> tuple[dict[int, float], set[int]]:
        """Chunks whose identifying keywords appear in the question (whole words), with the
        boost each earns, and the chunks named by an explicit identifier ("UN1017", a CAS
        number, "T008"). A bare "1017" earns only the keyword boost: "2024" is a year far
        more often than it is UN2024."""
        q = _norm(question)
        q = re.sub(r"\bun (\d{4})\b", r"un\1", q)
        # appended, not inserted, so a CAS number ("7783 06 4") stays one n-gram
        bare = [f"un{d}" for d in re.findall(r"\b(\d{4})\b", q)]
        words = q.split()
        hits: dict[int, float] = {}
        exact: set[int] = set()
        for n in range(1, MAX_NGRAM + 1):
            for start in range(len(words) - n + 1):
                key = " ".join(words[start:start + n])
                owners = self.keywords.get(key, ())
                if key in self.identifiers:
                    exact.update(owners)
                for i in owners:
                    hits[i] = max(hits.get(i, 0.0), IDENTIFIER_BOOST if key in self.identifiers else KEYWORD_BOOST)
        for key in bare:
            for i in self.keywords.get(key, ()):
                hits[i] = max(hits.get(i, 0.0), KEYWORD_BOOST)
        return hits, exact

    def chunk_scores(self, question: str, q_vec: np.ndarray) -> np.ndarray:
        scores = self.vectors @ q_vec
        hits, exact = self.keyword_hits(question)
        for i, boost in hits.items():
            scores[i] += boost
        # a question that names a UN/CAS number or protocol code exactly is answered by it,
        # even when the question has no other words for the embedding to match
        # (plus a sliver of the score, so tied identifier matches still rank by meaning)
        for i in exact:
            scores[i] = max(scores[i], IDENTIFIER_FLOOR + 0.1 * scores[i])
        return np.clip(scores, 0.0, 1.0)

    def identity_words(self, parts: list[dict]) -> set[str]:
        """Words that name the protocol itself (its title, codes, identifying keywords). Every
        line of the protocol is 'about' these, so they say nothing about which line answers
        the question. Keywords shared by many protocols ("dose", "Table 1") do not count."""
        words = set()
        for kw in [parts[0]["protocol"]] + [k for c in parts for k in c["keywords"][:MAX_KEYWORDS]]:
            forms = _keyword_forms(kw)
            if kw == parts[0]["protocol"] or forms & self.keywords.keys():
                for f in forms:
                    words |= _content_words(f)
        words |= {w[2:] for w in words if re.fullmatch(r"un\d{4}", w)}  # "UN1053" is printed "1053"
        return words

    # ------------------------------------------------------------ protocol view

    def protocol(self, pid: str) -> dict:
        parts = [self.chunks[i] for i in self.protocols[pid]]
        texts, banner = [], parts[0]["text"].split("\n", 1)[0]
        for n, c in enumerate(parts):
            text = c["text"]
            # ERG guide parts each repeat the guide banner printed at the top of the page
            if n and text.split("\n", 1)[0] == banner and "\n" in text:
                text = text.split("\n", 1)[1]
            texts.append(text)
        pages = sorted({c["page"] for c in parts})
        return {
            "protocol_id": pid,
            "protocol": parts[0]["protocol"],
            "manual": parts[0]["manual"],
            "kind": parts[0]["kind"],
            "page": pages[0],
            "pages": pages,
            "printed_pages": list(dict.fromkeys(c["printed_page"] for c in parts if c["printed_page"])),
            "parts": [c["id"] for c in parts],
            "text": "\n".join(texts),
        }

    def answer_candidates(self, pid: str) -> tuple[list[tuple[str, int]], np.ndarray, set[str]]:
        """answer_lines() plus their embeddings, cached per protocol."""
        if pid not in self._answer_cache:
            cands, identity = self.answer_lines(pid)
            self._answer_cache[pid] = (cands, self.embed([s for s, _ in cands]), identity)
        return self._answer_cache[pid]

    def answer_lines(self, pid: str) -> tuple[list[tuple[str, int]], set[str]]:
        """Every line of the protocol that can stand alone as a one-line answer, with the
        chunk position it came from, and the protocol's identity words. Headings and
        name/code lines are not answers. Cached per protocol."""
        if pid in self._lines_cache:
            return self._lines_cache[pid]
        parts = [self.chunks[i] for i in self.protocols[pid]]
        identity = self.identity_words(parts)
        labels = {_norm(c["protocol"]) for c in parts} | {_norm(c["section"]) for c in parts}
        cands, seen = [], set()
        for i, c in zip(self.protocols[pid], parts):
            for line in c["text"].split("\n"):
                line = BULLET_RE.sub("", line).strip()
                pieces = SENTENCE_RE.split(line) if len(line) > ANSWER_MAX_CHARS else [line]
                for s in pieces:
                    s = s.strip()
                    words = _content_words(s)
                    heading = (s.isupper() and len(s.split()) < 8 or BANNER_RE.match(s)
                               or any(_norm(s) in label for label in labels))
                    name_only = words and len(words & identity) / len(words) >= ANSWER_MAX_IDENTITY
                    if len(s.split()) < ANSWER_MIN_WORDS or heading or name_only or s in seen:
                        continue
                    seen.add(s)
                    cands.append((s, i))
        if not cands:  # a protocol of short lines only: fall back to its longest line
            i = self.protocols[pid][0]
            cands = [(max(self.chunks[i]["text"].split("\n"), key=len), i)]
        self._lines_cache[pid] = (cands, identity)
        return cands, identity

    def best_answer(self, question: str, q_vec: np.ndarray, pid: str) -> tuple[str, int]:
        """The line that best matches the question: semantic similarity, plus how many of the
        question's topic words (not the protocol's own name) the line contains. A question
        that only names the protocol ("T008", "snake bite") gets the line most representative
        of the protocol as a whole."""
        cands, vecs, identity = self.answer_candidates(pid)
        topic = _topic_words(question, identity)
        target = q_vec
        if not topic:
            centroid = self.vectors[self.protocols[pid]].mean(axis=0)
            target = q_vec + centroid / np.linalg.norm(centroid)
            target /= np.linalg.norm(target)
        best, best_score = cands[0], -1.0
        for (text, i), v in zip(cands, vecs):
            overlap = _overlap(topic, _content_words(text))
            score = float(v @ target) + ANSWER_OVERLAP_WEIGHT * overlap
            if len(text.split()) < ANSWER_SHORT_WORDS:
                score -= ANSWER_SHORT_PENALTY
            if score > best_score:
                best, best_score = (text, i), score
        text, i = best
        if len(text) > ANSWER_MAX_CHARS:
            text = text[:ANSWER_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + " …"
        return text, i

    # ------------------------------------------------------------ search

    def search(self, question: str, k: int) -> dict:
        question = (question or "").strip()
        if not question:
            return {"confident": False, "top_score": 0.0, "results": []}
        question = _expand(question)
        q_vec = self.embed([question])[0]
        scores = self.chunk_scores(question, q_vec)

        # a protocol scores as its best-matching chunk ...
        best_chunk: dict[int, int] = {}
        for i in np.argsort(-scores):
            p = int(self.chunk_protocol[i])
            if p not in best_chunk:
                best_chunk[p] = int(i)
                if len(best_chunk) == RERANK_POOL:
                    break
        # ... plus a bonus for containing what the question asks about, beyond the protocol's
        # name: "hydrogen sulfide exposure limits" prefers the record listing exposure limits
        # over a table row that only names hydrogen sulfide
        ranked = []
        for p, i in best_chunk.items():
            pid = self.protocol_ids[p]
            topic = _topic_words(question, self.answer_lines(pid)[1])
            c = self.chunks[i]
            found = _overlap(topic, _content_words(c["section"] + " " + c["text"]))
            ranked.append((min(1.0, float(scores[i]) + LEXICAL_BONUS * found), p, i))
        ranked.sort(key=lambda r: -r[0])
        ranked = ranked[:k]
        top_score = round(ranked[0][0], 4) if ranked else 0.0
        if top_score < THRESHOLD:
            return {"confident": False, "top_score": top_score, "results": []}

        results = []
        for score, p, i in ranked:
            if score < THRESHOLD:
                break  # not an answer: never shown, even below a confident top result
            pid = self.protocol_ids[p]
            c = self.chunks[i]
            answer, answer_chunk = self.best_answer(question, q_vec, pid)
            results.append({
                "text": c["text"],
                "manual": c["manual"],
                "section": c["section"],
                "page": c["page"],
                "score": round(score, 4),
                "answer": answer,
                "answer_page": self.chunks[answer_chunk]["page"],
                "chunk_id": c["id"],
                "protocol_id": pid,
                "protocol": c["protocol"],
                "chunk": self.protocol(pid),
            })
        return {"confident": True, "top_score": top_score, "results": results}


@lru_cache(maxsize=1)
def _engine() -> _Engine:
    return _Engine()


def search(question: str, k: int = TOP_K) -> dict:
    """
    {
      "confident": bool,       # False -> UI shows the refusal state; results is empty
      "top_score": float,      # 0.0-1.0, score of the best protocol
      "results": [             # at most k, one per protocol, each scoring >= THRESHOLD
        {
          "text": str,         # best-matching passage, VERBATIM from the manual
          "manual": str,       # "NIOSH Pocket Guide to Chemical Hazards"
          "section": str,      # "Hydrogen sulfide"
          "page": int,         # PDF page of that passage
          "score": float,      # 0.0-1.0
          "answer": str,       # one line of the protocol answering the question, verbatim
          "answer_page": int,  # PDF page that line is on
          "chunk_id": str,     # id of the passage in data/chunks.json
          "protocol_id": str,  # id of the whole protocol; get_protocol(protocol_id)
          "protocol": str,     # "T008 – Burns"
          "chunk": dict,       # the whole protocol, see get_protocol()
        }, ...
      ]
    }
    """
    return _engine().search(question, k)


def get_protocol(protocol_id: str) -> dict | None:
    """Every part of one protocol, in reading order:
    {"protocol_id", "protocol", "manual", "kind", "page", "pages", "printed_pages",
     "parts": [chunk ids], "text": full protocol text}"""
    engine = _engine()
    return engine.protocol(protocol_id) if protocol_id in engine.protocols else None


if __name__ == "__main__":
    import sys

    out = search(" ".join(sys.argv[1:]) or "hydrogen sulfide exposure")
    for r in out["results"]:
        r["chunk"] = {k: v for k, v in r["chunk"].items() if k != "text"}  # keep the console readable
    print(json.dumps(out, indent=2, ensure_ascii=False))
