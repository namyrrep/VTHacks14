"""Checks search.py end to end and suggests a confidence threshold.

    python eval_search.py

1. Corpus: every protocol is consistent (one manual, one name, parts in reading order) and
   unique (no two protocols share a name or a full text, no passage sits in two protocols).
2. Results: for every test question, at most 3 results, each a different protocol, scores
   sorted and above THRESHOLD, the answer a single verbatim line of the linked protocol,
   and the cited chunk one of that protocol's parts.
3. Threshold: in-corpus questions must be answered with an expected protocol in the top 3;
   out-of-corpus questions must be refused. Prints the score gap between the two sets.

Exits non-zero if any check fails.
"""
import sys
from collections import Counter

import search as S

# (question, protocol ids that count as a correct answer)
IN_CORPUS = [
    ("how do I treat a thermal burn", {"wv_t008"}),
    # T009 covers eye trauma; chemical-splash irrigation is printed in T008 Burns
    ("chemical splashed in patient's eye", {"wv_t008", "wv_gl016"}),
    ("foreign object stuck in the eye", {"wv_t009"}),
    ("hydrogen sulfide exposure limits", {"niosh_hydrogen_sulfide"}),
    ("evacuation distance for a chlorine leak", {"erg_t1_1017", "erg_t3_1017", "erg_guide124"}),
    ("UN1005 large spill", {"erg_t1_1005", "erg_t3_1005"}),
    ("what is the dose of naloxone for an opioid overdose", {"wv_naloxone_narcan", "wv_m007"}),
    ("patient having a seizure", {"wv_m004", "wv_pm004"}),
    ("adult cardiac arrest CPR", {"wv_c003"}),
    ("snake bite treatment", {"wv_e004"}),
    ("signs of heat stroke", {"wv_e002"}),
    ("hypothermia cold exposure care", {"wv_e003"}),
    ("anaphylaxis epinephrine", {"wv_e001", "wv_epinephrine_1_1_000", "wv_epipen_epipen_jr"}),
    ("carbon monoxide poisoning first aid", {"niosh_carbon_monoxide", "wv_r004"}),
    ("propane tank BLEVE fireball distance", {"erg_ref_bleve_safety_precautions", "erg_guide115"}),
    ("nerve agent exposure symptoms", {"wv_gl008", "erg_ref_criminal_or_terrorist_use_of_c_13"}),
    ("stroke assessment scale", {"wv_appendix_d", "wv_m003"}),
    ("how to control severe bleeding with a tourniquet", {"wv_t001", "wv_pt001"}),
    ("ammonia respirator recommendations", {"niosh_ammonia"}),
    ("unknown cargo spill on the highway", {"erg_guide111"}),
    ("pediatric glasgow coma scale", {"wv_appendix_b", "wv_appendix_c"}),
    # bystander wording (search.LAY_TERMS)
    ("someone is choking", {"wv_r001"}),
    ("guy got electrocuted", {"wv_t008"}),
    ("bee sting swelling throat", {"wv_e001", "wv_pe001"}),
    ("he hit his head and is confused", {"wv_t006"}),
    ("baby not breathing", {"wv_pr002", "wv_pc003", "wv_pm015"}),
    # exact identifiers alone
    ("7783-06-4", {"niosh_hydrogen_sulfide"}),
    ("T008", {"wv_t008"}),
    ("UN 1017", {"erg_t1_1017", "erg_t3_1017", "niosh_chlorine", "erg_guide124"}),
]

OUT_OF_CORPUS = [
    "what's the wifi password",
    "who won the super bowl last year",
    "recipe for chocolate chip cookies",
    "how do I reset my email password",
    "best pizza place near campus",
    "what time does the library close",
    "explain quantum computing",
    "how many moons does jupiter have",
    "write me a poem about the ocean",
    "what is the stock price of apple",
    "what happened in 2024",            # a year that is also a UN number
    "call me at 555 1234",
    # near the manuals' domain, but not covered by them
    "covid vaccine schedule",
    "how to treat depression",
    "dog bite rabies shots",
    "symptoms of the flu",
]


def check_corpus(engine) -> list[str]:
    problems = []
    names, texts = Counter(), Counter()
    for pid, parts in engine.protocols.items():
        chunks = [engine.chunks[i] for i in parts]
        if len({c["manual"] for c in chunks}) != 1:
            problems.append(f"{pid}: parts from several manuals")
        if len({c["protocol"] for c in chunks}) != 1:
            problems.append(f"{pid}: parts carry different protocol names")
        if [c["page"] for c in chunks] != sorted(c["page"] for c in chunks):
            problems.append(f"{pid}: parts out of page order")
        p = engine.protocol(pid)
        names[(p["manual"], p["protocol"])] += 1
        texts[p["text"]] += 1
        if not engine.answer_lines(pid)[0]:
            problems.append(f"{pid}: no line can serve as a one-line answer")
    problems += [f"protocol name used {n}x: {k}" for k, n in names.items() if n > 1]
    problems += [f"protocol text repeated {n}x: {t[:60]!r}" for t, n in texts.items() if n > 1]
    return problems


def check_result(question: str, out: dict) -> list[str]:
    problems = []
    rs = out["results"]
    if len(rs) > S.TOP_K:
        problems.append(f"{len(rs)} results")
    if len({r["protocol_id"] for r in rs}) != len(rs):
        problems.append("the same protocol returned twice")
    if [r["score"] for r in rs] != sorted((r["score"] for r in rs), reverse=True):
        problems.append("scores not sorted")
    if out["confident"] != bool(rs):
        problems.append("confident does not match whether results were returned")
    for r in rs:
        tag = r["protocol_id"]
        if r["score"] < S.THRESHOLD:
            problems.append(f"{tag}: score {r['score']} below threshold returned")
        if r["chunk_id"] not in r["chunk"]["parts"]:
            problems.append(f"{tag}: cited chunk {r['chunk_id']} is not part of the protocol")
        if r["text"] not in r["chunk"]["text"] and r["text"].split("\n", 1)[-1] not in r["chunk"]["text"]:
            problems.append(f"{tag}: cited passage missing from the protocol text")
        answer = r["answer"].removesuffix(" …")
        if "\n" in r["answer"] or not answer or answer not in r["chunk"]["text"]:
            problems.append(f"{tag}: answer is not one verbatim line of the protocol: {r['answer']!r}")
    return [f"{question!r}: {p}" for p in problems]


def main() -> int:
    engine = S._engine()
    problems = check_corpus(engine)
    print(f"corpus: {len(engine.chunks)} chunks, {len(engine.protocols)} protocols, "
          f"{len(problems)} consistency/uniqueness problem(s)")

    in_scores, out_scores = [], []
    print(f"\nIN-CORPUS (threshold {S.THRESHOLD})")
    for q, expected in IN_CORPUS:
        out = S.search(q)
        problems += check_result(q, out)
        # score the question as if there were no threshold, to measure the margin
        raw = S.THRESHOLD
        S.THRESHOLD = 0.0
        ranked = [r["protocol_id"] for r in S.search(q)["results"]]
        S.THRESHOLD = raw
        in_scores.append(out["top_score"])
        rank = next((n for n, pid in enumerate(ranked, 1) if pid in expected), None)
        ok = out["confident"] and rank is not None and ranked[rank - 1] in [r["protocol_id"] for r in out["results"]]
        if not ok:
            problems.append(f"{q!r}: expected {sorted(expected)}, got {ranked} (top {out['top_score']})")
        print(f"  {'ok ' if ok else 'BAD'} {out['top_score']:.3f} rank={rank} {q!r}")
        for r in out["results"]:
            print(f"        {r['score']:.3f} {r['protocol']} :: {r['answer'][:110]}")

    print("\nOUT-OF-CORPUS")
    for q in OUT_OF_CORPUS:
        out = S.search(q)
        problems += check_result(q, out)
        out_scores.append(out["top_score"])
        ok = not out["confident"]
        if not ok:
            problems.append(f"{q!r}: answered an out-of-corpus question ({out['top_score']})")
        print(f"  {'ok ' if ok else 'BAD'} {out['top_score']:.3f} {q!r}")

    lo, hi = min(in_scores), max(out_scores)
    print(f"\nlowest in-corpus top score {lo:.3f}, highest out-of-corpus top score {hi:.3f}"
          + (f" -> any threshold in ({hi:.3f}, {lo:.3f}] separates them; midpoint {(lo + hi) / 2:.3f}"
             if lo > hi else " -> the two sets overlap"))

    print(f"\n{len(problems)} problem(s)")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
