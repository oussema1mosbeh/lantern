# Step 3b: which search finds the right passage? Measured on the 55 answerable test questions.
#
#   word search (BM25)  - counts shared words, rare words count more. Fast, but "train" never matches "lost".
#   meaning search      - embeddings: passages and questions become 384 numbers; close numbers = close meaning.
#                         Two models compared: all-MiniLM-L6-v2 (used in Feedback Lens) and BGE-small.
#   hybrid              - both lists merged with reciprocal rank fusion: a passage ranked high by either wins.
#
# hit@k = the right passage is in the top k results. MRR = 1/rank of the first right passage, averaged.
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
questions = [q for q in json.loads((ROOT / "data" / "questions_gold.json").read_text(encoding="utf-8")) if q["gold"]]


def passage(c):
    return f"{c['title']}. {c['section']}. {c['text']}"


# ---- word search: BM25 ----
STOP = set("a an the of to in on for and or is are be by with as at it its this that from i my we our you your do does can "
           "what how when which who much many there their they them if not have has will may per any all".split())


def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-z0-9$%]+(?:[.,][0-9]+)*", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for d in self.docs for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + .5) / (f + .5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokens(query)
        return np.array([sum(self.idf.get(w, 0) * d[w] * (self.k1 + 1) / (d[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
                             for w in q if w in d) for d, L in zip(self.docs, self.len)])


def ranking(scores):
    return list(np.argsort(-scores))


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: each list gives 1/(k + rank) points to each passage."""
    s = np.zeros(len(chunks))
    for r in rankings:
        for rank, i in enumerate(r):
            s[i] += 1 / (k + rank + 1)
    return ranking(s)


def evaluate(name, rank_fn):
    res = {"easy": [], "hard": []}
    for q in questions:
        r = rank_fn(q["question"])
        first = next(k for k, i in enumerate(r) if i in q["gold"]) + 1
        res[q["type"]].append(first)
    allr = res["easy"] + res["hard"]
    hit = lambda rs, k: 100 * sum(x <= k for x in rs) / len(rs)
    row = {"method": name, "hit@1": hit(allr, 1), "hit@3": hit(allr, 3), "hit@5": hit(allr, 5),
           "MRR": float(np.mean([1 / x for x in allr])), "easy hit@3": hit(res["easy"], 3), "hard hit@3": hit(res["hard"], 3)}
    print(f"{name:<28} {row['hit@1']:6.1f} {row['hit@3']:6.1f} {row['hit@5']:6.1f} {row['MRR']:6.3f}   {row['easy hit@3']:6.1f} {row['hard hit@3']:6.1f}")
    return row, allr


if __name__ == "__main__":
    texts = [passage(c) for c in chunks]
    bm25 = BM25(texts)

    models = {
        "MiniLM": ("sentence-transformers/all-MiniLM-L6-v2", ""),
        "BGE-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    }
    emb = {}
    for name, (mid, prefix) in models.items():
        m = SentenceTransformer(mid, device="cuda")
        P = m.encode(texts, normalize_embeddings=True)
        Q = m.encode([prefix + q["question"] for q in questions], normalize_embeddings=True)
        emb[name] = (P, {q["question"]: v for q, v in zip(questions, Q)})

    def meaning(name):
        P, Q = emb[name]
        return lambda q: ranking(P @ Q[q])

    print(f"{'':<28} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}   {'easy@3':>6} {'hard@3':>6}")
    rows = []
    rows.append(evaluate("Word search (BM25)", lambda q: ranking(bm25.scores(q))))
    for name in models:
        rows.append(evaluate(f"Meaning search ({name})", meaning(name)))
    for name in models:
        rows.append(evaluate(f"Hybrid (BM25 + {name})", lambda q, n=name: rrf(ranking(bm25.scores(q)), meaning(n)(q))))

    # which questions does the best method still miss?
    best = max(rows, key=lambda r: (r[0]["hit@3"], r[0]["MRR"]))
    print(f"\nbest: {best[0]['method']}. Questions where the right passage is not in its top 3:")
    for q, first in zip(questions, best[1]):
        if first > 3:
            print(f"  rank {first:>2}  [{q['type']}] {q['question']}")
    (ROOT / "data" / "search_results.json").write_text(json.dumps([r[0] for r in rows], indent=1), encoding="utf-8")
