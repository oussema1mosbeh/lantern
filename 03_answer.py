# Step 4: re-rank the search results, find the answer, and decide when to say "I couldn't find this".
#
#   search      BGE-small meaning search (the winner of step 3) returns the 10 closest passages
#   re-rank     a cross-encoder reads the question and each passage TOGETHER and scores them again
#   answer, two approaches compared:
#     A. exact words   an extractive QA model (SQuAD 2.0) picks the words that answer the question
#     B. best sentence the cross-encoder scores every sentence of the 3 best passages; the app
#                      highlights the winner, like a person pointing at the line in the document
#   refuse      if the best score is too low, answer "I couldn't find this"
#
# Thresholds are tuned on one half of the questions and graded on the other half (then swapped),
# so the score is never measured on the questions used to tune it.
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parent
chunks = json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))
questions = json.loads((ROOT / "data" / "questions_gold.json").read_text(encoding="utf-8"))
TOP_K, READ = 10, 3          # passages from search, passages read closely

EMBEDDER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKERS = {
    "ms-marco-MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mxbai-xsmall": "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "bge-base": "BAAI/bge-reranker-base",
}
TAG = next((a for a in sys.argv[1:] if a in RERANKERS), "ms-marco-MiniLM")
RERANKER = RERANKERS[TAG]
READER = "deepset/tinyroberta-squad2"


def passage(c):
    return f"{c['title']}. {c['section']}. {c['text']}"


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"($0-9])", text) if s.strip()]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9$%.,:]+", s.lower())).strip(" .,")


def overlaps_evidence(q, text):
    """The shown text answers the question if it contains the evidence (or is part of it),
    or contains the reference answer, wherever in the documents it was found."""
    t = norm(text)
    if len(t) < 2:
        return False
    for s in q["sources"]:
        ev = norm(s["evidence"])
        if ev in t or (len(t) > 15 and t in ev):
            return True
    return norm(q["answer"]) in t


def words_correct(q, answer):
    a = norm(answer)
    if not a:
        return False
    if overlaps_evidence(q, answer):
        return True
    gold, pred = set(norm(q["answer"]).split()), set(a.split())
    return len(gold & pred) / max(len(gold), 1) >= .5


class Reader:
    """Extractive QA by hand (the browser app does exactly the same with the ONNX model):
    the model scores every token as a possible answer START and END; the answer is the best
    start..end pair inside the passage."""

    def __init__(self, name, dev, max_answer_tokens=30):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name).to(dev).eval()
        self.dev, self.max_len = dev, max_answer_tokens

    @torch.no_grad()
    def __call__(self, question, context):
        enc = self.tok(question, context, truncation="only_second", max_length=384, return_offsets_mapping=True, return_tensors="pt")
        offsets, seq = enc.pop("offset_mapping")[0].tolist(), enc.sequence_ids(0)
        out = self.model(**{k: v.to(self.dev) for k, v in enc.items()})
        ps, pe = out.start_logits[0].softmax(-1).cpu().numpy(), out.end_logits[0].softmax(-1).cpu().numpy()
        ctx = [i for i, s in enumerate(seq) if s == 1]
        best, span = 0.0, None
        for i in ctx:
            for j in ctx:
                if i <= j < i + self.max_len and ps[i] * pe[j] > best:
                    best, span = float(ps[i] * pe[j]), (offsets[i][0], offsets[j][1])
        return {"answer": context[span[0]:span[1]] if span else "", "score": best}


def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDER, device=dev)
    reranker = CrossEncoder(RERANKER, device=dev)
    reader = Reader(READER, dev)

    P = embedder.encode([passage(c) for c in chunks], normalize_embeddings=True)
    out = []
    for q in questions:
        qv = embedder.encode(QUERY_PREFIX + q["question"], normalize_embeddings=True)
        cand = list(np.argsort(-(P @ qv))[:TOP_K])
        rr = reranker.predict([(q["question"], passage(chunks[i])) for i in cand])
        order = [int(cand[k]) for k in np.argsort(-rr)]

        # A. exact words from the reader
        words = {"answer": "", "score": 0.0}
        for i in order[:READ]:
            r = reader(q["question"], chunks[i]["text"])
            if r["answer"].strip() and r["score"] > words["score"]:
                words = {"answer": r["answer"].strip(), "score": r["score"]}

        # B. best sentence: each sentence is scored with its section title, so "16 weeks at full pay" keeps its meaning
        pairs = [(i, s) for i in order[:READ] for s in sentences(chunks[i]["text"])]
        ss = reranker.predict([(q["question"], f"{chunks[i]['section']}. {s}") for i, s in pairs])
        k = int(np.argmax(ss))

        out.append({
            "id": q["id"], "type": q["type"],
            "search_rank": next((k + 1 for k, i in enumerate(cand) if i in q["gold"]), None),
            "rerank_rank": next((k + 1 for k, i in enumerate(order) if i in q["gold"]), None),
            "passage_score": float(max(rr)),
            "words": words["answer"], "words_score": words["score"],
            "words_ok": words_correct(q, words["answer"]) if q["sources"] else False,
            "sentence": pairs[k][1], "sentence_score": float(ss[k]), "sentence_doc": chunks[pairs[k][0]]["doc"],
            "sentence_page": chunks[pairs[k][0]]["page"],
            "sentence_ok": overlaps_evidence(q, pairs[k][1]) if q["sources"] else False,
        })
    return out


APPROACHES = {   # name: (correct-answer field, confidence field)
    "A. exact words (reader)": ("words_ok", "words_score"),
    "B. best sentence (re-ranker)": ("sentence_ok", "sentence_score"),
}


def grade(rows, ok, conf, t):
    good = sum((r[conf] < t) if r["type"] == "trick" else (r[conf] >= t and r[ok]) for r in rows)
    return good / len(rows)


def tune(rows, ok, conf):
    grid = sorted({r[conf] for r in rows} | {-99})
    return max(grid, key=lambda t: (grade(rows, ok, conf, t), -t))


def evaluate(rows, ok, conf):
    folds = [[r for r in rows if r["id"] % 2 == 0], [r for r in rows if r["id"] % 2 == 1]]
    stats = {t: [0, 0] for t in ("easy", "hard", "trick")}
    wrong = 0
    for test, train in ((folds[0], folds[1]), (folds[1], folds[0])):
        t = tune(train, ok, conf)
        for r in test:
            answered = r[conf] >= t
            stats[r["type"]][0] += (not answered) if r["type"] == "trick" else (answered and r[ok])
            stats[r["type"]][1] += 1
            wrong += answered and not (r["type"] != "trick" and r[ok])
    return stats, wrong


if __name__ == "__main__":
    print(f"re-ranker: {RERANKER}")
    rows = run()
    (ROOT / "data" / f"answers_raw_{TAG}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ans = [r for r in rows if r["type"] != "trick"]
    hit = lambda key, k: 100 * sum(1 for r in ans if r[key] and r[key] <= k) / len(ans)
    print(f"right passage 1st:   search {hit('search_rank', 1):.1f}%  ->  after re-ranking {hit('rerank_rank', 1):.1f}%")
    print(f"right passage top 3: search {hit('search_rank', 3):.1f}%  ->  after re-ranking {hit('rerank_rank', 3):.1f}%")

    summary = {}
    print(f"\n{'graded on questions not used for tuning':<34} {'easy':>7} {'hard':>7} {'trick':>7} {'overall':>8} {'wrong answers':>14}")
    for name, (ok, conf) in APPROACHES.items():
        forced = 100 * np.mean([r[ok] for r in ans])
        stats, wrong = evaluate(rows, ok, conf)
        pct = {t: 100 * s[0] / s[1] for t, s in stats.items()}
        overall = 100 * sum(s[0] for s in stats.values()) / len(rows)
        print(f"{name:<34} {pct['easy']:6.0f}% {pct['hard']:6.0f}% {pct['trick']:6.0f}% {overall:7.0f}% {wrong:>14}"
              f"   (right when always answering: {forced:.0f}%)")
        summary[name] = {**pct, "overall": overall, "wrong": wrong, "always_answer": forced}

    t = tune(rows, "sentence_ok", "sentence_score")
    (ROOT / "data" / f"answer_results_{TAG}.json").write_text(json.dumps({"threshold": t, **summary}, indent=1), encoding="utf-8")
    print(f"\napp threshold (tuned on all questions): best sentence score >= {t:.2f}\n")
    for r, q in zip(rows, questions):
        answered = r["sentence_score"] >= t
        good = (not answered) if r["type"] == "trick" else (answered and r["sentence_ok"])
        shown = f"[{r['sentence_doc']} p{r['sentence_page']}] {r['sentence'][:90]}" if answered else "I couldn't find this"
        print(f"  {'OK' if good else 'XX'} [{r['type']:<5}] {r['sentence_score']:6.2f}  {q['question'][:55]:<55} -> {shown}")
