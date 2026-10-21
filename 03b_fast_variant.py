# Step 6 follow-up: the browser runs the re-ranker ~50x slower than Python, so test a lighter pipeline.
#   current: re-rank the 10 best passages, then score the sentences of the 3 best      (~3,000 tokens per question)
#   light:   skip passage re-ranking; score the sentences of the N best SEARCH results  (~1,000 tokens per question)
# Same grading as step 4 (two-fold thresholds), same 8-bit models as the browser.
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
answer = __import__("03_answer")
export = __import__("04_export")

if __name__ == "__main__":
    chunks, questions = answer.chunks, answer.questions
    emb, rer = export.OnnxModel("embedder"), export.OnnxModel("reranker")
    P = np.concatenate([emb([answer.passage(c) for c in chunks[i:i + 16]]) for i in range(0, len(chunks), 16)])
    tok = rer.tok
    for n in (3, 4, 5):
        rows, tokens = [], []
        for q in questions:
            qv = emb([answer.QUERY_PREFIX + q["question"]])[0]
            top = list(np.argsort(-(P @ qv))[:n])
            pairs = [(int(i), s) for i in top for s in answer.sentences(chunks[i]["text"])]
            texts = [f"{chunks[i]['section']}. {s}" for i, s in pairs]
            tokens.append(sum(len(tok(q["question"], t)["input_ids"]) for t in texts))
            ss = rer([q["question"]] * len(texts), texts)
            k = int(np.argmax(ss))
            rows.append({"id": q["id"], "type": q["type"], "sentence_score": float(ss[k]),
                         "sentence_ok": answer.overlaps_evidence(q, pairs[k][1]) if q["sources"] else False})
        ans = [r for r in rows if r["type"] != "trick"]
        pos = [r["sentence_score"] for r in ans if r["sentence_ok"]]
        neg = [r["sentence_score"] for r in rows if r["type"] == "trick"]
        auc = np.mean([(p > m) + .5 * (p == m) for p in pos for m in neg])
        stats, wrong = answer.evaluate(rows, "sentence_ok", "sentence_score")
        sure = [r for r in rows if r["sentence_score"] >= .3]
        print(f"top {n} search passages: ~{np.mean(tokens):.0f} tokens/question | right sentence {100 * np.mean([r['sentence_ok'] for r in ans]):.0f}% "
              f"| AUC {auc:.2f} | sure {sum(r['sentence_ok'] for r in sure)}/{len(sure)} right, {sum(r['type'] == 'trick' for r in sure)} tricks "
              f"| tricks not found {sum(r['sentence_score'] < .02 for r in rows if r['type'] == 'trick')}/11")
        json.dump(rows, open(ROOT / "data" / f"answers_light_top{n}.json", "w"), indent=1)
