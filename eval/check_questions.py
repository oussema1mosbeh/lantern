# Checks the test questions: every evidence sentence and every answer must appear,
# word for word, in the document it points to. A typo here would make the exam unfair.
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def doc_text(name):
    raw = (ROOT / "documents" / "src" / f"{name}.html").read_text(encoding="utf-8")
    raw = re.sub(r"<(style|head)[\s\S]*?</\1>", " ", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return " ".join(text.split())


def norm(s):
    return " ".join(s.split())


questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
texts = {}
problems = 0
for q in questions:
    for src in q["sources"]:
        t = texts.setdefault(src["doc"], doc_text(src["doc"]))
        if norm(src["evidence"]) not in t:
            print(f"#{q['id']}: evidence not found in {src['doc']}: {src['evidence']!r}")
            problems += 1
        elif t.count(norm(src["evidence"])) > 1:
            print(f"#{q['id']}: evidence appears more than once in {src['doc']}: {src['evidence']!r}")
            problems += 1
    if q["answer"] and not any(norm(q["answer"]).lower() in texts[s["doc"]].lower() for s in q["sources"]):
        print(f"#{q['id']}: answer not found: {q['answer']!r}")
        problems += 1

print(Counter(q["type"] for q in questions), "questions")
print("problems:", problems)
