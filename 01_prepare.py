# Step 3a: read the PDFs and cut them into passages ("chunks") the search can return.
#
# Rules:
# - a chunk never crosses a page, so every answer can cite one page
# - a chunk stays inside one section and starts with its section title, so a passage
#   like "16 weeks at full pay" still knows it is about "6. Parental leave"
# - table rows are rewritten as "Column: value" text, so "Night | $2.75 per hour" reads
#   "Shift: Night · Shift premium: $2.75 per hour"
# Then each test question is linked to the chunk(s) holding its evidence sentence.
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "demo" / "samples"
OUT = ROOT / "data"
MAX_WORDS = 120

FOOTER = re.compile(r"^(Page \d+ of \d+|Fictional company created for a software demo|Meridian Freight Co\. · (Employee Handbook|IT & Security Policy|Customer Service Agreement))$")
HEADING = re.compile(r"^\d+(\.\d+)?\.?\s+[A-Z].{0,60}$")


def clean(s):
    return " ".join(s.replace("­", "").split())


def page_blocks(page):
    """Paragraphs and table rows of one page, in reading order. Table rows come back as lists of cells."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        lines = [(round(l["bbox"][0]), clean("".join(s["text"] for s in l["spans"]))) for l in b.get("lines", [])]
        lines = [(x, t) for x, t in lines if t and not FOOTER.match(t)]
        if not lines:
            continue
        xs = sorted({x for x, _ in lines})
        cols = [xs[0]] + [x for a, x in zip(xs, xs[1:]) if x - a > 40]     # left edges of columns
        if len(cols) > 1:                                                  # several columns = a table row
            cells = [" ".join(t for x, t in lines if c <= x < (cols[k + 1] if k + 1 < len(cols) else 1e9)) for k, c in enumerate(cols)]
            out.append(cells)
        else:
            out.append(clean(" ".join(t for _, t in lines)))
    return out


def sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", text)


def chunk_document(path):
    doc = pymupdf.open(path)
    title = doc[0].get_text("dict")["blocks"] and None
    chunks, section, sub, header = [], "", "", None
    for pno, page in enumerate(doc, start=1):
        buf = []

        def flush():
            if buf:
                head = " › ".join(h for h in (section, sub) if h)
                chunks.append({"doc": path.stem, "page": pno, "section": head, "text": " ".join(buf)})
                buf.clear()

        for blk in page_blocks(page):
            if isinstance(blk, list):                                  # table row
                if header is None:
                    header = blk
                    continue
                text = " · ".join(f"{h}: {c}" if h else c for h, c in zip(header + [""] * len(blk), blk)) + "."
            else:
                header = None
                if pno == 1 and title is None:                         # first block = document title
                    title = blk
                    continue
                if blk.startswith("Meridian Freight Co. · Version") or blk.startswith("Meridian Freight Co. · Standard"):
                    continue
                if HEADING.match(blk):
                    flush()
                    if re.match(r"^\d+\.\d+", blk):
                        sub = blk
                    else:
                        section, sub = blk, ""
                    continue
                text = blk
            # add the block, splitting by sentence when a chunk would grow too long
            for s in (sentences(text) if len(text.split()) > MAX_WORDS else [text]):
                if buf and len(" ".join(buf).split()) + len(s.split()) > MAX_WORDS:
                    flush()
                buf.append(s)
        flush()
    for c in chunks:
        c["title"] = title
    return chunks


def norm(s):
    return " ".join(s.split()).lower()


def gold_chunks(question, chunks):
    """Chunks that contain a question's evidence (or most of it, if a page break cut the sentence)."""
    ids = set()
    for src in question["sources"]:
        ev = norm(src["evidence"])
        for c in chunks:
            if c["doc"] != src["doc"]:
                continue
            t = norm(c["text"])
            half = len(ev) // 2
            if ev in t or (len(ev) > 50 and (ev[:half] in t or ev[half:] in t)):
                ids.add(c["id"])
    return sorted(ids)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    chunks = []
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        chunks += chunk_document(pdf)
    for i, c in enumerate(chunks):
        c["id"] = i
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    missing = 0
    for q in questions:
        q["gold"] = gold_chunks(q, chunks)
        q["pages"] = sorted({(chunks[i]["doc"], chunks[i]["page"]) for i in q["gold"]})
        if q["sources"] and not q["gold"]:
            print(f"question {q['id']}: evidence not found in any chunk")
            missing += 1
    (OUT / "chunks.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "questions_gold.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")

    words = [len(c["text"].split()) for c in chunks]
    print(f"{len(chunks)} chunks from {len(set(c['doc'] for c in chunks))} documents, "
          f"{min(words)}-{max(words)} words each (average {sum(words) / len(words):.0f})")
    print(f"questions linked to their chunks: {sum(1 for q in questions if q['gold'])}/{sum(1 for q in questions if q['sources'])}")
    for c in chunks[:4] + [c for c in chunks if "Night" in c["text"]][:1]:
        print(f"\n[{c['doc']} p{c['page']}] {c['section']}\n  {c['text'][:220]}")
