# Step 6a: pre-index the sample documents for the web app, with the SAME 8-bit model the browser uses,
# so the demo opens instantly and gives exactly the answers measured in step 5.
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
export = __import__("04_export")
answer = __import__("03_answer")

if __name__ == "__main__":
    chunks = answer.chunks
    emb = export.OnnxModel("embedder")
    vecs = np.concatenate([emb([answer.passage(c) for c in chunks[i:i + 16]]) for i in range(0, len(chunks), 16)]).astype(np.float32)
    decision = json.loads((ROOT / "data" / "decision.json").read_text())

    docs = []
    for pdf in sorted((ROOT / "demo" / "samples").glob("*.pdf")):
        d = pymupdf.open(pdf)
        title = next(c["title"] for c in chunks if c["doc"] == pdf.stem)
        docs.append({"id": pdf.stem, "title": title, "file": f"samples/{pdf.name}", "pages": len(d)})

    out = {
        "collection": "Meridian Freight Co.",
        "docs": docs,
        "chunks": [{k: c[k] for k in ("doc", "page", "section", "text")} for c in chunks],
        "vectors": base64.b64encode(vecs.tobytes()).decode(),
        "dim": int(vecs.shape[1]),
        "levels": decision["levels"],
        # test questions the app answers well, plus one it must refuse (the dog), to show both behaviors
        "questions": [
            {"tag": "HR", "q": "Can I keep my unused vacation days for next year?"},
            {"tag": "IT", "q": "Do I need to turn on the VPN at home just to read my email?"},
            {"tag": "Contract", "q": "We paid for guaranteed delivery and the truck was late. What do we get back?"},
            {"tag": "HR", "q": "Do I need a doctor's note if I'm sick for two days?"},
            {"tag": "Pay", "q": "How much extra do I earn per hour for working nights?"},
            {"tag": "Contract", "q": "How heavy can one of our pallets be?"},
            {"tag": "HR", "q": "My father passed away. How many days off can I take?"},
            {"tag": "Not in the docs", "q": "Can I bring my dog to the office?"},
        ],
    }
    (ROOT / "demo" / "data").mkdir(exist_ok=True)
    (ROOT / "demo" / "data" / "samples.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(docs)} documents, {len(chunks)} passages, {vecs.nbytes / 1e3:.0f} kB of vectors")
