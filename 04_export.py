# Step 5: shrink the two models so they run in the browser, then check nothing broke.
#
#   1. export each PyTorch model to ONNX (a format the browser runtime understands)
#   2. quantize to 8-bit: every weight stored in 1 byte instead of 4, so ~4x smaller
#   3. re-run the step 4 test with the 8-bit models on the CPU (like a visitor's laptop):
#      same answers? how fast?
#
# Files are written in the layout Transformers.js expects: demo/models/<name>/{config, tokenizer, onnx/model_quantized.onnx}
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
answer = __import__("03_answer")          # reuse the passages, sentences and grading of step 4

MODELS = {
    "embedder": ("BAAI/bge-small-en-v1.5", AutoModel),
    "reranker": ("BAAI/bge-reranker-base", AutoModelForSequenceClassification),
}
OUT = ROOT / "demo" / "models"
TMP = ROOT / "data" / "onnx"


class Wrap(torch.nn.Module):
    def __init__(self, m, kind):
        super().__init__()
        self.m, self.kind = m, kind

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state if self.kind == "embedder" else out.logits


def export(kind):
    mid, cls = MODELS[kind]
    dst = OUT / kind
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(mid)
    # copy the ORIGINAL tokenizer files: files re-saved by transformers v5 are read differently by
    # Transformers.js 3 (it split "Can" into the wrong pieces, so the browser model read gibberish)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(hf_hub_download(mid, f), dst / f)
    AutoConfig.from_pretrained(mid).save_pretrained(dst)
    model = cls.from_pretrained(mid).eval()
    sample = tok(["a question", "a longer passage of text"], ["a passage", "another"] if kind == "reranker" else None,
                 padding=True, return_tensors="pt")
    full = TMP / f"{kind}.onnx"
    torch.onnx.export(Wrap(model, kind), (sample["input_ids"], sample["attention_mask"]), str(full),
                      input_names=["input_ids", "attention_mask"],
                      output_names=["last_hidden_state" if kind == "embedder" else "logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                      opset_version=14, dynamo=False)
    q = dst / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(full), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    print(f"{kind}: {full.stat().st_size / 1e6:.0f} MB -> {q.stat().st_size / 1e6:.0f} MB (8-bit)")
    split(q)


PART = 70_000_000      # GitHub refuses files over 100 MB, so big models are stored in pieces


def split(path):
    """Cut a model file into numbered pieces; the app downloads them and glues them back together."""
    data = path.read_bytes()
    if len(data) <= PART:
        return
    parts = [data[i:i + PART] for i in range(0, len(data), PART)]
    for k, p in enumerate(parts):
        path.with_name(f"{path.name}.part{k}").write_bytes(p)
    (path.parent / "parts.json").write_text(json.dumps({"file": path.name, "parts": len(parts), "bytes": len(data)}))
    path.replace(TMP / f"{path.parent.parent.name}_{path.name}")     # keep the whole file outside demo/ for the check
    print(f"  split into {len(parts)} pieces")


def model_file(kind):
    whole = OUT / kind / "onnx" / "model_quantized.onnx"
    return whole if whole.exists() else TMP / f"{kind}_model_quantized.onnx"


class OnnxModel:
    """The 8-bit models, run like the browser will run them: ONNX Runtime on the CPU."""

    def __init__(self, kind):
        self.tok = AutoTokenizer.from_pretrained(OUT / kind)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4                    # a typical laptop browser gets a few threads
        self.s = ort.InferenceSession(str(model_file(kind)), opts, providers=["CPUExecutionProvider"])
        self.kind = kind

    def __call__(self, a, b=None):
        enc = self.tok(a, b, padding=True, truncation=True, max_length=512, return_tensors="np")
        out = self.s.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        if self.kind == "embedder":                      # BGE uses the first token ([CLS]) as the sentence vector
            v = out[:, 0]
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return 1 / (1 + np.exp(-out[:, 0]))              # re-ranker: relevance between 0 and 1


def check():
    chunks, questions = answer.chunks, answer.questions
    emb, rer = OnnxModel("embedder"), OnnxModel("reranker")
    t0 = time.perf_counter()
    P = np.concatenate([emb([answer.passage(c) for c in chunks[i:i + 16]]) for i in range(0, len(chunks), 16)])
    index_time = time.perf_counter() - t0

    full = {r["id"]: r for r in json.loads((ROOT / "data" / "answers_raw_bge-base.json").read_text())}
    levels = json.loads((ROOT / "data" / "decision.json").read_text())["levels"]
    same, times, rows = 0, [], []
    for q in questions:
        t = time.perf_counter()
        qv = emb([answer.QUERY_PREFIX + q["question"]])[0]
        cand = list(np.argsort(-(P @ qv))[:answer.TOP_K])
        rr = rer([q["question"]] * len(cand), [answer.passage(chunks[i]) for i in cand])
        order = [int(cand[k]) for k in np.argsort(-rr)]
        pairs = [(i, s) for i in order[:answer.READ] for s in answer.sentences(chunks[i]["text"])]
        ss = rer([q["question"]] * len(pairs), [f"{chunks[i]['section']}. {s}" for i, s in pairs])
        times.append(time.perf_counter() - t)
        k = int(np.argmax(ss))
        same += pairs[k][1] == full[q["id"]]["sentence"]
        ok = answer.overlaps_evidence(q, pairs[k][1]) if q["sources"] else False
        rows.append((q["type"], ok, float(ss[k])))

    def band(s):
        return "sure" if s >= levels["sure"] else "maybe" if s >= levels["maybe"] else "none"
    right = sum(ok for t, ok, _ in rows if t != "trick")
    sure = [(t, ok) for t, ok, s in rows if band(s) == "sure"]
    print(f"\nindexing the 3 documents ({len(chunks)} passages): {index_time:.1f} s")
    print(f"answering one question: {np.median(times):.2f} s typical, {max(times):.2f} s slowest (CPU, 4 threads)")
    print(f"same sentence chosen as the full-size models: {same}/{len(questions)}")
    print(f"right sentence (always answering): {100 * right / sum(1 for t, *_ in rows if t != 'trick'):.0f}%")
    print(f"'sure' answers: {sum(ok for _, ok in sure)}/{len(sure)} right, tricks among them: {sum(t == 'trick' for t, _ in sure)}")
    print(f"tricks answered 'couldn't find': {sum(band(s) == 'none' for t, _, s in rows if t == 'trick')}/11")


if __name__ == "__main__":
    if "--check" not in sys.argv:
        for kind in MODELS:
            export(kind)
    check()
