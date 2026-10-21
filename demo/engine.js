// Lantern's AI engine: the same pipeline as the Python scripts, running in the browser.
//   search    BGE-small turns the question into 384 numbers; the closest passages win
//   pinpoint  bge-reranker-base reads the question with every sentence of the 4 closest passages;
//             the best sentence is the answer
//   confidence  the best sentence's score decides: sure / not sure / not found
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL("./models/", location.href).href;

const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
ort.env.wasm.proxy = true;          // run the models in a background worker so the page never freezes

export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const TOP_K = 10, READ = 4;          // passages from search, passages read sentence by sentence
const M = {};

// ---------- loading the models (with progress, and kept in the browser cache for next visits) ----------
async function fetchBytes(url, onBytes) {
  let cache = null;
  try { cache = await caches.open("lantern-models-v1"); } catch { /* private mode: no cache */ }
  let res = cache ? await cache.match(url).catch(() => null) : null;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download ${url} (${res.status})`);
    if (cache) cache.put(url, res.clone()).catch(() => {});
  }
  const reader = res.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export async function loadModels(onProgress) {
  const manifest = await (await fetch("models/manifest.json")).json();
  const total = manifest.embedder.bytes + manifest.reranker.bytes;
  let got = 0;
  const tick = n => { got += n; onProgress({ loaded: got, total }); };
  const session = async kind => {
    const pieces = [];
    for (const f of manifest[kind].files) pieces.push(await fetchBytes(`models/${f}`, tick));
    const bytes = pieces.length === 1 ? pieces[0] : (() => {
      const all = new Uint8Array(pieces.reduce((s, p) => s + p.length, 0));
      let o = 0; for (const p of pieces) { all.set(p, o); o += p.length; }
      return all;                                       // the 4 pieces glued back into one model file
    })();
    return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  };
  [M.embTok, M.emb] = await Promise.all([AutoTokenizer.from_pretrained("embedder"), session("embedder")]);
  onProgress({ loaded: got, total, embedderReady: true });
  [M.rrTok, M.rr] = await Promise.all([AutoTokenizer.from_pretrained("reranker"), session("reranker")]);
  onProgress({ loaded: total, total, ready: true });
}

export const embedderReady = () => !!M.emb;

const feeds = enc => ({
  input_ids: new ort.Tensor("int64", enc.input_ids.data, enc.input_ids.dims),
  attention_mask: new ort.Tensor("int64", enc.attention_mask.data, enc.attention_mask.dims),
});

export async function embed(texts, batch = 8) {
  const out = [];
  for (let i = 0; i < texts.length; i += batch) {
    const enc = await M.embTok(texts.slice(i, i + batch), { padding: true, truncation: true });
    const h = (await M.emb.run(feeds(enc))).last_hidden_state;
    const [b, s, d] = h.dims;
    for (let k = 0; k < b; k++) {
      const v = h.data.slice(k * s * d, k * s * d + d);   // [CLS] token = the text's meaning
      let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
      out.push(v.map(x => x / n));
    }
  }
  return out;
}

async function rerank(question, texts, batch = 4) {
  const out = [];
  for (let i = 0; i < texts.length; i += batch) {
    const part = texts.slice(i, i + batch);
    const enc = await M.rrTok(part.map(() => question), { text_pair: part, padding: true, truncation: true });
    const logits = (await M.rr.run(feeds(enc))).logits.data;
    for (const l of logits) out.push(1 / (1 + Math.exp(-l)));   // relevance between 0 and 1
  }
  return out;
}

// ---------- the index: documents, passages and their vectors ----------
export function createIndex() { return { docs: [], chunks: [], vecs: [] }; }

export const docOf = (index, id) => index.docs.find(d => d.id === id);
export const passage = (index, c) => `${docOf(index, c.doc).title}. ${c.section}. ${c.text}`;

export function sentences(text) {
  return text.split(/(?<=[.!?])\s+(?=[A-Z"($0-9])/).filter(s => s.trim());
}

export async function ask(index, question, levels, onStage = () => {}) {
  const t0 = performance.now();
  onStage("search");
  const [qv] = await embed([QUERY_PREFIX + question]);
  const sims = index.vecs.map(v => { let s = 0; for (let k = 0; k < v.length; k++) s += v[k] * qv[k]; return s; });
  const cand = sims.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, TOP_K).map(x => x[1]);

  // the re-ranker reads every sentence of the closest passages (with its section title) and scores it.
  // Measured in 03b_fast_variant.py: as accurate as re-ranking whole passages first, with 4x less work.
  onStage("pinpoint");
  const pairs = cand.slice(0, READ).flatMap(i => sentences(index.chunks[i].text).map(s => ({ i, s, text: `${index.chunks[i].section}. ${s}` })));
  const byLength = [...pairs].sort((a, b) => a.text.length - b.text.length);     // similar lengths per batch = less padding
  const scores = await rerank(question, byLength.map(p => p.text), 8);
  byLength.forEach((p, k) => p.score = scores[k]);
  const ranked = [...pairs].sort((a, b) => b.score - a.score);
  const best = ranked[0];
  const level = best.score >= levels.sure ? "sure" : best.score >= levels.maybe ? "maybe" : "none";

  // other places worth checking: the best sentence of each other passage
  const seen = new Set([best.i]), others = [];
  for (const r of ranked) if (!seen.has(r.i) && others.length < 2) { seen.add(r.i); others.push(r); }

  // passages for Claude: the closest ones, best-scoring first
  const passScore = i => Math.max(...pairs.filter(p => p.i === i).map(p => p.score), 0);
  const order = cand.slice(0, 5).sort((a, b) => passScore(b) - passScore(a));
  return {
    level, best, others,
    passages: order.map(i => ({ chunk: index.chunks[i], score: passScore(i) })),
    stats: { searched: index.chunks.length, reread: READ, sentences: pairs.length, ms: performance.now() - t0 },
  };
}

// ---------- reading an uploaded PDF into passages (the JavaScript version of 01_prepare.py) ----------
const HEADING = /^\d+(\.\d+)?\.?\s+[A-Z].{0,60}$/;
const MAX_WORDS = 120;
const clean = s => s.replace(/­/g, "").replace(/\s+/g, " ").trim();

async function pageLines(page) {
  const { items } = await page.getTextContent();
  const rows = new Map();
  for (const it of items) {
    if (!it.str.trim()) continue;
    const y = Math.round(it.transform[5] / 2) * 2, size = Math.hypot(it.transform[2], it.transform[3]);
    if (!rows.has(y)) rows.set(y, []);
    rows.get(y).push({ x: it.transform[4], w: it.width, str: it.str, size });
  }
  return [...rows.entries()].sort((a, b) => b[0] - a[0]).map(([y, its]) => {
    its.sort((a, b) => a.x - b.x);
    const segs = [];                                     // pieces of a line far apart = table columns
    for (const it of its) {
      const last = segs[segs.length - 1];
      if (last && it.x - last.end < 18) { last.text += (it.x - last.end > 1.5 ? " " : "") + it.str; last.end = it.x + it.w; }
      else segs.push({ x: it.x, end: it.x + it.w, text: it.str });
    }
    return { y, size: Math.max(...its.map(i => i.size)), segs: segs.map(s => ({ x: s.x, text: clean(s.text) })) };
  });
}

export async function chunkPdf(pdf, docId, onPage = () => {}) {
  const pages = [];
  for (let p = 1; p <= pdf.numPages; p++) { pages.push(await pageLines(await pdf.getPage(p))); onPage(p, pdf.numPages); }
  // lines repeated on most pages are headers or footers
  const count = new Map();
  for (const lines of pages) for (const t of new Set(lines.map(l => l.segs.map(s => s.text).join(" ").replace(/\d+/g, "#")))) count.set(t, (count.get(t) || 0) + 1);
  const repeated = t => pdf.numPages > 2 && count.get(t.replace(/\d+/g, "#")) >= Math.ceil(pdf.numPages * .6);

  const chunks = [];
  let section = "", sub = "", header = null, title = "";
  pages.forEach((lines, k) => {
    const page = k + 1, buf = [];
    const flush = () => {
      if (buf.length) chunks.push({ doc: docId, page, section: [section, sub].filter(Boolean).join(" › "), text: buf.join(" ") });
      buf.length = 0;
    };
    const add = text => {
      const parts = text.split(/\s+/).length > MAX_WORDS ? sentences(text) : [text];
      for (const s of parts) {
        if (buf.length && (buf.join(" ") + " " + s).split(/\s+/).length > MAX_WORDS) flush();
        buf.push(s);
      }
    };
    let para = "", prev = null;
    const endPara = () => { if (para) add(para); para = ""; };
    for (const line of lines) {
      const text = line.segs.map(s => s.text).join(" ");
      if (!text || /^page \d+( of \d+)?$/i.test(text) || repeated(text)) continue;
      if (!title && page === 1) { title = text; prev = line; continue; }
      if (line.segs.length > 1) {                         // a table row
        endPara();
        const cells = line.segs.map(s => s.text);
        if (!header) header = cells;
        else add(cells.map((c, i) => header[i] ? `${header[i]}: ${c}` : c).join(" · ") + ".");
        prev = line; continue;
      }
      header = header && prev && prev.segs.length > 1 && line.segs[0].x > prev.segs[0].x + 30 ? header : null;
      if (HEADING.test(text)) {
        endPara(); flush();
        if (/^\d+\.\d+/.test(text)) sub = text; else { section = text; sub = ""; }
        prev = line; continue;
      }
      const gap = prev ? prev.y - line.y : 0;
      if (para && gap > line.size * 1.9) endPara();       // a bigger gap starts a new paragraph
      para += (para ? " " : "") + text;
      prev = line;
    }
    endPara(); flush();
  });
  return { title: title || docId, chunks };
}
