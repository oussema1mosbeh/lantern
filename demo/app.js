import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
import { ask, chunkPdf, createIndex, docOf, embed, loadModels, passage, sentences } from "./engine.js";

pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const mb = b => (b / 1e6).toFixed(0);
const isPhone = () => matchMedia("(max-width: 900px)").matches;
const ARROW = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';

const index = createIndex();
let levels = { sure: .3, maybe: .02 };
let suggestions = [];
const asked = new Set();
const pdfCache = new Map();
let busy = Promise.resolve();                       // the models answer one request at a time
const queue = fn => (busy = busy.then(fn, fn));

// ---------- the AI models: downloaded only after the visitor agrees ----------
// First visit: ask (about 330 MB). Later visits: the models are in the browser cache,
// so they load from this device without asking, since nothing is downloaded.
let progress = { loaded: 0, total: 1 };
let modelsReady = null;                             // the loading promise, once started
let waiting = [];                                   // questions and uploads waiting for the visitor's OK
let t0 = 0;

function startModels({ fromCache = false } = {}) {
  if (modelsReady) return modelsReady;
  t0 = performance.now();
  const bar = $("loadbar");
  bar.classList.remove("ask", "mini", "err");
  $("lbTitle").textContent = fromCache ? "Loading the AI from this device" : "Downloading the AI";
  $("lbMeta").textContent = "0%";
  $("lbNote").textContent = fromCache ? "No download needed: it was saved on your last visit."
    : "One time only. You can already type your question; it will be answered as soon as this finishes.";
  document.querySelectorAll(".consent").forEach(c => c.remove());
  document.querySelectorAll(".wait .waittxt").forEach(e => e.textContent = "Preparing the AI, one time only");
  modelsReady = loadModels(p => {
    progress = p;
    const pct = Math.round(100 * p.loaded / p.total);
    $("meter").style.width = pct + "%";
    const secs = (performance.now() - t0) / 1000, rate = p.loaded / Math.max(secs, .1);
    const left = rate > 0 ? Math.ceil((p.total - p.loaded) / rate) : 0;
    $("lbMeta").textContent = p.ready ? "" : `${pct}% · ${mb(p.loaded)} of ${mb(p.total)} MB${pct > 3 && left > 2 ? ` · about ${left > 90 ? Math.ceil(left / 60) + " min" : left + " s"} left` : ""}`;
    document.querySelectorAll(".waitpct").forEach(e => e.textContent = ` (${pct}%)`);
    if (p.ready) {
      bar.classList.add("done");
      $("lbTitle").textContent = "The AI is ready";
      $("lbNote").textContent = "It's saved on this device: next visits start instantly.";
      setTimeout(() => bar.classList.add("gone"), 1800);
    }
  }).catch(err => {
    console.error(err);
    modelsReady = null;
    bar.classList.add("err", "ask");
    $("lbTitle").textContent = "The AI couldn't load";
    $("lbMeta").textContent = "";
    $("lbNote").textContent = `${err.message}. Check your connection and try again.`;
    throw err;
  });
  const w = waiting; waiting = [];
  w.forEach(fn => fn());
  return modelsReady;
}

// resolves once the models are loaded; waits for the visitor's OK if the download hasn't started
const whenReady = () => modelsReady || new Promise(res => waiting.push(res)).then(() => modelsReady);

function askForModels() {
  const bar = $("loadbar");
  if (modelsReady) return;
  bar.classList.remove("mini", "gone");
  bar.classList.add("ask", "nudge");
  setTimeout(() => bar.classList.remove("nudge"), 900);
}

document.addEventListener("click", e => { if (e.target.closest("[data-consent]")) startModels(); });
$("notNow").onclick = () => { $("loadbar").classList.add("mini"); };

async function modelsCached() {
  try {
    const m = await (await fetch("models/manifest.json")).json();
    const cache = await caches.open("lantern-models-v1");
    for (const kind of ["embedder", "reranker"]) for (const f of m[kind].files) if (!await cache.match(`models/${f}`)) return false;
    return true;
  } catch { return false; }                         // private mode or blocked storage: ask
}
modelsCached().then(cached => { if (cached) startModels({ fromCache: true }); });

// ---------- the sample library ----------
let welcomeHtml = "";
async function boot() {
  const s = await (await fetch("data/samples.json")).json();
  levels = s.levels;
  suggestions = s.questions;
  const raw = Uint8Array.from(atob(s.vectors), c => c.charCodeAt(0));
  const all = new Float32Array(raw.buffer);
  index.docs.push(...s.docs.map(d => ({ ...d, source: "sample", status: "ready" })));
  index.chunks.push(...s.chunks);
  for (let i = 0; i < s.chunks.length; i++) index.vecs.push(all.subarray(i * s.dim, (i + 1) * s.dim));
  welcomeHtml = $("welcome").outerHTML;
  fillWelcome();
  renderDocs();
}

const chip = ({ tag, q }) => `<button type="button" class="sq" data-q="${esc(q)}"><span class="tag${tag === "Not in the docs" ? " trick" : ""}">${esc(tag)}</span><span class="qt">${esc(q)}</span>${ARROW}</button>`;

function fillWelcome() {
  const hasSamples = index.docs.some(d => d.source === "sample");
  $("sugg").innerHTML = hasSamples ? suggestions.map(chip).join("") : "";
  $("welcome").querySelector(".sugg-head").hidden = !hasSamples;
  $("ownCta").onclick = () => $("file").click();
}

function renderDocs() {
  const group = (title, list) => !list.length ? "" : `<div class="group"><div class="label"><span>${title}</span><span>${list.reduce((s, d) => s + (d.pages || 0), 0)} pages</span></div><ul>${list.map(d => {
    const n = index.chunks.filter(c => c.doc === d.id).length;
    return `<li><div class="doc${d.source === "upload" ? " up" : ""}${viewer.id === d.id ? " on" : ""}" data-id="${esc(d.id)}" role="button" tabindex="0" aria-label="Open ${esc(d.title)}">
      <span class="ic" aria-hidden="true"></span>
      <span class="t"><b title="${esc(d.title)}">${esc(d.title)}</b>
        <span>${d.status === "ready" ? `${d.pages} pages · ${n} passages` : esc(d.status)}</span>
        ${d.status !== "ready" ? `<div class="bar"><i style="width:${Math.round(100 * (d.progress || 0))}%"></i></div>` : ""}</span>
      ${d.status === "ready" ? `<button class="x" data-remove="${esc(d.id)}" aria-label="Remove ${esc(d.title)}" title="Remove from the library"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18"/></svg></button>` : ""}
    </div></li>`;
  }).join("")}</ul></div>`;
  const ups = index.docs.filter(d => d.source === "upload"), samples = index.docs.filter(d => d.source === "sample");
  $("docs").innerHTML = group("Your documents", ups) + group("Sample · Meridian Freight Co.", samples)
    || `<p class="empty">Your library is empty. Add a PDF to start.</p>`;
  $("chatTitle").textContent = ups.length ? "Your library" : samples.length ? "Meridian Freight Co." : "Your library";
  $("chatSub").textContent = ups.length
    ? `${index.docs.length} documents · ${index.docs.reduce((s, d) => s + (d.pages || 0), 0)} pages`
    : samples.length ? "Sample documents of a fictional company" : "Add a PDF to start";
}

$("docs").addEventListener("click", e => {
  const rm = e.target.closest("[data-remove]");
  if (rm) { e.stopPropagation(); return removeDoc(rm.dataset.remove); }
  const d = e.target.closest(".doc");
  if (d && docOf(index, d.dataset.id)?.status === "ready") { openPage(d.dataset.id, 1); if (isPhone()) setTab("page"); }
});
$("docs").addEventListener("keydown", e => { if (e.key === "Enter" && e.target.classList.contains("doc")) e.target.click(); });

function removeDoc(id) {
  const keep = index.chunks.map((c, i) => c.doc !== id ? i : -1).filter(i => i >= 0);
  index.vecs = keep.map(i => index.vecs[i]);
  index.chunks = keep.map(i => index.chunks[i]);
  index.docs = index.docs.filter(d => d.id !== id);
  pdfCache.delete(id);
  if (viewer.id === id) clearViewer();
  renderDocs();
  if ($("welcome")) fillWelcome();
  toast("Removed from the library");
}

// ---------- adding your own PDFs ----------
async function addFiles(files) {
  const pdfs = [...files].filter(f => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  if (!pdfs.length) return toast("Lantern reads PDF files");
  if (isPhone()) setTab("library");
  for (const f of pdfs) {
    const doc = { id: "up-" + Math.random().toString(36).slice(2, 9), title: f.name.replace(/\.pdf$/i, ""), pages: 0, source: "upload", status: "Reading pages…", progress: 0 };
    index.docs.push(doc);
    renderDocs();
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      doc.bytes = bytes;
      const pdf = await pdfjs.getDocument({ data: bytes.slice() }).promise;
      pdfCache.set(doc.id, Promise.resolve(pdf));
      doc.pages = pdf.numPages;
      if (pdf.numPages > 300) throw new Error(`${f.name} has ${pdf.numPages} pages; the demo reads up to 300`);
      const { chunks } = await chunkPdf(pdf, doc.id, (p, n) => { doc.progress = .3 * p / n; renderDocs(); });
      if (!chunks.length) throw new Error(`No text found in ${f.name}. Is it a scanned image?`);
      if (!progress.ready) {
        doc.status = modelsReady ? "Waiting for the AI…" : "Needs the AI download";
        renderDocs();
        if (!modelsReady) { askForModels(); toast("Download the AI to read your PDF"); if (isPhone()) setTab("chat"); }
      }
      try { await whenReady(); } catch { throw new Error(`The AI couldn't load, so “${doc.title}” wasn't read`); }
      doc.status = "Understanding…"; renderDocs();
      const vecs = [];
      for (let i = 0; i < chunks.length; i += 8) {
        vecs.push(...await queue(() => embed(chunks.slice(i, i + 8).map(c => passage(index, c)))));
        doc.progress = .3 + .7 * Math.min(1, (i + 8) / chunks.length); renderDocs();
      }
      if (!index.docs.includes(doc)) continue;          // removed while reading
      index.chunks.push(...chunks);
      index.vecs.push(...vecs);
      doc.status = "ready";
      renderDocs();
      toast(`“${doc.title}” is ready. Ask your first question.`);
      if (isPhone()) setTab("chat");
      input.focus();
    } catch (err) {
      console.error(err);
      index.docs = index.docs.filter(d => d !== doc);
      renderDocs();
      toast(err.message || "That PDF could not be read");
    }
  }
}
$("addBtn").onclick = () => $("file").click();
$("file").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
let dragDepth = 0;
addEventListener("dragenter", e => { if ([...e.dataTransfer.types].includes("Files")) { dragDepth++; $("dropzone").classList.add("on"); } });
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("dropzone").classList.remove("on"); } });
addEventListener("dragover", e => e.preventDefault());
addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; $("dropzone").classList.remove("on"); if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });

// ---------- asking ----------
const input = $("input");
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  input.classList.toggle("tall", input.scrollHeight > 160);
  $("send").disabled = !input.value.trim();
});
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });
$("form").onsubmit = e => { e.preventDefault(); const q = input.value.trim(); if (q) { input.value = ""; input.dispatchEvent(new Event("input")); submit(q); } };
addEventListener("keydown", e => {
  const typing = /INPUT|TEXTAREA/.test(document.activeElement?.tagName) || document.querySelector("dialog[open]");
  if (e.key === "/" && !typing) { e.preventDefault(); input.focus(); }
  if (!typing && viewer.id && (e.key === "ArrowLeft" || e.key === "ArrowRight")) $(e.key === "ArrowLeft" ? "prev" : "next").click();
});

const STEPS = [["search", "Searching every passage by meaning"], ["pinpoint", "Reading the 4 closest passages, sentence by sentence"]];

function submit(q) {
  if (!index.chunks.length) return toast("Add a PDF first, then ask your question");
  asked.add(q);
  $("welcome")?.remove();
  $("newBtn").disabled = false;
  const thread = $("thread");
  thread.insertAdjacentHTML("beforeend", `<div class="q"><div>${esc(q)}</div></div>`);
  const a = document.createElement("div");
  a.className = "a";
  const pct = Math.round(100 * progress.loaded / progress.total);
  a.innerHTML = `${modelsReady ? "" : `<div class="consent">
      <p><b>Lantern needs its AI to answer.</b> It runs on your device, so it downloads about 330 MB once, then answers instantly, even on your next visits.</p>
      <button class="btn primary" type="button" data-consent>Download and answer</button>
    </div>`}
    <ol class="steps">${progress.ready ? "" : `<li class="on wait"><span class="waittxt">${modelsReady ? "Preparing the AI, one time only" : "Waiting for your OK to download the AI"}</span><span class="waitpct">${modelsReady ? ` (${pct}%)` : ""}</span></li>`}
    ${STEPS.map(([k, t]) => `<li data-step="${k}">${t}</li>`).join("")}</ol>`;
  thread.append(a);
  a.scrollIntoView({ block: "end", behavior: "smooth" });
  queue(async () => {
    try {
      await whenReady();
      a.querySelector(".wait")?.classList.replace("on", "done");
      const res = await ask(index, q, levels, stage => {
        let past = true;
        a.querySelectorAll("[data-step]").forEach(li => {
          if (li.dataset.step === stage) { li.className = "on"; past = false; } else li.className = past ? "done" : "";
        });
      });
      renderAnswer(a, q, res);
    } catch (err) {
      console.error(err);
      a.innerHTML = `<div class="card"><p class="lead">Something went wrong: ${esc(err.message)}</p></div>`;
    }
  });
}

function marked(text, s) {
  const k = text.indexOf(s);
  return k < 0 ? esc(text) : esc(text.slice(0, k)) + "<mark>" + esc(s) + "</mark>" + esc(text.slice(k + s.length));
}

const LEVEL = {
  sure: { cls: "sure", label: "Found it", why: "Lantern is confident this sentence answers your question." },
  maybe: { cls: "maybe", label: "Not certain", why: "This sentence is related, but Lantern isn't sure it answers your question. Check the page." },
  none: { cls: "none", label: "Not found", why: "No sentence in your documents scored high enough to count as an answer." },
};

function renderAnswer(a, q, res) {
  const { level, best, others, stats } = res, c = index.chunks[best.i], L = LEVEL[level];
  const d = docOf(index, c.doc);
  const pct = Math.round(best.score * 100);
  const open = `data-open="${esc(c.doc)}|${c.page}" data-s="${esc(best.s)}"`;
  const other = (o, hidden) => {
    const oc = index.chunks[o.i];
    return `<button class="other${hidden ? " hidden" : ""}" data-open="${esc(oc.doc)}|${oc.page}" data-s="${esc(o.s)}"><small>${esc(docOf(index, oc.doc).title)} · page ${oc.page} · ${esc(oc.section)}</small>${esc(o.s)}</button>`;
  };
  const conf = `<span class="conf ${L.cls}" tabindex="0"><span class="pill ${L.cls}">${L.label}</span><span class="cbar" aria-hidden="true"><i style="width:${Math.max(4, pct)}%"></i></span><span class="tip" role="tooltip">${L.why} Match score: ${pct}%.</span></span>`;
  let body;
  if (level === "none") {
    body = `<p class="lead">I couldn't find this in your documents.</p>
      <p class="tips">Try other words (for example "leave" instead of "holiday"), or ask about one thing at a time.${claude.key ? "" : " Claude mode can also explain what the documents do and don't say."}</p>
      <div class="closest"><span class="sub">Closest passages, in case they help</span>${[best, ...others].map(o => other(o, false)).join("")}</div>`;
  } else {
    body = `${level === "maybe" ? `<p class="lead muted">This might answer it. Check the page to be sure:</p>` : ""}
      <blockquote class="quote"><mark>${esc(best.s)}</mark></blockquote>
      <div class="ctx"><span class="sec">${esc(c.section)}</span>${marked(c.text, best.s)}</div>`;
  }
  const rest = suggestions.filter(s => !asked.has(s.q) && index.docs.some(x => x.source === "sample")).slice(0, 2);
  a.innerHTML = `<article class="card ${level}">
    <div class="meta">${conf}${level === "none" ? "" : `<button class="src" ${open}><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3h9l5 5v13H6z"/></svg><b>${esc(d.title)}</b> · page ${c.page}</button>`}</div>
    ${body}
    <div class="acts">
      ${level === "none" ? "" : `<button class="btn ${level === "sure" ? "primary" : ""}" ${open}>Open page ${c.page} ${ARROW}</button>
        <button class="btn ghost" data-copy title="Copy the answer and its source"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>Copy</button>`}
      ${level !== "none" && others.length ? `<button class="btn ghost" data-more aria-expanded="false">Other places (${others.length})</button>` : ""}
      ${level === "none" && !claude.key ? `<button class="btn" data-tryclaude>Ask Claude instead</button>` : ""}
      <span class="trace" title="Passages searched → read → sentences scored → answer">${stats.searched} → ${stats.reread} → ${stats.sentences} → 1 · ${(stats.ms / 1000).toFixed(1)} s</span>
    </div>
    ${level !== "none" ? `<div class="others">${others.map(o => other(o, true)).join("")}</div>` : ""}
  </article>
  ${rest.length ? `<div class="next"><span>Try next</span>${rest.map(chip).join("")}</div>` : ""}`;
  a.querySelector("[data-more]")?.addEventListener("click", e => {
    const show = e.currentTarget.getAttribute("aria-expanded") !== "true";
    a.querySelectorAll(".others .other").forEach(o => o.classList.toggle("hidden", !show));
    e.currentTarget.setAttribute("aria-expanded", String(show));
    e.currentTarget.textContent = show ? "Hide other places" : `Other places (${others.length})`;
  });
  a.querySelector("[data-copy]")?.addEventListener("click", async e => {
    try {
      await navigator.clipboard.writeText(`“${best.s}”\n— ${d.title}, page ${c.page} (${c.section})`);
      e.currentTarget.lastChild.textContent = "Copied";
      setTimeout(() => { e.target.closest("[data-copy]").lastChild.textContent = "Copy"; }, 1600);
    } catch { toast("Copy isn't allowed in this browser"); }
  });
  a.querySelector("[data-tryclaude]")?.addEventListener("click", () => { pendingRetry = q; setMode("claude"); });
  if (level !== "none") {
    openPage(c.doc, c.page, [best.s]);
    if (isPhone()) $("tabs-badge").hidden = false;
  }
  if (claude.on && claude.key) claudeAnswer(a.querySelector(".card"), q, res);
}

$("thread").addEventListener("click", e => {
  const s = e.target.closest(".sq");
  if (s) return submit(s.dataset.q);
  const b = e.target.closest("[data-open]");
  if (b) {
    const [id, page] = b.dataset.open.split("|");
    openPage(id, +page, b.dataset.s ? [b.dataset.s] : []);
    if (isPhone()) setTab("page");
  }
});

$("newBtn").onclick = () => {
  $("thread").innerHTML = welcomeHtml;
  asked.clear();
  fillWelcome();
  $("newBtn").disabled = true;
  input.focus();
};

// ---------- answer mode: Quote (free, on device) or Claude (your key) ----------
const claude = { on: false, key: "" };
try { claude.key = sessionStorage.getItem("lantern-key") || ""; } catch { /* storage blocked */ }
let Anthropic = null, pendingRetry = null;
const SYSTEM = `You answer questions about an organization's documents, using only the numbered sources provided.
Cite the source of every statement with its number in square brackets, like [2].
If the sources don't answer the question, say clearly that the documents don't cover it, and mention the closest related information only if it helps.
Answer in plain language, in 1 to 4 sentences, in the same language as the question. Don't use Markdown.`;

function setMode(mode) {
  if (mode === "claude" && !claude.key) { $("keyInput").value = ""; $("keyDialog").showModal(); return; }
  claude.on = mode === "claude";
  document.querySelectorAll(".mode [role=radio]").forEach(b => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  if (claude.on && pendingRetry) { const q = pendingRetry; pendingRetry = null; submit(q); }
}
document.querySelectorAll(".mode [role=radio]").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("keyDialog").addEventListener("close", () => {
  const k = $("keyInput").value.trim();
  if ($("keyDialog").returnValue !== "ok" || !k) { pendingRetry = null; return; }
  claude.key = k;
  try { sessionStorage.setItem("lantern-key", k); } catch { /* kept in memory only */ }
  setMode("claude");
  toast("Claude will also write the next answers");
});

function withCites(text, n) {
  return esc(text).replace(/\[(\d+)\]/g, (m, k) => +k >= 1 && +k <= n ? `<button class="cite" data-cite="${+k - 1}" title="Open source ${k}">${k}</button>` : m);
}

async function claudeAnswer(card, q, res) {
  const sec = document.createElement("section");
  sec.className = "claude";
  sec.innerHTML = `<h4><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4z"/></svg>Claude's answer</h4><div class="txt typing"></div>`;
  card.append(sec);
  const txt = sec.querySelector(".txt");
  const srcs = res.passages.map(p => p.chunk);
  sec.addEventListener("click", e => {
    const b = e.target.closest("[data-cite]");
    if (!b) return;
    const c = srcs[+b.dataset.cite];
    openPage(c.doc, c.page, sentences(c.text));
    if (isPhone()) setTab("page");
  });
  try {
    if (!Anthropic) Anthropic = (await import("https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.132.0/+esm")).default;
    const client = new Anthropic({ apiKey: claude.key, dangerouslyAllowBrowser: true });
    const sources = srcs.map((c, k) => `[${k + 1}] ${docOf(index, c.doc).title}, page ${c.page}, ${c.section}\n${c.text}`).join("\n\n");
    const stream = client.beta.messages.stream({
      model: "claude-opus-5-5",
      max_tokens: 8000,
      output_config: { effort: "low" },
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: SYSTEM,
      messages: [{ role: "user", content: `Sources:\n\n${sources}\n\nQuestion: ${q}` }],
    });
    let text = "";
    for await (const ev of stream) {
      if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") {
        text += ev.delta.text;
        txt.innerHTML = withCites(text, srcs.length);
        sec.scrollIntoView({ block: "nearest" });
      }
    }
    const final = await stream.finalMessage();
    if (final.stop_reason === "refusal") txt.innerHTML = `<span class="err">Claude declined to answer this question.</span>`;
    else if (!text.trim()) txt.innerHTML = `<span class="err">Claude returned an empty answer.</span>`;
  } catch (err) {
    console.error(err);
    let msg = err.message || "Claude could not be reached.";
    if (Anthropic && err instanceof Anthropic.AuthenticationError) {
      msg = "That API key was refused. Choose Claude again to enter another one.";
      claude.key = "";
      try { sessionStorage.removeItem("lantern-key"); } catch { /* ignore */ }
      setMode("quote");
    } else if (Anthropic && err instanceof Anthropic.RateLimitError) msg = "Claude is busy or your key hit its limit. Try again in a minute.";
    else if (Anthropic && err instanceof Anthropic.APIConnectionError) msg = "Couldn't reach Claude. Check your connection.";
    txt.innerHTML = `<span class="err">${esc(msg)}</span>`;
  } finally {
    txt.classList.remove("typing");
  }
}

// ---------- the page viewer ----------
const viewer = { id: null, page: 1, targets: [], zoom: 1 };
let renderToken = 0;

function getPdf(id) {
  if (!pdfCache.has(id)) {
    const d = docOf(index, id);
    pdfCache.set(id, pdfjs.getDocument(d.file ? { url: d.file } : { data: d.bytes.slice() }).promise);
  }
  return pdfCache.get(id);
}

function clearViewer() {
  viewer.id = null;
  $("viewTitle").textContent = "No page open";
  $("viewPage").textContent = "The answer's page will open here";
  $("pages").innerHTML = "";
  ["prev", "next", "zoomIn", "zoomOut", "zoomFit"].forEach(k => $(k).disabled = true);
}

function openPage(id, page, targets = []) {
  Object.assign(viewer, { id, page, targets });
  renderDocs();
  renderPage();
}
$("prev").onclick = () => { if (viewer.page > 1) { viewer.page--; viewer.targets = []; renderPage(); } };
$("next").onclick = () => { viewer.page++; viewer.targets = []; renderPage(); };
$("zoomIn").onclick = () => { viewer.zoom = Math.min(2.5, viewer.zoom * 1.25); renderPage(true); };
$("zoomOut").onclick = () => { viewer.zoom = Math.max(.5, viewer.zoom / 1.25); renderPage(true); };
$("zoomFit").onclick = () => { viewer.zoom = 1; renderPage(true); };

async function renderPage(keepScroll = false) {
  const token = ++renderToken, d = docOf(index, viewer.id);
  if (!d) return;
  const pdf = await getPdf(viewer.id);
  viewer.page = Math.min(Math.max(1, viewer.page), pdf.numPages);
  const page = await pdf.getPage(viewer.page);
  if (token !== renderToken) return;
  const c = index.chunks.find(c => c.doc === d.id && c.page === viewer.page && c.section);
  $("viewTitle").textContent = d.title;
  $("viewPage").textContent = `Page ${viewer.page} of ${pdf.numPages}${c ? " · " + c.section.split(" › ")[0] : ""}`;
  $("prev").disabled = viewer.page <= 1;
  $("next").disabled = viewer.page >= pdf.numPages;
  $("zoomIn").disabled = viewer.zoom >= 2.5;
  $("zoomOut").disabled = viewer.zoom <= .5;
  $("zoomFit").disabled = viewer.zoom === 1;

  const box = $("pages"), fit = Math.min(box.clientWidth - 44, 900);
  const vp = page.getViewport({ scale: viewer.zoom * fit / page.getViewport({ scale: 1 }).width });
  // on big screens the whole interface is scaled up with CSS zoom; render the page with more pixels to match
  const uiZoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
  const dpr = Math.min((devicePixelRatio || 1) * uiZoom, 3);
  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(vp.width * dpr); canvas.height = Math.floor(vp.height * dpr);
  canvas.style.width = vp.width + "px"; canvas.style.height = vp.height + "px";
  await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null }).promise;
  if (token !== renderToken) return;
  const sheet = document.createElement("div");
  sheet.className = "sheet";
  sheet.style.width = vp.width + "px";
  sheet.append(canvas);
  const rects = await highlightRects(page, vp, viewer.targets);
  rects.forEach(([x, y, w, h], k) => sheet.insertAdjacentHTML("beforeend",
    `<div class="hl" style="left:${x - 2}px;top:${y - 1}px;width:${w + 4}px;height:${h + 2}px"></div>${k === 0 ? `<span class="hltag" style="top:${y - 1}px">Answer</span>` : ""}`));
  if (token !== renderToken) return;
  const top = box.scrollTop;
  box.replaceChildren(sheet);
  const first = sheet.querySelector(".hl");
  if (keepScroll) box.scrollTop = top;
  else if (first) box.scrollTo({ top: Math.max(0, parseFloat(first.style.top) - box.clientHeight / 3), behavior: "smooth" });
  else box.scrollTo({ top: 0 });
}

// find the sentence among the page's text pieces, ignoring spaces, and return rectangles to light up
async function highlightRects(page, vp, targets) {
  if (!targets.length) return [];
  const { items } = await page.getTextContent();
  let flat = "";
  const map = [];
  items.forEach((it, ii) => { for (let ci = 0; ci < it.str.length; ci++) if (!/\s/.test(it.str[ci])) { flat += it.str[ci].toLowerCase(); map.push([ii, ci]); } });
  const keys = targets.flatMap(t => t.includes(" · ")                       // a table row: light up each cell
    ? t.replace(/\.$/, "").split(" · ").map(p => p.replace(/^[^:]{1,40}:\s*/, "")) : [t]);
  const spans = new Map();
  let from = 0;
  for (const t of keys) {
    const key = t.replace(/\s+/g, "").toLowerCase();
    if (key.length < 2) continue;
    let at = flat.indexOf(key, from);
    if (at < 0) at = flat.indexOf(key);
    if (at < 0) continue;
    from = at + key.length;
    for (let k = at; k < at + key.length; k++) {
      const [ii, ci] = map[k], s = spans.get(ii) || [ci, ci];
      spans.set(ii, [Math.min(s[0], ci), Math.max(s[1], ci)]);
    }
  }
  const rects = [];
  for (const [ii, [a, b]] of spans) {
    const it = items[ii], [, , c, d, x, y] = it.transform, fh = Math.hypot(c, d), L = it.str.length;
    const r = vp.convertToViewportRectangle([x + it.width * a / L, y - fh * .26, x + it.width * (b + 1) / L, y + fh * .9]);
    rects.push([Math.min(r[0], r[2]), Math.min(r[1], r[3]), Math.abs(r[2] - r[0]), Math.abs(r[3] - r[1])]);
  }
  // join pieces on the same line into one bar
  rects.sort((p, q) => p[1] - q[1] || p[0] - q[0]);
  const out = [];
  for (const r of rects) {
    const l = out[out.length - 1];
    if (l && Math.abs(l[1] - r[1]) < 3 && r[0] - (l[0] + l[2]) < 14) { l[2] = Math.max(l[0] + l[2], r[0] + r[2]) - l[0]; l[3] = Math.max(l[3], r[3]); }
    else out.push([...r]);
  }
  return out;
}
addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(() => viewer.id && renderPage(true), 200); });

// ---------- the library drawer on medium screens ----------
const drawerMode = () => matchMedia("(min-width: 901px) and (max-width: 1180px)").matches;
function setDrawer(open) {
  $("app").classList.toggle("libopen", open);
  $("libBtn").setAttribute("aria-expanded", String(open));
}
$("libBtn").onclick = () => setDrawer(!$("app").classList.contains("libopen"));
$("scrim").onclick = () => setDrawer(false);
addEventListener("keydown", e => { if (e.key === "Escape" && $("app").classList.contains("libopen")) setDrawer(false); });
$("docs").addEventListener("click", e => { if (drawerMode() && e.target.closest(".doc") && !e.target.closest("[data-remove]")) setDrawer(false); });
matchMedia("(min-width: 901px) and (max-width: 1180px)").addEventListener("change", () => setDrawer(false));

// ---------- dialogs, phone tabs, toasts ----------
$("howBtn").onclick = () => $("howDialog").showModal();
document.querySelectorAll("dialog").forEach(d => d.addEventListener("click", e => { if (e.target === d) d.close(); }));

function setTab(tab) {
  $("app").dataset.tab = tab;
  document.querySelectorAll(".tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  if (tab === "page") { $("tabs-badge").hidden = true; if (viewer.id) renderPage(); }
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), 3200);
}

boot().then(() => {
  const q = new URLSearchParams(location.search).get("q");     // shareable links: ?q=your question
  if (q) submit(q.slice(0, 500));
}).catch(err => { console.error(err); toast("Could not load the sample documents"); });
