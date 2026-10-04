/* Forensic Examination Desk: client for the Forensics Agent API (same origin).
   All dynamic content is inserted with textContent; model output is never parsed as HTML. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MIN_WORDS = 20;
  const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
  const HISTORY_KEY = "fx_history_v1";
  const KEY_KEY = "fx_access_key";

  const INSTRUMENTS = {
    text_dl: ["Neural text classifier", "DeBERTa-v3, trained on three corpora"],
    text_slm: ["Statistical probe", "Binoculars perplexity ratio (advisory)"],
    text_llm: ["Language-model reviewer", "Independent written assessment"],
    image_dl: ["Manipulation network", "EfficientNet-B4 with Grad-CAM"],
    image_clip: ["Foundation-model probe", "CLIP ViT-L/14 linear probe"],
    image_general: ["General synthetic-image probe", "For images without a face"],
    image_vlm: ["Vision-model reviewer", "Independent written assessment"],
  };
  const VERDICT_WORD = { synthetic: "Synthetic", authentic: "Authentic", uncertain: "Inconclusive" };
  const VERDICT_SUB = {
    text: { synthetic: "Likely written or heavily rewritten by an AI language model", authentic: "No indication of AI authorship", uncertain: "The evidence does not support a firm conclusion" },
    image: { synthetic: "Likely AI-generated or manipulated", authentic: "No indication of manipulation or AI generation", uncertain: "The evidence does not support a firm conclusion" },
  };
  const EVIDENCE_KIND = { generator_fingerprint: "Generator profile", case_study: "Earlier case", literature: "Literature note" };

  let thresholds = null;
  let imageFile = null;
  let imageUrl = null;
  let busy = false;
  let timer = null;

  /* ---------- helpers ---------- */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const pct = (x) => `${Math.round(x * 100)}%`;
  const num = (x) => (typeof x === "number" ? x.toFixed(2) : "n/a");
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  function fmtBytes(n) { return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`; }
  function fmtDate(iso) {
    const d = iso ? new Date(iso) : new Date();
    return d.toLocaleString(undefined, { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  function authHeaders() {
    let key = "";
    try { key = sessionStorage.getItem(KEY_KEY) || ""; } catch (_) { /* storage unavailable */ }
    return key ? { "X-API-Key": key } : {};
  }

  /* ---------- status line ---------- */
  async function loadHealth() {
    const set = (dot, text, state, label) => { $(dot).className = `dot ${state}`; $(text).textContent = label; };
    try {
      const r = await fetch("/health");
      const j = await r.json();
      set("dot-api", "st-api", "ok", "Service online");
      const ready = j.text_dl_model && j.image_dl_model && j.calibration_file;
      set("dot-models", "st-models", ready ? "ok" : "bad", ready ? "Instruments loaded" : "Models missing");
      set("dot-judge", "st-judge", j.llm_configured ? "ok" : "warn", j.llm_configured ? "Model reviewer active" : "Model reviewer off");
      thresholds = j.thresholds || null;
    } catch (_) {
      set("dot-api", "st-api", "bad", "Service unreachable");
      set("dot-models", "st-models", "", "");
      set("dot-judge", "st-judge", "", "");
    }
  }

  /* ---------- tabs ---------- */
  function selectTab(which) {
    for (const t of ["text", "image"]) {
      const on = t === which;
      $(`tab-${t}`).setAttribute("aria-selected", String(on));
      $(`tab-${t}`).tabIndex = on ? 0 : -1;
      $(`pane-${t}`).hidden = !on;
    }
  }
  for (const t of ["text", "image"]) {
    $(`tab-${t}`).addEventListener("click", () => selectTab(t));
    $(`tab-${t}`).addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const o = t === "text" ? "image" : "text"; selectTab(o); $(`tab-${o}`).focus(); }
    });
  }

  /* ---------- text intake ---------- */
  function wordCount(s) { const m = s.trim().match(/\S+/g); return m ? m.length : 0; }
  function updateText() {
    const n = wordCount($("text-input").value);
    $("word-count").textContent = `${n.toLocaleString()} word${n === 1 ? "" : "s"}`;
    const enough = n >= MIN_WORDS;
    $("word-hint").textContent = enough ? "Ready" : `${MIN_WORDS - n} more needed`;
    $("word-hint").className = enough ? "" : (n > 0 ? "bad" : "");
    $("run-text").disabled = !enough || busy;
  }
  $("text-input").addEventListener("input", updateText);
  $("clear-text").addEventListener("click", () => { $("text-input").value = ""; updateText(); $("text-input").focus(); });
  $("run-text").addEventListener("click", () => submit("text"));

  /* ---------- image intake ---------- */
  function setImage(file) {
    if (!file) return;
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { showError("Unsupported file", "Choose a JPEG, PNG or WebP image."); return; }
    if (file.size > MAX_IMAGE_BYTES) { showError("File too large", `The image is ${fmtBytes(file.size)}; the limit is 10 MB.`); return; }
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageFile = file;
    imageUrl = URL.createObjectURL(file);
    const img = $("preview-img");
    img.onload = () => { $("pv-dim").textContent = `${img.naturalWidth} x ${img.naturalHeight} px`; };
    img.src = imageUrl;
    $("pv-name").textContent = file.name || "Pasted image";
    $("pv-size").textContent = fmtBytes(file.size);
    $("preview").hidden = false;
    $("dropzone").hidden = true;
    $("run-image").disabled = busy;
    selectTab("image");
  }
  function clearImage() {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageFile = null; imageUrl = null;
    $("preview").hidden = true; $("dropzone").hidden = false; $("file-input").value = "";
    $("run-image").disabled = true;
  }
  const dz = $("dropzone");
  dz.addEventListener("click", () => $("file-input").click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); setImage(e.dataTransfer.files[0]); });
  $("file-input").addEventListener("change", (e) => setImage(e.target.files[0]));
  $("clear-image").addEventListener("click", clearImage);
  $("run-image").addEventListener("click", () => submit("image"));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { e.preventDefault(); setImage(item.getAsFile()); }
  });

  /* ---------- history ---------- */
  function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch (_) { return []; } }
  function pushHistory(entry) {
    const list = [entry, ...loadHistory().filter((x) => x.id !== entry.id)].slice(0, 12);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch (_) { /* ignore */ }
    renderHistory();
  }
  function renderHistory() {
    const list = loadHistory();
    const ol = $("history-list");
    ol.replaceChildren();
    if (!list.length) { ol.append(h("li", { class: "empty", text: "No cases yet in this browser." })); return; }
    for (const c of list) {
      ol.append(h("li", {}, h("button", { type: "button", onclick: () => openStored(c.id) },
        h("span", { class: "h-id", text: `Case ${c.id.slice(0, 8).toUpperCase()}` }),
        h("span", { class: "h-meta", text: `${c.modality === "text" ? "Text" : "Image"} - ${fmtDate(c.ts)}` }),
        h("span", { class: `chip h-verdict ${c.verdict || "neutral"}`, text: VERDICT_WORD[c.verdict] || "Pending" }))));
    }
  }

  /* ---------- views ---------- */
  const view = $("view");
  function showEmpty() {
    view.replaceChildren(h("div", { class: "empty-state" },
      h("h2", { text: "No examination in progress" }),
      h("p", { text: "Submit a passage of text or an image. The instruments examine it independently, and a verifier reconciles their readings into a single finding with the supporting material cited." }),
      h("ol", {},
        h("li", { text: "Provide the material on the left." }),
        h("li", { text: "Review the finding, the individual instrument readings and the reference notes." }),
        h("li", { text: "Where the system refers a case for review, record your own determination so it informs later cases." }))));
  }
  function showError(title, detail) {
    view.replaceChildren(h("div", { class: "error-box", role: "alert" }, h("h2", { text: title }), h("p", { text: detail })));
  }
  function showWorking(modality) {
    const started = Date.now();
    const el = h("span", { class: "elapsed", text: "0 s" });
    view.replaceChildren(h("div", { class: "working" },
      h("h2", { text: "Examination in progress" }),
      h("p", { text: modality === "image" ? "Locating faces, running the image instruments and retrieving reference notes." : "Running the text instruments and retrieving reference notes." }),
      h("div", { class: "progress", role: "progressbar", "aria-label": "Working" }),
      el));
    clearInterval(timer);
    timer = setInterval(() => { el.textContent = `${Math.round((Date.now() - started) / 1000)} s elapsed`; }, 500);
  }

  /* ---------- request ---------- */
  function describeFailure(status, detail) {
    if (status === 401) { $("key-bar").hidden = false; return ["Access key required", "This service is protected. Enter the access key in the panel on the left and try again."]; }
    if (status === 429) return ["Too many requests", "Wait a minute before submitting again."];
    if (status === 503) return ["Service busy", "Another examination is still running. Try again shortly."];
    if (status === 413 || status === 415 || status === 400) return ["The material was not accepted", detail || "Check the file type, size or length."];
    return ["The examination could not be completed", detail || `The service returned status ${status}.`];
  }
  async function submit(modality) {
    if (busy) return;
    busy = true; $("run-text").disabled = true; $("run-image").disabled = true;
    showWorking(modality);
    try {
      let resp;
      if (modality === "text") {
        resp = await fetch("/analyze/text", { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify({ text: $("text-input").value }) });
      } else {
        const fd = new FormData();
        fd.append("file", imageFile, imageFile.name || "pasted.png");
        resp = await fetch("/analyze/image", { method: "POST", headers: authHeaders(), body: fd });
      }
      let body = null;
      try { body = await resp.json(); } catch (_) { /* non-JSON error */ }
      if (!resp.ok) { const [t, d] = describeFailure(resp.status, body && typeof body.detail === "string" ? body.detail : ""); showError(t, d); return; }
      pushHistory({ id: body.job_id, ts: new Date().toISOString(), modality: body.modality, verdict: body.verdict });
      renderReport(normaliseLive(body), { sourceUrl: modality === "image" ? imageUrl : null });
    } catch (err) {
      showError("Cannot reach the service", "Check that the server is running and try again.");
    } finally {
      clearInterval(timer); busy = false; updateText(); $("run-image").disabled = !imageFile;
    }
  }
  async function openStored(id) {
    showWorking("text");
    clearInterval(timer);
    try {
      const resp = await fetch(`/jobs/${encodeURIComponent(id)}`, { headers: authHeaders() });
      if (!resp.ok) { const [t, d] = describeFailure(resp.status, ""); showError(t, d); return; }
      renderReport(normaliseStored(await resp.json()), { sourceUrl: null });
    } catch (_) { showError("Cannot reach the service", "Check that the server is running and try again."); }
  }

  /* ---------- normalisation: live response and stored record share one shape ---------- */
  function normaliseLive(b) {
    return { id: b.job_id, modality: b.modality, verdict: b.verdict, confidence: b.confidence, fused: b.fused_score, escalate: b.escalate_to_human,
      reflexion: b.reflexion_used, notes: b.verifier_notes, tools: b.tool_verdicts, evidence: b.evidence, summary: b.summary,
      gradcam: b.gradcam_url, created: null, determination: null };
  }
  function normaliseStored(rec) {
    const r = rec.report || {};
    const tools = (r.tool_verdicts || []).map((v) => ({ tool: v.tool, score: v.score, confidence: v.confidence, explanation: v.explanation, error: v.error, features: v.features || {} }));
    const hasCam = tools.some((v) => v.features && v.features.gradcam_file);
    return { id: rec.job_id, modality: rec.modality, verdict: rec.final_verdict, confidence: rec.final_confidence, fused: rec.fused_score, escalate: rec.escalate,
      reflexion: !!r.reflexion_used, notes: rec.verifier_notes, tools, evidence: r.evidence || [], summary: r.summary,
      gradcam: hasCam ? `/jobs/${rec.job_id}/gradcam` : null, created: rec.created_at, determination: rec.human_correction };
  }

  /* ---------- report ---------- */
  function meter(fused, modality) {
    if (typeof fused !== "number") return null;
    const band = thresholds && thresholds[modality];
    const lo = band ? band.lo : 0.35;
    const hi = band ? band.hi : 0.65;
    return h("div", { class: "meter" },
      h("div", { class: "meter-track", role: "img", "aria-label": `Combined reading ${num(fused)} on a scale from authentic (0) to synthetic (1)` },
        h("span", { class: "z-a", style: `width:${lo * 100}%` }), h("span", { class: "z-u", style: `width:${(hi - lo) * 100}%` }), h("span", { class: "z-s", style: `width:${(1 - hi) * 100}%` }),
        h("span", { class: "meter-mark", style: `left:${clamp01(fused) * 100}%` })),
      h("div", { class: "meter-scale" }, h("span", { text: "Authentic" }), h("span", { text: "Inconclusive" }), h("span", { text: "Synthetic" })),
      h("div", { class: "meter-caption", text: `Combined reading ${num(fused)}. The shaded bands show where the decision thresholds, set on separate calibration data, fall.` }));
  }

  function instrumentRow(v) {
    const [name, sub] = INSTRUMENTS[v.tool] || [v.tool, ""];
    const nameCell = h("td", { class: "name" }, name, h("small", { text: sub }));
    if (v.error) {
      const reason = String(v.explanation || "").replace(/^[A-Za-z]+(?:Error|Exception|Exceeded):\s*/, "").replace(/\{[\s\S]*$/, "").trim();
      const friendly = /API_KEY is not set/i.test(reason) ? "No API key is configured for this reviewer." : reason;
      return h("tr", { class: "off" }, nameCell, h("td", { class: "reading", text: "Not used" }), h("td", { text: "-" }), h("td", { class: "remark", text: friendly || "Unavailable for this case." }));
    }
    const reading = h("td", { class: "reading" },
      h("div", { class: "num" }, num(v.score), h("span", { text: " synthetic" })),
      h("div", { class: "bar", role: "img", "aria-label": `Reading ${num(v.score)}` }, h("i", { style: `width:${clamp01(v.score) * 100}%` }), h("b")));
    return h("tr", {}, nameCell, reading, h("td", { class: "num", text: pct(v.confidence) }), h("td", { class: "remark", text: v.explanation }));
  }

  async function loadFigure(imgEl, url) {
    try {
      const r = await fetch(url, { headers: authHeaders() });
      if (!r.ok) throw new Error(String(r.status));
      imgEl.src = URL.createObjectURL(await r.blob());
    } catch (_) { imgEl.alt = "Heat map unavailable"; }
  }

  async function sendDetermination(id, label, out) {
    out.textContent = "Recording...";
    try {
      const r = await fetch(`/jobs/${encodeURIComponent(id)}/correction`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify({ label }) });
      if (!r.ok) throw new Error(String(r.status));
      const j = await r.json();
      out.textContent = j.case_memory_updated ? `Recorded as ${label}. It will be shown as a verified earlier case in future examinations.` : `Recorded as ${label}.`;
    } catch (_) { out.textContent = "The determination could not be recorded."; }
  }

  function renderReport(r, opts) {
    const verdict = r.verdict || "uncertain";
    const caseId = r.id.slice(0, 8).toUpperCase();
    const sections = [];

    sections.push(h("section", {},
      h("div", { class: "case-head" },
        h("div", {}, h("h2", { text: `Case ${caseId}` }),
          h("div", { class: "meta" }, `${r.modality === "text" ? "Text examination" : "Image examination"} - ${fmtDate(r.created)} - reference `, h("code", { text: r.id }))),
        h("button", { class: "btn secondary small no-print", type: "button", onclick: () => window.print() }, "Print record")),
      h("div", { class: "finding", style: "margin-top:18px" },
        h("div", { class: "label", text: "Finding" }),
        h("div", { class: `word ${verdict}`, text: VERDICT_WORD[verdict] }),
        h("div", { class: "sub" }, VERDICT_SUB[r.modality][verdict], typeof r.confidence === "number" ? h("span", { text: ` - confidence ${pct(r.confidence)}` }) : null)),
      meter(r.fused, r.modality),
      r.escalate
        ? h("div", { class: "referral" }, h("strong", { text: "Referred for review" }), h("div", {}, h("p", { text: "A person should confirm this finding before anyone relies on it." }), r.notes ? h("p", { class: "disclaimer", text: r.notes.split(" | ")[0] }) : null))
        : h("div", { class: "cleared", text: "The instruments agree and the combined reading is outside the inconclusive band. No referral was required." })));

    if (r.summary) sections.push(h("section", {}, h("h3", { text: "Summary" }), h("p", { class: "lede", text: r.summary })));

    sections.push(h("section", {},
      h("h3", { text: "Instrument readings" }),
      h("table", { class: "instruments" },
        h("thead", {}, h("tr", {}, h("th", { text: "Instrument" }), h("th", { text: "Reading" }), h("th", { text: "Reliability" }), h("th", { text: "Remarks" }))),
        h("tbody", {}, r.tools.map(instrumentRow))),
      r.reflexion ? h("p", { class: "disclaimer", style: "margin-top:10px", text: "The model reviewer re-examined the material once after the instruments disagreed." }) : null));

    if (r.gradcam) {
      const cam = h("img", { alt: "Regions that most influenced the manipulation network" });
      loadFigure(cam, r.gradcam);
      sections.push(h("section", {}, h("h3", { text: "Figure" }),
        h("div", { class: "figure-pair" },
          opts.sourceUrl ? h("figure", {}, h("img", { src: opts.sourceUrl, alt: "Submitted image" }), h("figcaption", { text: "Fig. 1  Submitted image." })) : null,
          h("figure", {}, cam, h("figcaption", { text: `Fig. ${opts.sourceUrl ? 2 : 1}  Regions that most influenced the manipulation network (warmer is stronger). Indicative only.` })))));
    }

    const notes = (r.notes || "").split(" | ").filter(Boolean);
    if (notes.length) sections.push(h("section", {}, h("h3", { text: "Verifier notes" }), h("ul", { class: "notes" }, notes.map((n) => h("li", { text: n })))));

    sections.push(h("section", {}, h("h3", { text: "Reference material" }),
      r.evidence.length
        ? h("ul", { class: "refs" }, r.evidence.map((e) => h("li", {},
            h("div", { class: "ref-head" }, h("span", { text: EVIDENCE_KIND[e.evidence_type] || e.evidence_type }), h("code", { text: e.source_id }),
              h("span", { text: `similarity ${num(e.similarity_score)}` }), e.verified ? h("span", { class: "chip neutral", text: `verified: ${e.label || "n/a"}` }) : (e.evidence_type === "case_study" ? h("span", { class: "chip neutral", text: "unverified, system-generated" }) : null)),
            h("p", { text: String(e.content_snippet || "").replace(/^#+\s*/, "") }))))
        : h("p", { class: "disclaimer", text: "No supporting reference material was retrieved for this case." })));

    const status = h("p", { class: "done", "aria-live": "polite" });
    const form = h("form", { onsubmit: (ev) => { ev.preventDefault(); const val = new FormData(ev.target).get("label"); if (val) sendDetermination(r.id, val, status); } },
      h("label", {}, h("input", { type: "radio", name: "label", value: "authentic", required: true }), "Authentic"),
      h("label", {}, h("input", { type: "radio", name: "label", value: "synthetic" }), "Synthetic"),
      h("button", { class: "btn small", type: "submit" }, "Record determination"));
    sections.push(h("section", { class: "determination" }, h("h3", { text: "Examiner's determination" }),
      h("p", { class: "hint", text: r.determination ? `Recorded earlier as ${r.determination}. You can change it.` : "If you know the true origin, record it. Verified cases are used to flag contradictions in later examinations." }),
      form, status));

    sections.push(h("section", {}, h("p", { class: "disclaimer", text: "This report is a statistical indication produced by automated instruments. It is not proof of origin, and degraded, compressed or deliberately altered material lowers its reliability." })));

    view.replaceChildren(h("article", { class: "report" }, sections));
    view.firstChild.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------- access key (only shown after a 401) ---------- */
  $("key-save").addEventListener("click", () => {
    try { sessionStorage.setItem(KEY_KEY, $("key-input").value); } catch (_) { /* ignore */ }
    $("key-bar").hidden = true;
  });

  /* ---------- start ---------- */
  showEmpty();
  renderHistory();
  updateText();
  loadHealth().then(() => {
    const requested = new URLSearchParams(location.search).get("case");      // /ui/?case=<job id> opens a stored case
    if (requested && /^[0-9a-f-]{36}$/.test(requested)) openStored(requested);
  });
})();
