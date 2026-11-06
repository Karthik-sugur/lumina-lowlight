"use strict";
/* LUMINA engine integration for the LUMORA v4 interface (replaces its in-browser processing and the
   unconnected localhost FastAPI stubs). Backend: the Hugging Face Space's JSON API "restore" (api.py).
   One request returns the classical, deep-learning and hybrid outputs, the image diagnosis, intermediate
   maps, no-reference quality checks, timings and a recommendation, so switching modes, the Comparison Lab
   and the Technology previews need no further requests. No secrets: the API is public and stateless. */
(() => {
  const API = ((window.LUMINA_CONFIG || {}).apiBase || "").replace(/\/+$/, "");
  const TIMEOUT_MS = 120000;  // covers a cold start of the free Space plus processing
  const SEND_MAX = 2048;      // photos are downsized before upload; the engine processes at <=1280 px
  const KEY = { classical: "classical", deep: "dl", hybrid: "hybrid" };   // UI mode -> engine key
  const UIMODE = { classical: "classical", dl: "deep", hybrid: "hybrid" };
  const NAME = { classical: "Classical CV", dl: "Deep Learning", hybrid: "Hybrid Intelligence" };
  const ORDER = ["classical", "dl", "hybrid"];
  const PRESETS = [["Natural night mood", "Natural night"], ["Auto (balanced)", "Auto"], ["Maximum visibility", "Max visibility"]];
  const SAMPLES = [["01_lolv2_indoor_1", "Meeting room"], ["02_lolv2_stadium_seats", "Stadium seats"],
    ["03_exdark_bus_stop_streetlight", "Bus stop, night"], ["04_exdark_dark_road_lamps", "Road with lamps"],
    ["05_exdark_neon_sign_coloured_lights", "Neon sign"], ["08_exdark_dogs_near_black", "Near-black (hard)"]];
  const E = { online: null, res: null, key: null, imgId: 0, base: {}, preset: "Auto (balanced)", strength: 1,
    busy: false, req: 0, bench: null, tech: 0 };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");   // engine text uses **bold** only
  const fmt = (v, d = 2) => (v === undefined || v === null || Number.isNaN(+v) ? "n/a" : Number(v).toFixed(d));
  const pct = (v) => `${Math.round(v * 100)}%`;
  const neutral = () => !S.p.exposure && !S.p.contrast && !S.p.saturation && !S.p.denoise && !S.p.sharp;
  const current = () => (E.res && E.res._img === E.imgId ? E.res : null);
  const cacheKey = () => `${E.imgId}|${E.preset}|${E.strength}`;
  sync();   // fine-tune sliders start neutral (ui.js), so the default view is the engine's exact output

  /* ---------- engine status (replaces the localhost /api/health check) ---------- */
  const chip = document.createElement("span");
  chip.className = "chip un"; chip.id = "bkst"; chip.style.marginLeft = "auto";
  chip.setAttribute("role", "status"); chip.textContent = "Engine: checking…";
  $("#st").append(chip);
  function paint() {
    const checking = E.online === null;
    chip.className = "chip " + (checking ? "ill" : E.online ? "ok" : "un");
    chip.textContent = checking ? "Engine: connecting…" : E.online ? "Engine online · v3" : "Engine offline";
    chip.title = E.online ? `LUMINA engine: ${API}` : `Can't reach ${API}. The free Hugging Face engine may be waking up; it is retried automatically.`;
    $$(".mode").forEach((m) => {
      const c = m.querySelector(".chip");
      if (c) { c.className = "chip " + (checking ? "ill" : E.online ? "ok" : "un"); c.textContent = checking ? "Connecting…" : E.online ? "Ready" : "Engine offline"; }
    });
  }
  async function health() {
    try {
      const r = await fetch(`${API}/gradio_api/info`, { cache: "no-store", signal: AbortSignal.timeout(90000) });
      E.online = r.ok;
    } catch { E.online = false; }
    paint();
  }

  /* ---------- input: remember the original file, reset results on every new image ---------- */
  let pending = null;
  const _loadFile = loadFile;
  loadFile = function (f) { pending = f; return _loadFile(f); };
  const _load = load;
  load = function (src) {
    S.file = src instanceof HTMLImageElement ? pending : null;
    E.imgId++; E.base = {};
    _load(src);
    $("#an").insertAdjacentHTML("afterbegin", '<p class="note"><b>Quick estimate</b> (in your browser). Press <b>Restore image</b> for the engine’s measurements.</p>');
    setStatus("idle", "Image loaded. Press Auto Enhance, or choose a mode and Restore image.");
  };
  $("#sm").innerHTML = "";
  SAMPLES.forEach(([n, label]) => {
    const b = document.createElement("button");
    b.innerHTML = `<img src="samples/${n}.jpg" alt="" style="filter:brightness(1.7)">${esc(label)}`;
    b.onclick = async () => {
      try {
        const blob = await (await fetch(`samples/${n}.jpg`)).blob();
        loadFile(new File([blob], `${n}.jpg`, { type: "image/jpeg" }));
      } catch { setStatus("error", "Couldn’t load that sample photo."); }
    };
    $("#sm").append(b);
  });

  async function toDataUrl() {
    const src = S.file ? await createImageBitmap(S.file, { imageOrientation: "from-image" }) : null;
    const w0 = src ? src.width : S.img.width, h0 = src ? src.height : S.img.height;
    const k = Math.min(1, SEND_MAX / Math.max(w0, h0)), c = document.createElement("canvas");
    c.width = Math.round(w0 * k); c.height = Math.round(h0 * k);
    E.sentNote = k < 1 ? `Your photo is ${w0}×${h0} px; it was reduced to ${c.width}×${c.height} px before upload (the free server’s size limit), so results are ${c.width}×${c.height} px.` : "";
    // no resize needed: send the file's own bytes, so re-encoding adds no JPEG blocks before the engine measures them
    if (k === 1 && S.file && /^image\/(jpeg|png|webp)$/.test(S.file.type)) {
      return await new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = () => bad(new Error("Couldn’t read the file.")); r.readAsDataURL(S.file); });
    }
    const x = c.getContext("2d");
    if (src) x.drawImage(src, 0, 0, c.width, c.height);
    else { const t = document.createElement("canvas"); put(t, S.img); x.drawImage(t, 0, 0, c.width, c.height); }
    return c.toDataURL("image/jpeg", 0.95);
  }

  /* ---------- API: POST /gradio_api/call/restore, then read the server-sent event stream ---------- */
  function parseSSE(text) {
    let last = {};
    for (const block of text.split(/\r?\n\r?\n/)) {
      const ev = (block.match(/^event:\s*(.*)$/m) || [])[1];
      const data = (block.match(/^data:\s*([\s\S]*)$/m) || [])[1];
      if (ev === "complete" || ev === "error") last = { event: ev, data };
    }
    return last;
  }
  async function callRestore(dataUrl, preset, strength) {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      const r1 = await fetch(`${API}/gradio_api/call/restore`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
        body: JSON.stringify({ data: [dataUrl, preset, strength] }) });
      if (r1.status === 413) throw new Error("That image is too large for the engine. Try a smaller photo.");
      if (!r1.ok) throw new Error(`The engine answered HTTP ${r1.status}. It may be starting up — please retry in a minute.`);
      const { event_id } = await r1.json();
      const ev = parseSSE(await (await fetch(`${API}/gradio_api/call/restore/${event_id}`, { signal: ctl.signal })).text());
      if (ev.event === "complete") return JSON.parse(ev.data)[0];
      if (ev.event === "error") {
        let msg = null; try { msg = JSON.parse(ev.data)?.error; } catch { /* non-JSON payload */ }
        throw new Error(msg || "Processing failed on the engine. Please try another photo.");
      }
      throw new Error("The engine sent an unexpected response. Please retry.");
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The engine took too long. If it was asleep it should be awake now — please press Restore again.");
      if (e instanceof TypeError) throw new Error(`Couldn’t reach the LUMINA engine (${API}). Check your connection, or retry in a minute while it starts up.`);
      throw e;
    } finally { clearTimeout(timer); }
  }

  /* ---------- studio ---------- */
  async function baseImage(k) {   // decoded engine output (ImageData), cached per result
    if (!E.base[k]) {
      // onload, not img.decode(): decode() can stay pending in background/hidden tabs, which froze the UI
      const img = await new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => reject(new Error("Couldn’t display the engine’s result. Please press Restore again."));
        im.src = E.res.results[k].image;
      });
      const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
      const x = c.getContext("2d"); x.drawImage(img, 0, 0);
      E.base[k] = x.getImageData(0, 0, c.width, c.height);
    }
    return E.base[k];
  }
  function pickMode(apiKey) {
    S.mode = UIMODE[apiKey];
    $$("[name=md]").forEach((r) => (r.checked = r.value === S.mode));
  }
  async function show(first = false) {
    const res = current(); if (!res) return;
    const k = KEY[S.mode], r = res.results[k], rec = res.recommendation;
    let im = await baseImage(k);
    if (!neutral()) im = enhance(im, S.p);
    put($("#cR"), im);
    $("#cR").hidden = $("#hd").hidden = $("#chr").hidden = false;
    $("#chr").textContent = NAME[k] + (k === rec.method ? " · Recommended" : "");
    if (first || !$("#hd").style.left) mv({ clientX: $("#zw").getBoundingClientRect().left + $("#zw").offsetWidth / 2 });
    $("#dl").disabled = false;
    const stale = E.key !== cacheKey() ? " · settings changed, press Restore image to apply" : "";
    setStatus("done", `${NAME[k]}: ${r.ms} ms on the engine · all three pipelines ${res.total_ms} ms · ${Math.round(res.wall_ms)} ms incl. upload`
      + (neutral() ? "" : " · fine-tuned in your browser") + (res.input.note ? ` · ${res.input.note}` : "") + stale);
  }
  async function restore(auto = false) {
    if (!S.img) return setStatus("error", "Add an image first.");
    if (E.busy) return;
    if (current() && E.key === cacheKey()) { if (auto) pickMode(E.res.recommendation.method); return show(); }
    const id = ++E.req, key = cacheKey(), img = E.imgId;
    E.busy = true; $("#go").disabled = $("#au").disabled = true;
    setStatus("busy", "Restoring on the LUMINA engine: classical, deep learning and hybrid…");
    const slow = setTimeout(() => id === E.req && setStatus("busy", "Still working — the free engine may be waking up (up to a minute)…"), 8000);
    const t0 = performance.now();
    try {
      const res = await callRestore(await toDataUrl(), E.preset, E.strength);
      if (id !== E.req || img !== E.imgId) return;          // superseded, or a new image was loaded
      res.wall_ms = performance.now() - t0; res._img = img;
      if (E.sentNote) res.input.note = `${E.sentNote} ${res.input.note || ""}`.trim();
      E.res = res; E.key = key; E.base = {}; E.online = true; paint();
      if (auto) pickMode(res.recommendation.method);
      await show(true);
      renderAnalysis();
    } catch (e) {
      if (id === E.req) setStatus("error", e.message);
    } finally {
      clearTimeout(slow);
      if (id === E.req) { E.busy = false; $("#go").disabled = $("#au").disabled = false; }
    }
  }
  $("#go").onclick = () => restore(false);
  $("#au").onclick = () => {
    if (!S.img) return setStatus("error", "Add an image first.");
    Object.assign(S.p, { exposure: 0, contrast: 0, saturation: 0, denoise: 0, sharp: 0, gamma: 1 }); sync();
    setPreset("Auto (balanced)"); setStrength(1);
    restore(true);
  };
  $$("[name=md]").forEach((r) => r.addEventListener("change", () => current() && show()));
  let tune = 0;   // debounce live fine-tuning (a timer, unlike requestAnimationFrame, also runs in background tabs)
  $$("[data-k]").forEach((i) => i.addEventListener("input", () => {
    if (!current()) return;
    clearTimeout(tune); tune = setTimeout(() => show(), 40);
  }));
  $("#dl").onclick = () => $("#cR").toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `lumina-${KEY[S.mode]}${neutral() ? "" : "-finetuned"}.png`;
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1e3);
  }, "image/png");

  /* engine settings (real controls) above the teammate's fine-tune sliders */
  const eng = document.createElement("details");
  eng.className = "dt"; eng.open = true;
  eng.innerHTML = `<summary>Engine settings</summary>
    <div class="sl"><label>Intent</label><div class="row2" id="eintent" role="group" aria-label="Intent" style="flex-wrap:wrap;margin-top:6px">
      ${PRESETS.map(([v, l]) => `<button class="pill sm" data-v="${esc(v)}" aria-pressed="false">${l}</button>`).join("")}</div></div>
    <div class="sl"><label>Brightness<span id="v_eb">1.00×</span></label><input type="range" id="eb" min="0.6" max="1.4" step="0.05" value="1"></div>
    <p class="note">These change how the engine restores the photo; press <b>Restore image</b> to apply.</p>`;
  $("#sls").prepend(eng);
  $$("#sls details summary").forEach((s) => { if (s.textContent !== "Engine settings") s.textContent = "Fine-tune: " + s.textContent.toLowerCase(); });
  $("#sls").insertAdjacentHTML("beforeend", '<p class="note">Fine-tuning runs in your browser on top of the engine’s result. All at 0 shows the engine’s output exactly.</p>');
  function markStale() { if (current()) setStatus("idle", "Settings changed — press Restore image to re-run on the engine."); }
  function setPreset(v) {
    E.preset = v;
    $$("#eintent button").forEach((b) => { const on = b.dataset.v === v; b.classList.toggle("dk", on); b.setAttribute("aria-pressed", String(on)); });
  }
  function setStrength(v) { E.strength = v; $("#eb").value = v; $("#v_eb").textContent = `${(+v).toFixed(2)}×`; }
  $$("#eintent button").forEach((b) => (b.onclick = () => { setPreset(b.dataset.v); markStale(); }));
  $("#eb").oninput = (e) => { setStrength(+e.target.value); markStale(); };
  setPreset(E.preset);

  /* analysis tab: the engine's measurements (the quick in-browser estimate stays available) */
  const row = (label, val, frac, tip) => `<div class="brow"><div><span data-tip="${esc(tip)}" tabindex="0">${label}</span><b>${val}</b></div><i><b style="width:${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%"></b></i></div>`;
  const DIAG = [["illumination", "Illumination"], ["noise", "Noise"], ["sharpness", "Sharpness"], ["colour", "Colour quality"],
    ["compression", "Compression"], ["resolution", "Resolution"], ["recoverability", "Recoverability"]];
  function diagnosticsHTML(r) {   // every label comes with the measured value behind it
    if (!r.diagnostics) return "";
    const item = (k, v, sub) => `<div style="padding:7px 0;border-bottom:1px solid var(--line)"><div style="display:flex;justify-content:space-between;gap:10px;font-size:13px"><span>${k}</span><b style="font-weight:500;text-align:right">${v}</b></div>${sub ? `<div class="note" style="font-size:12px;margin-top:2px">${sub}</div>` : ""}</div>`;
    return `<h4 style="font-weight:500;margin:4px 0 4px">Image quality diagnostics</h4>`
      + DIAG.filter(([k]) => r.diagnostics[k]).map(([k, n]) => item(n, esc(r.diagnostics[k].label), esc(r.diagnostics[k].value))).join("")
      + `<details class="dt" open style="margin-top:12px"><summary>Enhancement breakdown</summary><div style="margin:6px 0 2px">${r.strategy.cases.map((c) => `<span class="chip ill" style="margin:0 4px 4px 0">${esc(c)}</span>`).join("")}</div>`
      + r.strategy.operations.map((o) => item(esc(o.stage), esc(o.decision), esc(o.why))).join("") + `</details>`;
  }
  function analysisHTML(r) {
    const d = r.diagnosis, rec = r.recoverability, lv = { High: "ok", Medium: "ill", Low: "un" }[rec.level] || "un";
    return `<p class="note"><b>Measured by the LUMINA engine</b><span class="chip ${lv}">Recoverability: ${esc(rec.level)}</span><br>${esc(rec.why)}</p>`
      + diagnosticsHTML(r) + `<h4 style="font-weight:500;margin:14px 0 8px">Measurements</h4>`
      + row("Brightness", pct(d.brightness), d.brightness, "Mean luminance of the input (bar: 0–100%).")
      + row("Near-black pixels", pct(d.dark_fraction), d.dark_fraction, "Share of pixels with luminance below 10%.")
      + row("Sensor noise σ (as captured)", fmt(d.noise_sigma, 1), d.noise_sigma / 10, "Immerkær estimate on the input, in /255 units (bar: 0–10). 8-bit quantisation hides noise in dark images.")
      + row("Grain after brightening σ", fmt(r.grain_uncovered, 1), r.grain_uncovered / 40, "Noise the brightening uncovers, before restoration (90th percentile, /255; bar: 0–40).")
      + row("Lighting spread", `${fmt(d.lighting_spread, 1)} stops`, d.lighting_spread / 8, "Difference between bright (p90) and dark (p10) illumination, in photographic stops (bar: 0–8).")
      + row("Colour cast", `${fmt(d.colour_cast, 0)} (${esc(d.cast_hue)})`, d.colour_cast / 30, "Chroma of the average mid-tone colour in Lab; 0 = neutral (bar: 0–30).")
      + row("Contrast", fmt(d.contrast, 2), d.contrast / 2, "RMS contrast relative to mean brightness (bar: 0–2).")
      + `<p class="note">Coloured light sources: <b>${d.coloured_lights ? "yes, white balance backed off" : "none detected"}</b></p>`;
  }
  function renderAnalysis() {
    const r = current(); if (!r) return;
    $("#an").innerHTML = analysisHTML(r)
      + `<details class="dt" open><summary>What we noticed</summary><ul class="note">${r.noticed.map((s) => `<li>${md(s)}</li>`).join("")}</ul></details>`
      + `<details class="dt"><summary>What the engine did</summary><ul class="note">${r.actions.map((s) => `<li>${md(s)}</li>`).join("")}</ul></details>`
      + `<details class="dt"><summary>Quick estimate (in your browser)</summary>${bars(S.a)}</details>`;
  }

  /* ---------- Comparison Lab ---------- */
  const dlCanvas = (c, name) => c.toBlob((b) => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1e3);
  }, "image/png");
  renderCompare = function () {
    const el = $("#cmp");
    if (!S.img) { el.innerHTML = empty("Load an image in the Studio to compare approaches."); return; }
    const res = current(), ar = `style="aspect-ratio:${S.img.width}/${S.img.height}"`;
    const q = (x) => `<ul><li>Brightness<b>${pct(x.quality.brightness)}</b></li><li>Grain σ<b>${fmt(x.quality.noise, 1)}</b></li><li>Blown-out<b>${fmt(x.quality.clipped_pct, 2)}%</b></li><li>Colour cast<b>${fmt(x.quality.colour_cast, 1)}</b></li>${x.quality.sharpness !== undefined ? `<li>Detail<b>${fmt(x.quality.sharpness, 3)}</b></li><li>Colourfulness<b>${fmt(x.quality.colourfulness, 2)}</b></li><li>Blockiness<b>${fmt(x.quality.blockiness, 2)}</b></li>` : ""}${x.ms !== undefined ? `<li>Time<b>${x.ms} ms</b></li>` : ""}</ul>`;
    const pane = (k) => !res
      ? `<figure class="pane${k === "hybrid" ? " hy" : ""}"><div class="pv nap" ${ar}><div class="na"><div>Not run yet<br><small>Press “Run all methods on the engine”</small></div></div></div><h4>${NAME[k]}</h4><ul><li>Metrics<b>—</b></li><li>Time<b>—</b></li></ul></figure>`
      : `<figure class="pane${k === "hybrid" ? " hy" : ""}"><div class="pv" ${ar}><canvas data-c="${k}"></canvas></div><h4>${NAME[k]}${res.recommendation.method === k ? '<span class="chip ok">Recommended</span>' : ""}</h4>${q(res.results[k])}<button class="pill sm" data-dl="${k}" style="margin:4px 6px 6px">Download</button></figure>`;
    el.innerHTML = `<div class="row2" style="margin:0 0 14px;align-items:center;flex-wrap:wrap"><button class="pill sm dk" id="runcmp" ${E.busy ? "disabled" : ""}>${res ? "Re-run all methods" : "Run all methods on the engine"}</button><span class="note" id="cmpst">${res ? `Same input through all three pipelines · engine time ${res.total_ms} ms · enhanced at ${res.input.processed_width}×${res.input.processed_height} px · output ${res.input.output_width || res.input.processed_width}×${res.input.output_height || res.input.processed_height} px` : "Classical, Deep Learning and Hybrid run on the LUMINA engine in one request."}</span></div>
      <div class="cg"><figure class="pane"><div class="pv" ${ar}><canvas id="k0"></canvas></div><h4>Original</h4>${res ? q(res.input) : ""}<button class="pill sm" data-dl="original" style="margin:4px 6px 6px">Download</button></figure>${ORDER.map(pane).join("")}</div>
      ${res ? `<div class="card" style="margin-top:16px"><h4 style="font-weight:500">Recommendation: ${esc(NAME[res.recommendation.method])}</h4><p class="note">${esc(res.recommendation.why)}</p>
        <p class="note">Penalty, lower is better: ${ORDER.map((k) => `${NAME[k]} <b>${fmt(res.recommendation.scores[k], 2)}</b>`).join(" · ")}. Score = ${Object.entries(res.recommendation.weights).map(([k, w]) => `${w}×${k}`).join(" + ")}${res.recommendation.margin ? `; another method replaces Hybrid only if its score is lower by more than ${res.recommendation.margin}` : ""}, where exposure = |brightness − ${pct(res.recommendation.target_brightness)} target| / 0.2, noise = σ / 5, clipping = blown-out % / 1, colour = cast / 15 + oversaturation above 0.55 / 0.2, detail = shortfall vs the sharpest output, artefacts = blockiness above 1.1 / 0.4.</p>
        <p class="note">These are no-reference checks: your photo has no ground-truth reference, so PSNR, SSIM and LPIPS can’t be computed for it. They are measured on reference datasets in Benchmarks.</p></div>`
      : '<p class="note">Metrics appear after the run. They are no-reference checks; PSNR and SSIM need a ground-truth reference (see Benchmarks).</p>'}`;
    put($("#k0"), S.img);
    if (res) ORDER.forEach(async (k) => put(el.querySelector(`canvas[data-c="${k}"]`), await baseImage(k)));
    $("#runcmp").onclick = async () => {
      $("#runcmp").disabled = true; $("#cmpst").textContent = "Running all three pipelines on the engine…";
      await restore(false);
      if (!current()) $("#cmpst").textContent = $("#stx").textContent;   // show the error
      else renderCompare();
      $("#runcmp").disabled = false;
    };
    el.querySelectorAll("[data-dl]").forEach((b) => (b.onclick = () => {
      const k = b.dataset.dl, c = k === "original" ? $("#k0") : el.querySelector(`canvas[data-c="${k}"]`);
      dlCanvas(c, `lumina-${k === "original" ? "original" : k}.png`);
    }));
  };

  /* ---------- Technology: the seven stages with this photo's real intermediate outputs ---------- */
  const STAGES = [
    ["Image analysis", "Measures brightness, noise, sharpness, colour, compression and lighting so every later stage adapts to this photo.",
      "Luminance statistics; Immerkær noise estimate (σ in /255 units); lighting spread = p90 − p10 of log₂ illumination (stops); colour cast = chroma of the mean mid-tone colour in Lab; colourfulness = chroma ÷ lightness; sharpness = exposure-normalised Laplacian energy; JPEG blockiness = 8-px grid step energy. These set the brightness target, gain limit, pre-denoising, CLAHE, white balance, the classical path's saturation ceiling and detail recovery.",
      "RGB image → diagnosis → adaptive parameters", "analysis"],
    ["Illumination estimation", "Finds how much light reached each part of the scene, so dark regions can be lifted without blowing out bright ones.",
      "T = max(R, G, B), refined with an edge-aware guided filter in √-space (radius 1.5% of the image size), so the map is smooth inside regions but follows object edges, which prevents halos.",
      "Image → illumination map T", [["viz", "Illumination map T"]]],
    ["Retinex decomposition", "Splits the photo into the light and the scene’s own colours and texture, then re-lights it.",
      "I = R × T. The Retinex candidate is I · T^(γ−1) with the gain capped at 25×; γ is solved per image by bisection so the mean brightness hits the adaptive target.",
      "Image + T → reflectance → Retinex candidate", [["viz", "Reflectance R = I / T"], ["viz", "Retinex gain T^(gamma-1)"]]],
    ["Classical enhancement", "The complete classical branch: Retinex re-lighting, adaptive CLAHE, edge-preserving denoising and colour correction.",
      "Two bilateral passes on luma plus a luma-guided filter on chroma, with strength from the noise map; CLAHE on Lab lightness; partial gray-world white balance that backs off for coloured lights; soft highlight roll-off.",
      "Image → Classical CV result", [["viz", "Classical candidate (Retinex)"], ["res", "classical"]]],
    ["Deep-learning restoration", "A fine-tuned Zero-DCE++ network estimates per-pixel tone curves; a quality check picks how strongly to apply them.",
      "10.6k-parameter curve network fine-tuned on LOL (seeded 435/50 train/validation split). Curves are pixel-wise and monotone, so the network re-tones pixels but can’t invent content. Deep Learning mode shows this network’s output on its own.",
      "Image → curve map → Deep Learning result", [["viz", "Curve strength |A|"], ["res", "dl"]]],
    ["Adaptive hybrid fusion", "Blends both candidates pixel by pixel, a restoration U-Net removes the noise brightening uncovered, then detail is recovered only where it is real.",
      "Fusion weight = well-exposedness × detail ÷ amplification, blended with a 5-level Laplacian pyramid. The 145k-parameter restoration U-Net (v3) is conditioned on a noise-level map derived from the illumination model. Detail recovery runs after denoising: removed detail stronger than the local noise is re-injected, then two-scale guided unsharp masking (cored at the residual noise, halo-clamped) with strengths that drop for noisy, already-sharp or blocky photos. Large photos are enhanced at ≤1280 px and returned at full size by transferring the photo’s own above-noise fine detail.",
      "Both candidates + noise map → restoration → detail recovery → Hybrid result", [["viz", "DL candidate (Zero-DCE++)"], ["viz", "Fusion weight (DL share)"], ["viz", "Noise-level map (to DL)"], ["viz", "Removed noise / corrections (x4)"], ["viz", "Recovered detail (x4)"], ["res", "hybrid"]]],
    ["Quality validation", "Checks every output with no-reference measures and recommends one, transparently, for this photo.",
      "Brightness vs target, residual grain (Immerkær σ), blown-out pixels, colour (cast + oversaturation), detail and JPEG artefacts form a weighted penalty. PSNR, SSIM and LPIPS need a ground-truth photo, so they are reported on benchmark datasets (Benchmarks page).",
      "Results → checks → recommendation", "quality"],
  ];
  const fig = (src, title, cap) => `<figure style="margin:0 0 12px"><img src="${src}" alt="${esc(title)}" style="width:100%;height:auto;border-radius:16px;display:block;margin-bottom:6px"><p class="note"><b>${esc(title)}</b>${cap ? ` — ${esc(cap)}` : ""}</p></figure>`;
  renderTech = function () {
    const T = E.tech;
    $("#ts").innerHTML = STAGES.map((s, i) => `<button class="${i === T ? "on" : i < T ? "past" : ""}" data-i="${i}" aria-current="${i === T}"><b>${i + 1}</b>${s[0]}</button>`).join('<i class="cx" aria-hidden="true"></i>');
    $$("#ts button").forEach((b) => (b.onclick = () => { E.tech = +b.dataset.i; renderTech(); }));
    const s = STAGES[T];
    $("#td").innerHTML = `<span class="tag">Stage ${T + 1} of 7</span><h2 style="font-weight:300;margin:10px 0">${s[0]}</h2><p class="note" style="font-size:14px">${s[1]}</p><details><summary>Technical detail</summary><p class="note">${s[2]}</p></details><p class="note"><b>Flow:</b> ${s[3]}</p><div class="row2"><button class="pill sm" id="pv" ${T ? "" : "disabled"}>← Previous</button><button class="pill sm dk" id="nx" ${T < 6 ? "" : "disabled"}>Next →</button></div>`;
    $("#pv").onclick = () => { E.tech--; renderTech(); };
    $("#nx").onclick = () => { E.tech++; renderTech(); };
    const p = $("#tp"), res = current();
    if (!S.img) { p.innerHTML = empty("Load an image in the Studio to see a real output for this stage."); return; }
    if (!res) {
      p.innerHTML = `<div class="empty"><p>Run the engine once to see this stage’s real output for your photo.</p><button class="pill sm dk" id="trun" ${E.busy ? "disabled" : ""}>Run on the engine</button></div>`;
      $("#trun").onclick = async () => { $("#trun").disabled = true; $("#trun").textContent = "Running…"; await restore(false); renderTech(); };
      return;
    }
    if (s[4] === "analysis") { p.innerHTML = "<h4>Computed from your image by the engine</h4><br>" + analysisHTML(res); return; }
    if (s[4] === "quality") {
      p.innerHTML = `<h4>Checks on your photo’s three outputs</h4><table><tr><th>Method</th><th>Brightness</th><th>Grain σ</th><th>Blown-out</th><th>Detail</th><th>Penalty</th></tr>${ORDER.map((k) => { const x = res.results[k]; return `<tr><td>${NAME[k]}${res.recommendation.method === k ? '<span class="chip ok">Best</span>' : ""}</td><td>${pct(x.quality.brightness)}</td><td>${fmt(x.quality.noise, 1)}</td><td>${fmt(x.quality.clipped_pct, 2)}%</td><td>${fmt(x.quality.sharpness, 3)}</td><td>${fmt(res.recommendation.scores[k], 2)}</td></tr>`; }).join("")}</table><p class="note">${esc(res.recommendation.why)}</p>`;
      return;
    }
    const byName = Object.fromEntries(res.intermediates.map((i) => [i.name, i]));
    p.innerHTML = s[4].map(([type, k]) => type === "res"
      ? fig(res.results[k].image, `${NAME[k]} output`, `${res.results[k].ms} ms on the engine`)
      : (byName[k] ? fig(byName[k].image, k, byName[k].caption) : "")).join("")
      + `<p class="note">Real outputs computed by the engine for your photo. Heatmaps use the colour-blind-safe cividis scale (dark = low, yellow = high).</p>`;
  };

  /* ---------- Benchmarks: the evaluation's own numbers (results/metrics.json → benchmarks.json) ---------- */
  const BT = {
    PSNR: "Peak signal-to-noise ratio. Higher is better. Needs a ground-truth photo.",
    SSIM: "Structural similarity to a reference, 0 to 1. Higher is better.",
    LPIPS: "Learned perceptual distance to a reference. Lower is better.",
    NIQE: "Naturalness score that needs no reference. Lower is better.",
    "Processing latency": "Time to restore one image.",
    "Noise reduction": "Residual noise left in the output (Immerkær σ, /255). Lower is better.",
    "Edge preservation": "Correlation of edges with the reference (EPI, 0 to 1). Higher is better.",
  };
  const PAIRED = ["LOL-v1 eval15", "LOL-v2 real test (clean subset)", "LOL-v2 synthetic test"];
  const COLORS = { classical: "#ff9e72", dl: "#c93d7a", hybrid: "#6d3fd8" };
  renderBench = async function () {
    const el = $("#bn");
    try { if (!E.bench) E.bench = await (await fetch("benchmarks.json")).json(); }
    catch { el.innerHTML = '<div class="empty"><p>Couldn’t load the benchmark data.</p></div>'; return; }
    const b = E.bench, avg = (m, k) => PAIRED.reduce((s, set) => s + b.main[set][m][k], 0) / PAIRED.length;
    const names = Object.fromEntries(b.methods.map((m) => [m.key, m.name]));
    const lat = b.latency_ms[names.hybrid], res = current(), NP = PAIRED.reduce((s, set) => s + b.n_images[set], 0);
    const cards = [
      ["PSNR", "Needs ground truth", `${fmt(avg("hybrid", "psnr"), 2)} dB`, `Hybrid, average of 3 unseen paired test sets (${NP} pairs, sets weighted equally).`],
      ["SSIM", "Needs ground truth", fmt(avg("hybrid", "ssim"), 3), `Hybrid, same ${NP} pairs.`],
      ["LPIPS", "Needs ground truth", fmt(avg("hybrid", "lpips"), 3), `Hybrid, same ${NP} pairs.`],
      ["NIQE", "No-reference", fmt(avg("hybrid", "niqe"), 2), `Hybrid on the paired sets; ${fmt(b.main["ExDark (real-world, no reference)"].hybrid.niqe, 2)} on ${b.n_images["ExDark (real-world, no reference)"]} real-world ExDark photos.`],
      ["Processing latency", "Runtime", `${fmt(lat.ms_600x400, 0)} ms`, `Hybrid, 600×400, laptop CPU (${fmt(lat.ms_1280x960 / 1000, 1)} s at 1280×960${lat.ms_2048x1536_full_size ? `, ${fmt(lat.ms_2048x1536_full_size / 1000, 1)} s for a 2048 px photo returned full size` : ""}).${res ? ` This session on the engine: ${res.results.hybrid.ms} ms.` : ""}`],
      ["Noise reduction", "Derived", fmt(avg("hybrid", "noise"), 2), `Residual σ (hybrid); deep learning only ${fmt(avg("dl", "noise"), 2)}, classical only ${fmt(avg("classical", "noise"), 2)}.`],
      ["Edge preservation", "Needs ground truth", fmt(avg("hybrid", "epi"), 3), `EPI (hybrid), same ${NP} pairs.`],
    ];
    const ps = ["classical", "dl", "hybrid"].map((k) => avg(k, "psnr")), top = Math.max(...ps) * 1.15;
    const bar = (k, i) => { const v = ps[i], h = (v / top) * 70, x = 62 + i * 75; return `<rect x="${x}" y="${80 - h}" width="26" height="${h}" rx="6" fill="${COLORS[k]}"/><text x="${x + 13}" y="${74 - h}" text-anchor="middle" font-size="10" fill="#2b1a2e">${v.toFixed(2)}</text>`; };
    const ab = b.ablation, sets = Object.keys(ab).filter((s) => ab[s]["Hybrid (full)"].psnr !== undefined), variants = Object.keys(ab[sets[0]]);
    const BEFORE = "previous pipeline (before quality-aware update)", baSets = Object.keys(ab).filter((s) => ab[s][BEFORE]);
    const baCell = (v, k, d) => (v[k] === undefined ? "—" : fmt(v[k], d));
    const pairedTable = (set) => `<table><tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>NIQE ↓</th><th>ms</th></tr>${b.methods.map((m) => { const r = b.main[set][m.key]; return `<tr><td>${esc(m.name)}</td><td>${fmt(r.psnr, 2)}</td><td>${fmt(r.ssim, 3)}</td><td>${fmt(r.lpips, 3)}</td><td>${fmt(r.niqe, 2)}</td><td>${fmt(b.latency_ms[m.name]?.ms_600x400, 0)}</td></tr>`; }).join("")}</table>`;
    const ex = "ExDark (real-world, no reference)", conds = Object.keys(b.robustness);
    el.innerHTML = `<p class="lg"><b>Data status</b><span><span class="chip ok">Actual</span> measured on unseen test images (results/metrics.json)</span><span><span class="chip un">Unavailable</span> metric needs a reference that doesn't exist</span></p>
      <div class="bg">${cards.map(([n, t, v, note]) => `<div class="card lift"><div class="mrow"><span class="tag">${t}</span><span class="chip ok">Actual</span></div><h4 style="margin-top:10px;font-weight:500" data-tip="${esc(BT[n])}" tabindex="0">${n}</h4><div class="v">${v}</div><p class="note">${esc(note)}</p></div>`).join("")}</div>
      <div class="g3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
        <div class="card"><h4 style="font-weight:500">Classical vs Deep vs Hybrid</h4><svg viewBox="0 0 300 100" width="100%" role="img" aria-label="Average PSNR: ${["classical", "dl", "hybrid"].map((k, i) => `${names[k]} ${ps[i].toFixed(2)} dB`).join(", ")}"><path d="M30 80H270" stroke="#f0e4ee"/>${["classical", "dl", "hybrid"].map(bar).join("")}<text x="150" y="96" text-anchor="middle" font-size="9" fill="#7b6670">Average PSNR (dB), ${NP} unseen pairs</text></svg>
          <p class="lg"><span><i style="background:#ff9e72"></i>Classical CV</span><span><i style="background:#c93d7a"></i>Deep Learning</span><span><i style="background:#6d3fd8"></i>Hybrid</span></p>
          <p class="note">Hybrid also leads on SSIM (${fmt(avg("hybrid", "ssim"), 3)} vs ${fmt(avg("dl", "ssim"), 3)} / ${fmt(avg("classical", "ssim"), 3)}). It is <b>not</b> best on every set: on LOL-v2 synthetic, deep learning alone wins.</p></div>
        <div class="card"><h4 style="font-weight:500">Ablation study</h4><table><tr><th>Configuration</th>${sets.map((s) => `<th>${esc(s.replace(" test", ""))} PSNR</th><th>SSIM</th>`).join("")}</tr>${variants.map((v) => `<tr><td>${esc(v)}</td>${sets.map((s) => `<td>${fmt(ab[s][v].psnr, 2)}</td><td>${fmt(ab[s][v].ssim, 3)}</td>`).join("")}</tr>`).join("")}</table><p class="note">Each row removes one component from the hybrid (the last row is the whole pipeline before the image-quality-aware update).</p></div>
      </div>
      ${baSets.length ? `<div class="g3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-top:16px">
        <div class="card"><h4 style="font-weight:500">Image-quality-aware update: before → after</h4><table><tr><th>Test set</th><th></th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>NIQE</th><th>Blockiness</th></tr>${baSets.map((s) => [["before", ab[s][BEFORE]], ["after", ab[s]["Hybrid (full)"]]].map(([l, v]) => `<tr><td>${l === "before" ? esc(s.replace(" test", "").replace(" (real-world, no reference)", "")) : ""}</td><td>${l}</td><td>${baCell(v, "psnr", 2)}</td><td>${baCell(v, "ssim", 3)}</td><td>${baCell(v, "lpips", 3)}</td><td>${baCell(v, "niqe", 2)}</td><td>${baCell(v, "blockiness", 2)}</td></tr>`).join("")).join("")}</table><p class="note">Hybrid on unseen test sets. ExDark has no reference, so only no-reference checks apply there.</p></div>
        ${b.auto && b.auto[PAIRED[0]] ? `<div class="card"><h4 style="font-weight:500">Auto mode: method chosen per photo</h4><table><tr><th>Test set</th><th>Picks C / DL / H</th><th>Auto PSNR</th><th>Always hybrid</th><th>Oracle</th></tr>${PAIRED.map((s) => { const a = b.auto[s]; return `<tr><td>${esc(s.replace(" test", ""))}</td><td>${["classical", "dl", "hybrid"].map((k) => a.picks[k] || 0).join(" / ")}</td><td>${fmt(a.auto.psnr, 2)}</td><td>${fmt(b.main[s].hybrid.psnr, 2)}</td><td>${fmt(a.oracle_best_psnr.psnr, 2)}</td></tr>`; }).join("")}</table><p class="note">Auto uses only no-reference checks (no ground truth); the oracle picks the best PSNR per photo and is shown only as headroom.</p></div>` : ""}
      </div>` : ""}
      <details class="dt" style="margin-top:16px"><summary>Every test set, method and condition</summary>
        ${PAIRED.map((s) => `<h4 style="font-weight:500;margin:14px 0 6px">${esc(s)} <span class="note">· ${b.n_images[s]} pairs · paired metrics</span></h4>${pairedTable(s)}`).join("")}
        <h4 style="font-weight:500;margin:14px 0 6px">${esc(ex)} <span class="note">· ${b.n_images[ex]} photos · no reference</span></h4><table><tr><th>Method</th><th>NIQE ↓</th><th>Residual σ ↓</th><th>Clipped %</th><th>PSNR / SSIM</th></tr>${b.methods.map((m) => { const r = b.main[ex][m.key]; return `<tr><td>${esc(m.name)}</td><td>${fmt(r.niqe, 2)}</td><td>${fmt(r.noise, 2)}</td><td>${fmt(r.overexposed, 2)}</td><td><span class="chip un">Unavailable</span></td></tr>`; }).join("")}</table>
        <h4 style="font-weight:500;margin:14px 0 6px">Robustness to darker and uneven lighting <span class="note">· PSNR dB, 30 unseen references</span></h4><div style="overflow-x:auto"><table><tr><th>Method</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${Object.keys(b.robustness[conds[0]]).map((k) => `<tr><td>${esc(names[k])}</td>${conds.map((c) => `<td>${fmt(b.robustness[c][k].psnr, 2)}</td>`).join("")}</tr>`).join("")}</table></div>
      </details>
      <p class="note">Full evaluation, model selection, failure cases and limitations: <a href="https://github.com/darshchandran/lumina-lowlight/blob/main/eval_P1_T76.md" target="_blank" rel="noopener">eval_P1_T76.md</a>. Latency measured on ${esc(b.machine)}.</p>`;
  };

  paint();
  health();
  setInterval(health, 30000);
  addEventListener("focus", health);
  route();
})();
