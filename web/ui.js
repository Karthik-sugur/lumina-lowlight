/* LUMINA UI: the teammate's LUMORA v4 interface code (routing, studio viewer, sliders, zoom,
   tooltips, accessibility). Processing, samples and the Compare/Technology/Benchmarks renderers
   were replaced by real engine integration in lumina.js; see docs/FRONTEND.md. */
const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
const pages = [
    "home",
    "studio",
    "technology",
    "compare",
    "benchmarks",
    "about",
  ],
  shell = $(".shell");
const S = {
  mode: "classical",
  img: null,
  res: null,
  t: null,
  a: null,
  z: 1,
  /* fine-tune adjustments applied in the browser on top of the engine result; neutral by default */
  p: { exposure: 0, contrast: 0, saturation: 0, denoise: 0, sharp: 0, gamma: 1 },
};
function route() {
  const h = location.hash.slice(1),
    p = pages.includes(h) ? h : "home";
  $$("main>section").forEach((s) =>
    s.classList.toggle("on", s.dataset.page === p),
  );
  $$(".links a").forEach((a) =>
    a.classList.toggle("on", a.getAttribute("href") === "#" + p),
  );
  shell.classList.toggle("home", p === "home");
  $("#nv").classList.remove("open");
  $("#bg").setAttribute("aria-expanded", "false");
  scrollTo(0, 0);
  if (p === "compare") renderCompare();
  if (p === "technology") renderTech();
  if (p === "benchmarks") renderBench();
}
addEventListener("hashchange", route);
$("#bg").onclick = () => {
  const o = $("#nv").classList.toggle("open");
  $("#bg").setAttribute("aria-expanded", o);
};
$("#hero").addEventListener("pointermove", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  e.currentTarget.style.setProperty(
    "--px",
    (e.clientX - r.left) / r.width - 0.5,
  );
  e.currentTarget.style.setProperty(
    "--py",
    (e.clientY - r.top) / r.height - 0.5,
  );
});
const empty = (m) =>
  `<div class="empty"><p>${m}</p><a class="pill dk" href="#studio">Open Studio</a></div>`;
const put = (cv, im) => {
  cv.width = im.width;
  cv.height = im.height;
  cv.getContext("2d").putImageData(im, 0, 0);
};
/* analysis + processing (real, classical, in-browser) */
function analyse(im) {
  const { data: d, width: w, height: h } = im,
    Y = (i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  let n = 0,
    s = 0,
    s2 = 0,
    r = 0,
    g = 0,
    b = 0,
    nz = 0;
  const bm = new Float32Array(64),
    bc = new Float32Array(64);
  for (let y = 1; y < h - 1; y += 2)
    for (let x = 1; x < w - 1; x += 2) {
      const i = (y * w + x) * 4,
        v = Y(i) / 255;
      n++;
      s += v;
      s2 += v * v;
      r += d[i];
      g += d[i + 1];
      b += d[i + 2];
      nz +=
        Math.abs(4 * Y(i) - Y(i - 4) - Y(i + 4) - Y(i - w * 4) - Y(i + w * 4)) /
        1020;
      const k =
        Math.min(7, ((y * 8) / h) | 0) * 8 + Math.min(7, ((x * 8) / w) | 0);
      bm[k] += v;
      bc[k]++;
    }
  const m = s / n,
    sd = Math.sqrt(Math.max(0, s2 / n - m * m)),
    mx = Math.max(r, g, b) / n,
    mn = Math.min(r, g, b) / n,
    bs = [...bm].map((v, i) => v / (bc[i] || 1)),
    bmm = bs.reduce((a, c) => a + c) / 64,
    bsd = Math.sqrt(bs.reduce((a, c) => a + (c - bmm) ** 2, 0) / 64);
  return {
    brightness: m,
    contrast: Math.min(1, sd * 2.5),
    noise: Math.min(1, (nz / n) * 10),
    cast: Math.min(1, ((mx - mn) / (mx + 1e-6)) * 2),
    uniformity: Math.max(0, 1 - bsd / (bmm + 0.02)),
  };
}
const bars = (a) =>
  [
    ["Brightness", "brightness"],
    ["Contrast", "contrast"],
    ["Noise (est.)", "noise"],
    ["Colour cast", "cast"],
    ["Illumination uniformity", "uniformity"],
  ]
    .map(
      ([l, k]) =>
        `<div class="brow"><div><span>${l}</span><b>${Math.round(a[k] * 100)}%</b></div><i><b style="width:${Math.round(a[k] * 100)}%"></b></i></div>`,
    )
    .join("");
function blur(src, w, h, n) {
  const f = Float32Array.from(src),
    g = new Float32Array(f.length);
  for (let k = 0; k < n; k++) {
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4,
          l = x ? i - 4 : i,
          r = x < w - 1 ? i + 4 : i;
        for (let c = 0; c < 3; c++)
          g[i + c] = (f[l + c] + f[i + c] + f[r + c]) / 3;
      }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4,
          u = y ? i - w * 4 : i,
          d = y < h - 1 ? i + w * 4 : i;
        for (let c = 0; c < 3; c++)
          f[i + c] = (g[u + c] + g[i + c] + g[d + c]) / 3;
      }
  }
  return f;
}
function enhance(im, p) {
  const { width: w, height: h, data } = im,
    N = w * h * 4,
    f =
      p.denoise > 0
        ? blur(data, w, h, Math.max(1, Math.round(p.denoise / 3.4)))
        : Float32Array.from(data),
    g = 2 ** p.exposure,
    ct = 1 + p.contrast / 100,
    st = 1 + p.saturation / 100,
    gi = 1 / (p.gamma || 1);
  for (let i = 0; i < N; i += 4) {
    const v = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const x = Math.pow(Math.min(1, (f[i + c] / 255) * g), gi);
      v[c] = (x - 0.5) * ct + 0.5;
    }
    const y = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    for (let c = 0; c < 3; c++)
      f[i + c] = Math.min(1, Math.max(0, y + (v[c] - y) * st)) * 255;
  }
  if (p.sharp > 0) {
    const b = blur(f, w, h, 1),
      a = (p.sharp / 100) * 1.5;
    for (let i = 0; i < N; i += 4)
      for (let c = 0; c < 3; c++) f[i + c] += a * (f[i + c] - b[i + c]);
  }
  const o = new ImageData(w, h);
  for (let i = 0; i < N; i += 4) {
    for (let c = 0; c < 3; c++) o.data[i + c] = f[i + c];
    o.data[i + 3] = 255;
  }
  return o;
}
/* studio */
function setStatus(k, m) {
  $("#st").className = "st " + k;
  $("#stx").textContent = m;
  $("#ld").hidden = k !== "busy";
}
const SL = [
  ["exposure", "Exposure (stops)", 0, 4, 0.05],
  ["contrast", "Contrast", -50, 100, 1],
  ["saturation", "Saturation", -50, 100, 1],
  ["denoise", "Denoising", 0, 100, 1],
  ["sharp", "Sharpness", 0, 100, 1],
];
$("#sls").innerHTML = SL.map(
  ([k, l, a, b, s]) =>
    `<div class="sl"><label>${l}<span id="v_${k}"></span></label><input type="range" data-k="${k}" min="${a}" max="${b}" step="${s}"></div>`,
).join("");
function sync() {
  SL.forEach(([k]) => {
    $(`[data-k=${k}]`).value = S.p[k];
    $("#v_" + k).textContent = S.p[k];
  });
}
$$("[data-k]").forEach(
  (i) =>
    (i.oninput = () => {
      S.p[i.dataset.k] = +i.value;
      $("#v_" + i.dataset.k).textContent = i.value;
    }),
);
sync();
$$("[name=md]").forEach((r) => (r.onchange = () => (S.mode = r.value)));
$$("#tabs button").forEach(
  (b) =>
    (b.onclick = () => {
      $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
      $$(".tp").forEach((p) => (p.hidden = p.dataset.t !== b.dataset.t));
    }),
);
function load(src) {
  const w0 = src.naturalWidth || src.width,
    h0 = src.naturalHeight || src.height,
    k = Math.min(1, 1400 / Math.max(w0, h0)),
    c = document.createElement("canvas");
  c.width = Math.round(w0 * k);
  c.height = Math.round(h0 * k);
  const x = c.getContext("2d");
  x.drawImage(src, 0, 0, c.width, c.height);
  S.img = x.getImageData(0, 0, c.width, c.height);
  S.res = null;
  S.t = null;
  S.a = analyse(S.img);
  S.p.gamma = 1;
  put($("#cO"), S.img);
  $("#cR").hidden = $("#hd").hidden = $("#chr").hidden = true;
  $("#dl").disabled = true;
  $("#dz").hidden = true;
  $("#vw").hidden = false;
  zoom(1);
  $("#an").innerHTML = bars(S.a);
  setStatus("idle", "Image loaded. Try Auto Enhance, then Restore.");
}
function loadFile(f) {
  if (!f) return;
  if (!/^image\/(jpeg|png|webp)$/.test(f.type))
    return setStatus("error", "Unsupported format. Use JPG, PNG or WebP.");
  if (f.size > 15e6) return setStatus("error", "That file is over 15 MB.");
  const im = new Image();
  im.onload = () => {
    load(im);
    URL.revokeObjectURL(im.src);
  };
  im.onerror = () => setStatus("error", "Couldn’t decode that image.");
  im.src = URL.createObjectURL(f);
}
$("#fi").onchange = (e) => loadFile(e.target.files[0]);
$("#rp").onclick = () => $("#fi").click();
const dr = $("#drop");
["dragover", "dragenter"].forEach((t) =>
  dr.addEventListener(t, (e) => {
    e.preventDefault();
    dr.classList.add("over");
  }),
);
["dragleave", "drop"].forEach((t) =>
  dr.addEventListener(t, (e) => {
    e.preventDefault();
    dr.classList.remove("over");
  }),
);
dr.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
function zoom(z) {
  S.z = Math.min(4, Math.max(1, z));
  $("#zw").style.width = S.z * 100 + "%";
  $("#zl").textContent = Math.round(S.z * 100) + "%";
}
$("#zi").onclick = () => zoom(S.z + 0.5);
$("#zo").onclick = () => zoom(S.z - 0.5);
$("#zf").onclick = () => zoom(1);
let drag = false;
const mv = (e) => {
  const r = $("#zw").getBoundingClientRect(),
    x = Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100));
  $("#cR").style.clipPath = `inset(0 0 0 ${x}%)`;
  $("#hd").style.left = x + "%";
};
$("#zw").addEventListener("pointerdown", (e) => {
  drag = true;
  mv(e);
});
addEventListener("pointermove", (e) => drag && mv(e));
addEventListener("pointerup", () => (drag = false));
function renderCompare() {} /* implemented with real engine data in lumina.js (Comparison Lab) */
$("#zm").oninput = (e) => $("#cmp").style.setProperty("--z", e.target.value);
$("#cmp").addEventListener("pointermove", (e) => {
  const p = e.target.closest(".pv");
  if (!p) return;
  const r = p.getBoundingClientRect();
  e.currentTarget.style.setProperty(
    "--ox",
    ((e.clientX - r.left) / r.width) * 100 + "%",
  );
  e.currentTarget.style.setProperty(
    "--oy",
    ((e.clientY - r.top) / r.height) * 100 + "%",
  );
});
function renderTech() {} /* implemented with real engine data in lumina.js (technology stages) */
function renderBench() {} /* implemented with real engine data in lumina.js (benchmarks) */
$$(".band h2").forEach(
  (h) => (h.innerHTML = h.textContent.replace(/(\S+)$/, "<em>$1</em>")),
);
const TIPS = {
  exposure:
    "Brightens the whole photo, measured in photographic stops. Each +1 stop doubles the light.",
  contrast: "Separates darks from lights. Too much can crush shadow detail.",
  saturation: "How vivid colours look. Low-light photos often look washed out.",
  denoise:
    "Smooths sensor grain. Higher values remove more noise but also soften fine detail.",
  sharp: "Restores edge crispness after denoising.",
};
$$("#sls .sl").forEach((sl) => {
  const lab = sl.querySelector("label"),
    k = sl.querySelector("input").dataset.k,
    t = document.createElement("span"),
    q = document.createElement("i");
  q.className = "q";
  q.tabIndex = 0;
  q.textContent = "?";
  q.dataset.tip = TIPS[k];
  q.setAttribute("aria-label", TIPS[k]);
  t.append(lab.firstChild, q);
  lab.prepend(t);
});
const grp = (ti, keys, op) => {
  const d = document.createElement("details");
  d.className = "dt";
  d.open = op;
  d.innerHTML = `<summary>${ti}</summary>`;
  $$("#sls .sl")
    .filter((s) => keys.includes(s.querySelector("input").dataset.k))
    .forEach((s) => d.append(s));
  return d;
};
$("#sls").append(
  grp("Light &amp; colour", ["exposure", "contrast", "saturation"], true),
  grp("Detail", ["denoise", "sharp"], false),
);
$$(".mode").forEach((m, i) =>
  m
    .querySelector("small")
    .insertAdjacentHTML(
      "beforebegin",
      i
        ? '<span class="chip un">Backend not connected</span>'
        : '<span class="chip ok">Ready</span>',
    ),
);
const tbs = $$("#tabs button");
tbs.forEach((b) => {
  b.setAttribute("role", "tab");
  b.setAttribute("aria-selected", b.classList.contains("on"));
  b.addEventListener("click", () =>
    tbs.forEach((x) => x.setAttribute("aria-selected", x === b)),
  );
});
const AT = {
  Brightness: "Average light level. Low values mean an underexposed photo.",
  Contrast: "Spread between the darkest and lightest areas.",
  "Noise (est.)":
    "Rough estimate of sensor grain. A heuristic, not a calibrated measurement.",
  "Colour cast": "How strongly one colour tints the image.",
  "Illumination uniformity":
    "How evenly the scene is lit. Low values mean strong bright and dark patches.",
};
const _ld = load;
load = function (s) {
  _ld(s);
  $$("#an .brow span").forEach((e) => {
    e.dataset.tip = AT[e.textContent];
    e.tabIndex = 0;
  });
};
const zw0 = $("#zw");
zw0.tabIndex = 0;
zw0.setAttribute("role", "slider");
zw0.setAttribute(
  "aria-label",
  "Before and after comparison. Use the left and right arrow keys.",
);
zw0.addEventListener("keydown", (e) => {
  if ((e.key !== "ArrowLeft" && e.key !== "ArrowRight") || $("#cR").hidden)
    return;
  e.preventDefault();
  let x = parseFloat($("#hd").style.left) || 50;
  x = Math.min(100, Math.max(0, x + (e.key === "ArrowLeft" ? -4 : 4)));
  const r = zw0.getBoundingClientRect();
  mv({ clientX: r.left + (r.width * x) / 100 });
});
