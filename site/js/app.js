import { prepare, forward, softmax, argmax } from "./net.js";
import { computeFacts } from "./facts.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const NAMES = ["Annual crop", "Forest", "Herbaceous vegetation", "Highway", "Industrial", "Pasture", "Permanent crop", "Residential", "River", "Sea / lake"];
const ABBR = ["AC", "Fo", "HV", "Hw", "In", "Pa", "PC", "Re", "Ri", "SL"];
const GROUP = ["farm", "nature", "nature", "built", "built", "nature", "farm", "built", "nature", "nature"];
const GROUP_INFO = { nature: ["Natural surfaces", "var(--blue)"], farm: ["Farmland", "var(--aqua)"], built: ["Built-up", "var(--orange)"] };
const DETECTORS = ["Blue band", "Green band", "Red band", "Near infrared", "Shortwave IR 1", "Shortwave IR 2", "Red edge",
  "Water", "Vegetation", "Bare / built", "Edges ↔", "Edges ↕", "Bright spots", "Dark spots", "Infrared edges ↔", "Infrared edges ↕"];
const INIT = { kaiming: ["Random start (Kaiming)", "var(--slate)", "Random start"], fable_as_written: ["Written network, as written", "var(--orange)", "As written"],
  fable_rescaled: ["Written network, rescaled", "var(--blue)", "Rescaled"] };
const sd = (a) => { const m = mean(a); return a.length > 1 ? Math.sqrt(a.reduce((q, v) => q + (v - m) ** 2, 0) / (a.length - 1)) : 0; };
const T975 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179];   // t quantiles by degrees of freedom
const COND = { none: ["No augmentation", "var(--aqua)"], standard: ["Standard augmentation", "var(--orange)"], strong: ["Strong augmentation", "var(--blue)"] };

const D = {};
let samples = null;                       // Float32Array per sample image, loaded after first paint

// ── small UI helpers ─────────────────────────────────────────────────────────
function fit(el, draw) {
  let last = 0;
  const run = () => { const w = Math.floor(el.clientWidth); if (w && w !== last) { last = w; draw(w); } };
  new ResizeObserver(run).observe(el);
  run();
  return () => { last = 0; run(); };
}
function seg(el, onChange) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  });
}
function chips(el, labels, onChange, initial = 0) {
  el.innerHTML = labels.map((l, i) => `<button data-i="${i}" class="${i === initial ? "on" : ""}">${esc(l)}</button>`).join("");
  const set = (i) => $$("button", el).forEach((b) => b.classList.toggle("on", +b.dataset.i === i));
  el.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { set(+b.dataset.i); onChange(+b.dataset.i); } });
  return set;
}
function thumbStyle(k, mode) {
  const c = Math.floor(k / D.net.per_class), j = k % D.net.per_class;
  return `background-image:url(data/samples_${mode}.png);background-position:${(c / 9) * 100}% ${(j / (D.net.per_class - 1)) * 100}%`;
}
const tip = $("#tip");
function showTip(e) {
  const t = e.target.closest?.("[data-tip]");
  if (!t) { tip.classList.remove("on"); return; }
  tip.textContent = t.dataset.tip;
  tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.max(6, Math.min(window.innerWidth - r.width - 6, e.clientX - r.width / 2)) + "px";
  tip.style.top = (e.clientY - r.height - 12 < 6 ? e.clientY + 16 : e.clientY - r.height - 12) + "px";
}
document.addEventListener("pointermove", showTip);
document.addEventListener("pointerdown", showTip);
window.addEventListener("scroll", () => tip.classList.remove("on"), { passive: true });

// ── confusion matrix ─────────────────────────────────────────────────────────
function matrix(el) {
  let cm = null;
  const paint = () => {
    if (!cm) return;
    $$("rect.cell", el).forEach((r) => {
      const i = +r.dataset.i, j = +r.dataset.j, n = cm[i].reduce((a, b) => a + b, 0), f = cm[i][j] / n;
      r.setAttribute("fill-opacity", f === 0 ? 0 : 0.08 + 0.92 * Math.pow(f, 0.6));
      r.dataset.tip = `True class: ${NAMES[i]}\nPredicted: ${NAMES[j]}\n${cm[i][j]} of ${n} images (${pct(f, 0)})`;
    });
  };
  fit(el, (w) => {
    const L = w < 380 ? 92 : 132, c = (w - L) / 10, g = Math.max(1, c * 0.06);
    let s = `<svg viewBox="0 0 ${w} ${c * 10 + 40}" role="img" aria-label="Confusion matrix">`;
    for (let i = 0; i < 10; i++) {
      s += `<text x="${L - 8}" y="${i * c + c / 2 + 4}" text-anchor="end">${w < 380 ? NAMES[i].split(" ")[0] : NAMES[i]}</text>`;
      s += `<text x="${L + i * c + c / 2}" y="${c * 10 + 15}" text-anchor="middle" class="faint">${ABBR[i]}</text>`;
      for (let j = 0; j < 10; j++) {
        s += `<rect x="${L + j * c}" y="${i * c}" width="${c - g}" height="${c - g}" rx="${c * 0.12}" fill="var(--bg)"/>`;
        s += `<rect class="cell" data-i="${i}" data-j="${j}" x="${L + j * c}" y="${i * c}" width="${c - g}" height="${c - g}" rx="${c * 0.12}" fill="var(--${i === j ? "blue" : "orange"})" fill-opacity="0"/>`;
      }
    }
    s += `<text x="${L + c * 5}" y="${c * 10 + 34}" text-anchor="middle" class="faint">${w < 420 ? "predicted · blue = correct" : "predicted class · blue = correct, orange = mistakes"}</text></svg>`;
    el.innerHTML = s;
    paint();
  });
  return (m) => { cm = m; paint(); };
}

// ── dot rows: a few runs per starting point ──────────────────────────────────
function dotRows(el, getRows) {
  return fit(el, (w) => {
    const { rows, ref, refLabel } = getRows();
    const all = rows.flatMap((r) => r.values).concat(ref ?? []);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = Math.max(0.004, (hi - lo) * 0.12); lo -= pad; hi += pad;
    const l = 8, r = 58, rh = 54, top = ref != null ? 18 : 4, h = top + rows.length * rh + 28;
    const x = (v) => l + ((v - lo) / (hi - lo)) * (w - l - r);
    let s = `<svg viewBox="0 0 ${w} ${h}">`;
    const step = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1].find((t) => (hi - lo) / t <= 6) || 0.1;
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${h - 24}" stroke="var(--grid)"/>`;
      s += `<text x="${x(t)}" y="${h - 8}" text-anchor="middle" class="faint">${(t * 100).toFixed(step < 0.01 ? 1 : 0)}%</text>`;
    }
    if (ref != null) {
      s += `<line x1="${x(ref)}" x2="${x(ref)}" y1="${top - 2}" y2="${h - 24}" stroke="var(--ink2)" stroke-dasharray="4 3"/>`;
      s += `<text x="${x(ref) + 5}" y="11" class="faint">${refLabel}</text>`;
    }
    rows.forEach((row, i) => {
      const y = top + i * rh + 36, m = mean(row.values);
      s += `<text x="${l}" y="${y - 20}" class="strong halo">${row.label}</text>`;
      s += `<line x1="${x(m)}" x2="${x(m)}" y1="${y - 11}" y2="${y + 11}" stroke="${row.color}" stroke-width="3" stroke-linecap="round"/>`;
      row.values.forEach((v, k) => {
        s += `<circle cx="${x(v)}" cy="${y}" r="6.5" fill="${row.color}" stroke="var(--surface)" stroke-width="2" data-tip="${row.label}\nseed ${k}: ${pct(v, 2)}"/>`;
      });
      s += `<text x="${w - 2}" y="${y + 4}" text-anchor="end" class="strong">${pct(m)}</text>`;
    });
    el.innerHTML = s + "</svg>";
  });
}

// ── stepper ──────────────────────────────────────────────────────────────────
function stepper(el, title, steps) {
  let k = 0;
  const draw = () => {
    const s = steps[k];
    el.innerHTML = `<div class="head"><span class="label">${title}</span><span class="dots">${steps.map((_, i) =>
      `<button data-i="${i}" class="${i === k ? "on" : ""}" aria-label="Step ${i + 1}">${i + 1}</button>`).join("")}</span>
      <button class="btn next">${k === steps.length - 1 ? "↺ Start over" : "Next →"}</button></div>
      <div class="time">${s.time}</div><h4>${s.title}</h4><p>${s.text}</p>${s.quote ? `<blockquote>“${s.quote}”</blockquote>` : ""}`;
  };
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    k = b.classList.contains("next") ? (k + 1) % steps.length : +b.dataset.i;
    draw();
  });
  draw();
}

// ── hero ─────────────────────────────────────────────────────────────────────
function hero() {
  const rounds = D.write.w3.rounds, base = D.write.w3.baselines;
  const set = matrix($("#hero-matrix"));
  const acc = $("#hero-acc");
  const states = ["First attempt, written with no feedback", "After 1 round of feedback", "After 2 rounds of feedback", "After 3 rounds of feedback"];
  let shown = rounds[0].test.accuracy, timer = null, raf = 0;
  const go = (r) => {
    setChip(r);
    set(rounds[r].test.confusion);
    $("#hero-state").textContent = states[r];
    const from = shown, to = rounds[r].test.accuracy, t0 = performance.now();
    cancelAnimationFrame(raf);
    const tick = (t) => {
      const u = Math.min(1, (t - t0) / 600);
      shown = from + (to - from) * (1 - Math.pow(1 - u, 3));
      acc.textContent = pct(shown);
      if (u < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  };
  const stop = () => { clearInterval(timer); timer = null; $("#hero-play").textContent = "▶ Play the feedback rounds"; };
  const setChip = chips($("#hero-chips"), rounds.map((r) => "Round " + r.round), (i) => { stop(); go(i); });
  $("#hero-play").addEventListener("click", () => {
    if (timer) return stop();
    let r = 0; go(0);
    $("#hero-play").textContent = "❚❚ Pause";
    timer = setInterval(() => { r++; go(r); if (r >= rounds.length - 1) stop(); }, 1700);
  });
  $("#hero-refs").innerHTML = `<div>For comparison</div>
    <div><b>${pct(base.constant)}</b> always guessing the most common class</div>
    <div><b>${pct(base.centroid)}</b> a simple classifier fitted on real data</div>
    <div><b>${pct(base.trained)}</b> the same network, trained normally</div>`;
  go(0);
}

// ── gallery and architecture ─────────────────────────────────────────────────
function gallery() {
  const el = $("#gallery");
  const draw = (mode) => {
    el.innerHTML = NAMES.map((n, c) => `<div class="col"><div class="name">${n}</div>${[0, 1, 2].slice(0, D.net.per_class).map((j) =>
      `<div class="thumb" style="${thumbStyle(c * D.net.per_class + j, mode)}"></div>`).join("")}</div>`).join("");
  };
  seg($("#gallery-mode"), draw);
  draw("rgb");
}
function arch() {
  const stages = [
    ["Image", "13 bands, 64 × 64 pixels", "One satellite patch. Each pixel holds 13 measurements of reflected light, from blue to shortwave infrared."],
    ["16 detectors", "first layer · 1,888 weights", "Each detector slides a 3 × 3 window over the image and mixes the 13 bands. The model decided what each one should look for: a band, water, vegetation, an edge."],
    ["Shrink", "average over 4 × 4 blocks", "The 16 response maps are averaged over blocks of 4 × 4 pixels, about 40 metres on the ground. No weights here."],
    ["32 features", "second layer · 4,640 weights", "A second set of 3 × 3 windows combines the detectors into 32 features, such as “share of the image that is dark in infrared” or “coarse edges”."],
    ["One value each", "average over the image", "Each feature is averaged over the whole image. The image is now described by 32 numbers."],
    ["10 class scores", "last layer · 330 weights", "Each class gets a score from the 32 numbers. The model wrote this layer as “how close is this image to my idea of each class?”. The highest score wins."],
  ];
  $("#arch").innerHTML = stages.map((s, i) => `${i ? "<i>→</i>" : ""}<button data-i="${i}"><b>${s[0]}</b><span>${s[1]}</span></button>`).join("");
  const pick = (i) => { $$("#arch button").forEach((b) => b.classList.toggle("on", +b.dataset.i === i)); $("#arch-note").textContent = stages[i][2]; };
  $("#arch").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) pick(+b.dataset.i); });
  $("#arch").addEventListener("pointerover", (e) => { const b = e.target.closest("button"); if (b) pick(+b.dataset.i); });
  pick(1);
}

// ── live network ─────────────────────────────────────────────────────────────
const nets = [];
function ramp(t) {
  const a = [13, 20, 34], b = [77, 151, 238], c = [255, 255, 255];
  const [p, q, u] = t < 0.6 ? [a, b, t / 0.6] : [b, c, (t - 0.6) / 0.4];
  return [0, 1, 2].map((k) => p[k] + (q[k] - p[k]) * u);
}
function live() {
  let k = 4, r = D.net.rounds.length - 1, mode = "rgb", detMax = null;
  const thumbs = $("#live-thumbs"), maps = $("#live-maps");
  const pc = D.net.per_class, grid = D.net.labels.map((_, p) => (p % 10) * pc + Math.floor(p / 10));
  thumbs.innerHTML = grid.map((i) => `<button class="thumb" data-k="${i}" style="${thumbStyle(i, mode)}" aria-label="${NAMES[D.net.labels[i]]} image"></button>`).join("");
  maps.innerHTML = DETECTORS.map((n) => `<figure><canvas width="64" height="64"></canvas><figcaption>${n}</figcaption></figure>`).join("");
  const draw = () => {
    $$(".thumb", thumbs).forEach((b) => b.classList.toggle("on", +b.dataset.k === k));
    if (!samples) { $("#live-verdict").textContent = "Loading the test images…"; return; }
    const out = forward(nets[r], samples[k], D.net.used_bands);
    if (!detMax) {                        // one brightness scale per detector, shared by all sample images
      detMax = new Float64Array(16).fill(1e-9);
      for (const im of samples) { const a = forward(nets[r], im, D.net.used_bands).a1; for (let j = 0; j < a.length; j++) if (a[j] > detMax[j >> 12]) detMax[j >> 12] = a[j]; }
    }
    $$("canvas", maps).forEach((cv, o) => {
      const ctx = cv.getContext("2d"), img = ctx.createImageData(64, 64);
      const mx = detMax[o];
      for (let j = 0; j < 4096; j++) {
        const c = ramp(Math.sqrt(Math.min(1, out.a1[o * 4096 + j] / mx)));
        img.data[j * 4] = c[0]; img.data[j * 4 + 1] = c[1]; img.data[j * 4 + 2] = c[2]; img.data[j * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    });
    const L = Array.from(out.logits), win = argmax(L), truth = D.net.labels[k];
    const lo = Math.min(...L), hi = Math.max(...L);
    $("#live-scores").innerHTML = `<div class="bars">${L.map((v, c) => `<div class="row ${c === win ? "win" : ""} ${c === win && win !== truth ? "miss" : ""} ${c === truth ? "truth" : ""}">
      <span class="name">${NAMES[c]}${c === truth ? " ●" : ""}</span><span class="track"><span class="fill" style="display:block;width:${4 + 96 * (v - lo) / (hi - lo || 1)}%"></span></span><span class="val">${v.toFixed(0)}</span></div>`).join("")}</div>`;
    $("#live-verdict").innerHTML = win === truth
      ? `<span class="ok">It says ${NAMES[win]}. Correct.</span>`
      : `<span class="no">It says ${NAMES[win]}. The image is ${NAMES[truth]} (●).</span>`;
  };
  thumbs.addEventListener("click", (e) => { const b = e.target.closest(".thumb"); if (b) { k = +b.dataset.k; draw(); } });
  chips($("#live-rounds"), D.net.rounds.map((x) => x.round === 0 ? "First attempt" : "Round " + x.round), (i) => { r = i; draw(); }, r);
  draw();
  return draw;
}

// ── feedback rounds ──────────────────────────────────────────────────────────
function rounds() {
  let exp = "w3", r = 0;
  const series = { w3: ["13 spectral bands", "var(--blue)"], w2: ["Red, green, blue only", "var(--orange)"] };
  const redraw = fit($("#rounds-chart"), (w) => {
    const l = 38, rt = w < 420 ? 96 : 150, t = 30, b = 34, h = 320, ymax = 0.9;
    const x = (i) => l + 34 + (i / 3) * (w - l - rt - 48), y = (v) => t + (1 - v / ymax) * (h - t - b);
    let s = `<svg viewBox="0 0 ${w} ${h}">`;
    for (let v = 0; v <= 0.81; v += 0.2) s += `<line x1="${l}" x2="${w - rt}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${y(v) + 4}" text-anchor="end" class="faint">${v * 100 | 0}%</text>`;
    const base = D.write[exp].baselines;
    for (const [key, label] of [["trained", "trained normally"], ["centroid", "fitted on data"], ["constant", "constant guess"]]) {
      s += `<line x1="${l}" x2="${w - rt + 6}" y1="${y(base[key])}" y2="${y(base[key])}" stroke="var(--ink2)" stroke-dasharray="4 3"/>`;
      s += `<text x="${w - rt + 10}" y="${y(base[key]) + 4}"><tspan class="strong">${pct(base[key])}</tspan>${w < 420 ? "" : " " + label}</text>`;
      if (w < 420) s += `<text x="${w - rt + 10}" y="${y(base[key]) + 17}" class="faint">${label}</text>`;
    }
    for (const e of exp === "w3" ? ["w2", "w3"] : ["w3", "w2"]) {
      const on = e === exp, acc = D.write[e].rounds.map((q) => q.test.accuracy), col = on ? series[e][1] : "var(--axis)";
      s += `<polyline points="${acc.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="${on ? 2.5 : 2}" stroke-linejoin="round"/>`;
      acc.forEach((v, i) => {
        s += `<circle cx="${x(i)}" cy="${y(v)}" r="${on && i === r ? 8 : 5}" fill="${col}" stroke="var(--surface)" stroke-width="2" data-tip="${series[e][0]}\nround ${i}: ${pct(v)} on the test images"/>`;
        if (on && i === r) s += `<text x="${x(i)}" y="${y(v) - 14}" text-anchor="middle" class="strong halo" style="font-size:14px">${pct(v)}</text>`;
      });
    }
    for (let i = 0; i < 4; i++) s += `<text x="${x(i)}" y="${h - 10}" text-anchor="middle" class="${i === r ? "strong" : ""}">${w < 480 ? (i === 0 ? "first" : "round " + i) : i === 0 ? "first attempt" : "round " + i}</text>`;
    s += `<circle cx="${l + 6}" cy="10" r="5" fill="${series[exp][1]}"/><text x="${l + 16}" y="14">${series[exp][0]}</text>`;
    const other = exp === "w3" ? "w2" : "w3", off = l + 30 + series[exp][0].length * 6.6;
    s += `<circle cx="${off}" cy="10" r="5" fill="var(--axis)"/><text x="${off + 10}" y="14" class="faint">${series[other][0]}</text>`;
    $("#rounds-chart").innerHTML = s + "</svg>";
  });
  const text = () => {
    const R = D.write[exp].rounds, q = R[r];
    $("#rounds-title").textContent = r === 0 ? "First attempt · the model's explanation" : `Revision after feedback round ${r} · the model's explanation`;
    const said = $("#rounds-said"), none = q.explanation.startsWith("[");
    said.classList.toggle("none", none);
    said.textContent = none ? "The explanation for this round was not saved, only the code." : q.explanation;
    said.scrollTop = 0;
    let moves = "";
    if (r > 0) {
      const d = q.practice.recall.map((v, c) => [v - R[r - 1].practice.recall[c], c]).sort((a, b) => b[0] - a[0]);
      const fmt = ([, c]) => `${NAMES[c]} ${pct(R[r - 1].practice.recall[c], 0)} → ${pct(q.practice.recall[c], 0)}`;
      const up = d.slice(0, 2).filter((v) => v[0] > 0.04), down = d.slice(-1).filter((v) => v[0] < -0.04);
      moves = `<b>Largest changes on the practice images.</b> ${[...up, ...down].map(fmt).join(" · ") || "None above 4 points."}`;
    }
    $("#rounds-moves").innerHTML = moves + `<div>Practice score shown to the model: ${pct(q.practice.accuracy)}. Test score, never shown: ${pct(q.test.accuracy)}.</div>`;
  };
  const setChip = chips($("#rounds-chips"), ["First attempt", "Round 1", "Round 2", "Round 3"], (i) => { r = i; redraw(); text(); });
  seg($("#rounds-mode"), (v) => { exp = v; redraw(); text(); });
  setChip(0); text();
}

// ── spectra ──────────────────────────────────────────────────────────────────
function spectra() {
  let c = 1;
  const bel = D.spectra.believed[0];
  const order = bel.bands.map((b, j) => ({ b, j, ...D.write.bands[b] })).sort((p, q) => p.nm - q.nm);
  const redraw = fit($("#spectra-chart"), (w) => {
    const l = 44, rt = 14, t = 28, b = 48, h = 300, ymax = 0.4;
    const x = (i) => l + (w < 480 ? 14 : 40) + (i / (order.length - 1)) * (w - l - rt - (w < 480 ? 28 : 90)), y = (v) => t + (1 - v / ymax) * (h - t - b);
    let s = `<svg viewBox="0 0 ${w} ${h}">`;
    for (let v = 0; v <= 0.41; v += 0.1) s += `<line x1="${l}" x2="${w - rt}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${y(v) + 4}" text-anchor="end" class="faint">${v.toFixed(1)}</text>`;
    const m = order.map((o) => D.spectra.measured[c][o.b]), f = order.map((o) => bel.values[o.j][c]);
    s += `<polyline points="${f.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="var(--blue)" stroke-width="2.5" stroke-dasharray="6 4"/>`;
    s += `<polyline points="${m.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="var(--ink)" stroke-width="2"/>`;
    order.forEach((o, i) => {
      s += `<circle cx="${x(i)}" cy="${y(f[i])}" r="5" fill="var(--blue)" stroke="var(--surface)" stroke-width="2" data-tip="${o.id} ${o.name}\nmodel assumed ${f[i].toFixed(3)}\nmeasured ${m[i].toFixed(3)}"/>`;
      s += `<circle cx="${x(i)}" cy="${y(m[i])}" r="4" fill="var(--ink)" stroke="var(--surface)" stroke-width="2" data-tip="${o.id} ${o.name}\nmeasured ${m[i].toFixed(3)}\nmodel assumed ${f[i].toFixed(3)}"/>`;
      s += `<text x="${x(i)}" y="${h - 28}" text-anchor="middle" class="strong">${w < 480 ? o.id : o.name.replace("shortwave infrared", "shortwave IR")}</text><text x="${x(i)}" y="${h - 12}" text-anchor="middle" class="faint">${w < 480 ? o.nm : o.nm + " nm"}</text>`;
    });
    s += `<line x1="${l}" x2="${l + 22}" y1="10" y2="10" stroke="var(--ink)" stroke-width="2"/><text x="${l + 28}" y="14">measured</text>`;
    s += `<line x1="${l + 100}" x2="${l + 122}" y1="10" y2="10" stroke="var(--blue)" stroke-width="2.5" stroke-dasharray="6 4"/><text x="${l + 128}" y="14">what the model assumed</text>`;
    s += `<text x="${l}" y="${t - 6}" class="faint" style="display:none">reflectance</text>`;
    $("#spectra-chart").innerHTML = s + "</svg>";
  });
  chips($("#spectra-classes"), NAMES, (i) => { c = i; redraw(); }, c);
}

// ── recall per class ─────────────────────────────────────────────────────────
function recall() {
  const rec = D.write.w3.rounds.at(-1).test.recall;
  const idx = rec.map((v, c) => c).sort((a, b) => rec[b] - rec[a]);
  $("#recall-chart").innerHTML = `<div class="legend">${Object.values(GROUP_INFO).map(([n, col]) => `<span><i style="background:${col}"></i>${n}</span>`).join("")}</div>
    <div class="bars" style="font-size:13.5px">${idx.map((c) => `<div class="row" style="grid-template-columns:150px minmax(0,1fr) 44px" data-tip="${NAMES[c]}: ${pct(rec[c])} of its test images recognised">
    <span class="name">${NAMES[c]}</span><span class="track" style="height:18px"><span class="fill" style="display:block;width:${rec[c] * 100}%;background:${GROUP_INFO[GROUP[c]][1]}"></span></span><span class="val">${pct(rec[c], 0)}</span></div>`).join("")}</div>`;
}

// ── starting points ──────────────────────────────────────────────────────────
function init() {
  const budgets = [...new Set(D.init.runs.map((r) => r.budget))].sort((a, b) => a - b);
  const keys = Object.keys(INIT);
  const runs = (b, i) => D.init.runs.filter((r) => r.budget === b && r.init === i).sort((p, q) => p.seed - q.seed);
  const nImg = (b) => runs(b, "kaiming")[0].n_train.toLocaleString("en-US");
  let b = budgets[1];
  $("#init-legend").innerHTML = Object.values(INIT).map(([n, col]) => `<span><i style="background:${col}"></i>${n}</span>`).join("")
    + `<span><i style="background:none;border-top:2px dashed var(--ink2);border-radius:0;width:18px;height:0;vertical-align:middle"></i>Written network before any training</span>`;

  // A. every budget at once: individual runs, mean and one standard deviation
  fit($("#init-strip"), (w) => {
    const l = 46, rt = 10, t = 16, bt = 50, h = w < 520 ? 340 : 380;
    const all = D.init.runs.map((r) => r.test).concat(D.init.fable_untrained);
    const lo = Math.floor(Math.min(...all) * 10 - 0.2) / 10, hi = Math.ceil(Math.max(...all) * 10 + 0.2) / 10;
    const y = (v) => t + (1 - (v - lo) / (hi - lo)) * (h - t - bt), gw = (w - l - rt) / budgets.length, slot = gw / 3.6;
    let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Test accuracy per starting point and label budget">`;
    for (let v = lo; v <= hi + 1e-9; v += 0.1) s += `<line x1="${l}" x2="${w - rt}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    const ry = y(D.init.fable_untrained);
    s += `<line x1="${l}" x2="${w - rt}" y1="${ry}" y2="${ry}" stroke="var(--ink2)" stroke-dasharray="5 4"/>`;
    s += `<text x="${w - rt - 4}" y="${ry + 15}" text-anchor="end" class="halo">${w < 520 ? "before training" : "written network before any training"}: ${pct(D.init.fable_untrained)}</text>`;
    budgets.forEach((bud, g) => {
      const gx = l + g * gw;
      if (g) s += `<line x1="${gx}" x2="${gx}" y1="${t}" y2="${h - bt + 6}" stroke="var(--grid)"/>`;
      s += `<text x="${gx + gw / 2}" y="${h - 26}" text-anchor="middle" class="strong" style="font-size:13.5px">${nImg(bud)}${w < 520 ? "" : " labelled images"}</text>`;
      s += `<text x="${gx + gw / 2}" y="${h - 9}" text-anchor="middle">${w < 520 ? "images (" + bud * 100 + "%)" : bud * 100 + "% of the training set"}</text>`;
      keys.forEach((k, j) => {
        const [name, col] = INIT[k], v = runs(bud, k).map((r) => r.test), m = mean(v), dev = sd(v);
        const cx = gx + gw / 2 + (j - 1) * slot;
        runs(bud, k).forEach((r) => {
          const jx = (((r.seed * 7 + j * 3) % 9) - 4) / 4 * slot * 0.2;
          s += `<circle cx="${cx - slot * 0.2 + jx * 0.6}" cy="${y(r.test)}" r="3.4" fill="${col}" fill-opacity=".45" data-tip="${name}\n${nImg(bud)} images, seed ${r.seed}: ${pct(r.test)}"/>`;
        });
        const mx = cx + slot * 0.22;
        s += `<line x1="${mx}" x2="${mx}" y1="${y(m - dev)}" y2="${y(m + dev)}" stroke="${col}" stroke-width="2.5"/>`;
        for (const e of [m - dev, m + dev]) s += `<line x1="${mx - 5}" x2="${mx + 5}" y1="${y(e)}" y2="${y(e)}" stroke="${col}" stroke-width="2.5"/>`;
        s += `<circle cx="${mx}" cy="${y(m)}" r="6" fill="${col}" stroke="var(--surface)" stroke-width="2" data-tip="${name}\n${nImg(bud)} images\nmean ${pct(m)}, standard deviation ${(dev * 100).toFixed(1)} points, ${v.length} runs"/>`;
        if (w >= 520) s += `<text x="${mx}" y="${y(m + dev) - 8}" text-anchor="middle" class="strong halo" style="font-size:${w < 520 ? 11.5 : 13}px">${(m * 100).toFixed(1)}</text>`;
      });
    });
    $("#init-strip").innerHTML = s + "</svg>";
  });

  // the same numbers as a table, with the paired comparison against the random start
  let rows = "";
  budgets.forEach((bud, g) => {
    const base = runs(bud, "kaiming");
    keys.forEach((k, j) => {
      const rs = runs(bud, k), v = rs.map((r) => r.test);
      let gain = "", ahead = "";
      if (k !== "kaiming") {
        const d = rs.map((r, i) => r.test - base[i].test), half = T975[d.length - 1] * sd(d) / Math.sqrt(d.length);
        gain = `${Math.abs(mean(d)) < 0.0005 ? "" : mean(d) > 0 ? "+" : "−"}${Math.abs(mean(d) * 100).toFixed(1)} (${((mean(d) - half) * 100).toFixed(1)} to ${((mean(d) + half) * 100).toFixed(1)})`.replace(/-/g, "−");
        ahead = `${d.filter((x) => x > 0).length} of ${d.length}`;
      }
      rows += `<tr class="${j === 0 && g ? "sep" : ""}"><td>${j === 0 ? nImg(bud) : ""}</td><td><i style="background:${INIT[k][1]}"></i>${INIT[k][2]}</td><td>${(mean(v) * 100).toFixed(1)} ± ${(sd(v) * 100).toFixed(1)}</td><td>${(Math.min(...v) * 100).toFixed(1)} to ${(Math.max(...v) * 100).toFixed(1)}</td><td>${gain}</td><td>${ahead}</td></tr>`;
    });
  });
  $("#init-table").innerHTML = `<thead><tr><th>Labelled images</th><th>Starting point</th><th>Test accuracy, mean ± s.d.</th><th>Lowest to highest run</th><th>Gain over random start, same seed (95% interval)</th><th>Runs ahead of random start</th></tr></thead><tbody>${rows}</tbody>`;

  // B. training curves: mean and one standard deviation
  $("#init-budget").innerHTML = budgets.map((v) => `<button data-v="${v}" class="${v === b ? "on" : ""}">${nImg(v)} images (${v * 100}%)</button>`).join("");
  const curves = fit($("#init-curves"), (w) => {
    const l = 46, rt = 12, t = 10, bt = 46, h = w < 520 ? 280 : 340, n = runs(b, "kaiming")[0].val.length;
    const x = (i) => l + (i / (n - 1)) * (w - l - rt), y = (v) => t + (1 - v) * (h - t - bt);
    let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Validation accuracy per epoch">`;
    for (let v = 0; v <= 1.001; v += 0.2) s += `<line x1="${l}" x2="${w - rt}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    const step = n <= 30 ? 5 : n <= 60 ? 10 : 40;
    for (let e = step; e <= n; e += step) s += `<line x1="${x(e - 1)}" x2="${x(e - 1)}" y1="${h - bt}" y2="${h - bt + 5}" stroke="var(--axis)"/><text x="${x(e - 1)}" y="${h - bt + 19}" text-anchor="middle">${e}</text>`;
    s += `<text x="${x(0)}" y="${h - bt + 19}" text-anchor="middle">1</text><text x="${(l + w - rt) / 2}" y="${h - 6}" text-anchor="middle" class="axis-title">training epoch</text>`;
    const means = {}, devs = {};
    for (const k of keys) {
      const rs = runs(b, k);
      means[k] = rs[0].val.map((_, i) => mean(rs.map((q) => q.val[i])));
      devs[k] = rs[0].val.map((_, i) => sd(rs.map((q) => q.val[i])));
      const up = means[k].map((m, i) => `${x(i)},${y(Math.min(1, m + devs[k][i]))}`), dn = means[k].map((m, i) => `${x(i)},${y(Math.max(0, m - devs[k][i]))}`).reverse();
      s += `<polygon points="${up.concat(dn).join(" ")}" fill="${INIT[k][1]}" fill-opacity=".16"/>`;
    }
    for (const k of keys) s += `<polyline points="${means[k].map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="${INIT[k][1]}" stroke-width="2.4" stroke-linejoin="round"/>`;
    s += `<line id="cross" y1="${t}" y2="${h - bt}" stroke="var(--ink2)" visibility="hidden"/><rect id="hit" x="${l}" y="${t}" width="${w - l - rt}" height="${h - t - bt}" fill="transparent"/>`;
    const el = $("#init-curves"); el.innerHTML = s + "</svg>";
    const hit = $("#hit", el), cross = $("#cross", el);
    const move = (e) => {
      const box = hit.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - box.left) / box.width) * (n - 1))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      hit.dataset.tip = `Epoch ${i + 1}, mean ± s.d.\n` + keys.map((k) => `${INIT[k][2]}: ${(means[k][i] * 100).toFixed(1)} ± ${(devs[k][i] * 100).toFixed(1)}`).join("\n");
    };
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", () => cross.setAttribute("visibility", "hidden"));
  });

  // C. recall per class, random start against the rescaled written network
  const classes = fit($("#init-classes"), (w) => {
    const withRecall = (k) => runs(b, k).filter((r) => r.recall);
    const A = withRecall("kaiming"), B = withRecall("fable_rescaled");
    if (!A.length) { $("#init-classes").textContent = ""; return; }
    const ra = NAMES.map((_, c) => mean(A.map((r) => r.recall[c]))), rb = NAMES.map((_, c) => mean(B.map((r) => r.recall[c])));
    const order = NAMES.map((_, c) => c).sort((p, q) => (rb[q] - ra[q]) - (rb[p] - ra[p]));
    const l = w < 520 ? 118 : 170, rt = 54, t = 6, rh = 27, h = t + rh * 10 + 46, lo = Math.floor(Math.min(...ra, ...rb) * 10) / 10;
    const x = (v) => l + ((v - lo) / (1 - lo)) * (w - l - rt);
    let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Recall per class for two starting points">`;
    for (let v = lo; v <= 1.001; v += w < 520 ? 0.2 : 0.1) s += `<line x1="${x(v)}" x2="${x(v)}" y1="${t}" y2="${t + rh * 10}" stroke="var(--grid)"/><text x="${x(v)}" y="${t + rh * 10 + 16}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    order.forEach((c, i) => {
      const yy = t + i * rh + rh / 2, d = rb[c] - ra[c];
      s += `<text x="${l - 10}" y="${yy + 4}" text-anchor="end" class="strong">${w < 520 ? NAMES[c].split(" ")[0] : NAMES[c]}</text>`;
      s += `<line x1="${x(ra[c])}" x2="${x(rb[c])}" y1="${yy}" y2="${yy}" stroke="var(--axis)" stroke-width="3"/>`;
      s += `<circle cx="${x(ra[c])}" cy="${yy}" r="5.5" fill="var(--slate)" stroke="var(--surface)" stroke-width="2" data-tip="${NAMES[c]}\nRandom start: ${pct(ra[c])}"/>`;
      s += `<circle cx="${x(rb[c])}" cy="${yy}" r="5.5" fill="var(--blue)" stroke="var(--surface)" stroke-width="2" data-tip="${NAMES[c]}\nRescaled written network: ${pct(rb[c])}"/>`;
      s += `<text x="${w - 2}" y="${yy + 4}" text-anchor="end" class="strong">${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)}</text>`;
    });
    s += `<text x="${(l + w - rt) / 2}" y="${h - 2}" text-anchor="middle" class="axis-title">${w < 520 ? "recall" : "share of the class's test images recognised"} (mean of ${A.length} runs)</text>`;
    $("#init-classes").innerHTML = s + "</svg>";
  });
  seg($("#init-budget"), (v) => { b = +v; curves(); classes(); });
}

function temperature() {
  let k = 22, picked = false;
  const net = () => nets[nets.length - 1];
  const draw = () => {
    $("#temp-thumb").style.cssText = thumbStyle(k, "rgb");
    const truth = D.net.labels[k];
    $("#temp-truth").textContent = "True class: " + NAMES[truth];
    if (!samples) return;
    if (!picked) {
      picked = true;
      const bad = samples.findIndex((im, i) => { const L = forward(net(), im, D.net.used_bands).logits; return argmax(L) !== D.net.labels[i] && Math.max(...softmax(L)) > 0.999; });
      if (bad >= 0) { k = bad; return draw(); }
    }
    const t = +$("#temp-slider").value / 100, scale = Math.pow(D.net.rescale, t);
    const L = forward(net(), samples[k], D.net.used_bands).logits, p = softmax(L, scale), win = argmax(L);
    $("#temp-bars").innerHTML = `<div class="bars">${p.map((v, c) => `<div class="row ${c === win ? "win" : ""} ${c === win && win !== truth ? "miss" : ""} ${c === truth ? "truth" : ""}">
      <span class="name">${NAMES[c]}${c === truth ? " ●" : ""}</span><span class="track"><span class="fill" style="display:block;width:${v * 100}%;transition:none"></span></span><span class="val">${v > 0.9995 ? "100%" : pct(v, v < 0.0995 ? 1 : 0)}</span></div>`).join("")}</div>`;
    const sorted = [...L].sort((a, b) => b - a);
    $("#temp-read").textContent = `Last layer divided by ${(1 / scale).toFixed(scale > 0.5 ? 1 : 0)}. Gap between the two best scores: ${((sorted[0] - sorted[1]) * scale).toFixed(1)}. The network is ${win === truth ? "right" : "wrong"} about this image, and ${p[win] > 0.999 ? "completely certain" : pct(p[win], 0) + " confident"}.`;
  };
  const n = D.net.labels.length;
  $("#temp-prev").addEventListener("click", () => { k = (k + n - 1) % n; draw(); });
  $("#temp-next").addEventListener("click", () => { k = (k + 1) % n; draw(); });
  $("#temp-slider").addEventListener("input", draw);
  draw();
  return draw;
}

// ── READ ─────────────────────────────────────────────────────────────────────
function read() {
  const M = D.read.models, C = D.read.conv1_corr, condOrder = ["none", "standard", "strong"];
  const orders = { anon: M.map((_, i) => i), seed: M.map((_, i) => i).sort((a, b) => M[a].seed - M[b].seed || condOrder.indexOf(M[a].condition) - condOrder.indexOf(M[b].condition)) };
  let mode = "anon";
  const place = () => {
    const pos = []; orders[mode].forEach((m, p) => { pos[m] = p; });
    const mat = $("#read-matrix"), c = +mat.dataset.c, L = +mat.dataset.l;
    $$(".mv", mat).forEach((el) => {
      const i = el.dataset.i, j = el.dataset.j;
      el.style.transform = `translate(${j === undefined ? 0 : L + pos[+j] * c}px, ${i === undefined ? 0 : pos[+i] * c}px)`;
    });
    $$("text.rl", mat).forEach((t) => { t.textContent = mode === "seed" ? `${M[+t.dataset.i].id} · seed ${"ABC"[M[+t.dataset.i].seed]}` : `Model ${M[+t.dataset.i].id}`; });
    const rk = $("#read-rank"), step = +rk.dataset.step, l0 = +rk.dataset.l;
    if (!mat.dataset.c || !rk.dataset.step) return;
    $$(".mv", rk).forEach((el) => {
      const m = +el.dataset.m, gap = mode === "seed" ? Math.floor(pos[m] / 3) * 0.5 - 0.5 : 0;
      el.style.transform = `translate(${l0 + (pos[m] + gap + 0.5) * step}px, 0px)`;
      const dot = $("circle", el);
      if (dot) dot.setAttribute("fill", mode === "seed" ? COND[M[m].condition][1] : "var(--muted)");
    });
    const lg = $("#read-legend"); if (lg) lg.style.opacity = mode === "seed" ? 1 : 0;
  };
  fit($("#read-matrix"), (w) => {
    const L = w < 380 ? 82 : 100, c = (w - L) / 9, g = Math.max(1, c * 0.06);
    let s = `<svg viewBox="0 0 ${w} ${c * 9 + 4}">`;
    for (let i = 0; i < 9; i++) {
      s += `<text class="mv rl" data-i="${i}" x="${L - 8}" y="${c / 2 + 4}" text-anchor="end"></text>`;
      for (let j = 0; j < 9; j++) {
        const r = Math.max(0, C[i][j]);
        s += `<g class="mv" data-i="${i}" data-j="${j}"><rect width="${c - g}" height="${c - g}" rx="${c * 0.1}" fill="var(--bg)"/><rect width="${c - g}" height="${c - g}" rx="${c * 0.1}" fill="var(--blue)" fill-opacity="${r}" data-tip="Model ${M[i].id} and Model ${M[j].id}\ncorrelation ${C[i][j].toFixed(2)}"/></g>`;
      }
    }
    const mat = $("#read-matrix"); mat.innerHTML = s + "</svg>"; mat.dataset.c = c; mat.dataset.l = L;
    place();
  });
  fit($("#read-rank"), (w) => {
    const l = 40, rt = 8, t = 34, b = 30, h = 290, lo = 9.1, hi = 9.85, step = (w - l - rt) / 10;
    const y = (v) => t + (1 - (v - lo) / (hi - lo)) * (h - t - b);
    let s = `<svg viewBox="0 0 ${w} ${h}">`;
    for (let v = 9.2; v < hi; v += 0.2) s += `<line x1="${l}" x2="${w - rt}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${l - 6}" y="${y(v) + 4}" text-anchor="end" class="faint">${v.toFixed(1)}</text>`;
    M.forEach((m, i) => {
      s += `<g class="mv" data-m="${i}"><circle cx="0" cy="${y(m.head_effective_rank)}" r="7" fill="var(--muted)" stroke="var(--surface)" stroke-width="2" data-tip="Model ${m.id}\neffective rank ${m.head_effective_rank.toFixed(2)} of 10"/><text x="0" y="${h - 10}" text-anchor="middle">${m.id}</text></g>`;
    });
    const rk = $("#read-rank");
    rk.innerHTML = s + "</svg>" + `<div class="legend" id="read-legend" style="margin:6px 0 0;transition:opacity .5s">${condOrder.map((k) => `<span><i style="background:${COND[k][1]}"></i>${COND[k][0]}</span>`).join("")}</div>`;
    rk.dataset.step = step; rk.dataset.l = l;
    place();
  });
  seg($("#read-mode"), (v) => { mode = v; place(); });
  stepper($("#steps-read"), "How the model reasoned", [
    { time: "Step 1 · 19:22", title: "Measure what the prompt suggests, and more", text: "It computed about thirty statistics for each network. The cues the prompt pointed to, such as the symmetry of the filters, were weak and noisy. One signal was strong: three networks stood apart in their last layer (the right-hand chart)." },
    { time: "Step 2", title: "Doubt the clean signal", text: "Before trusting that cluster, it looked for a hidden cause that could explain it.", quote: "Models 02, 03, and 05 form a clean cluster in the classifier features, but I want to rule out a seed confound first, so I'm computing pairwise weight correlations." },
    { time: "Step 3 · 19:24", title: "Find the hidden structure", text: "The correlations showed three tight groups: networks that had started from the same random numbers and barely moved their first layer during training. Each group had to contain exactly one network of each kind, so one puzzle with 1,680 possible answers became three small ones. Press “Grouped the way it found” in the figure." },
    { time: "Step 4", title: "Order each group with a physical argument", text: "Strong augmentation scrambles colours, so a network trained with it should rely less on colour and on fewer features overall. In every group, the same network was the odd one out on both counts.", quote: "The correlation matrix shows three seed groups, each with one outlier model (02, 03, 05), and the features rank the models monotonically." },
    { time: "Step 5 · 19:26", title: "Commit, then look at the key", text: "It wrote down its answer with a confidence for each part: 95% for the three strongly augmented networks, 80% for the rest. Only then was the answer key opened. All nine were right, and so were the three groups." },
  ]);
}

// ── EDIT ─────────────────────────────────────────────────────────────────────
function edit() {
  const E = D.edit; let mode = "original";
  const setM = matrix($("#edit-matrix"));
  const std = fit($("#edit-std"), (w) => {
    const h = 70, v = E.row_std[mode], mx = 0.19, bw = w / v.length;
    let s = `<svg viewBox="0 0 ${w} ${h + 16}">`;
    v.forEach((x, i) => { s += `<rect x="${i * bw}" y="${h - (x / mx) * h}" width="${Math.max(0.6, bw - 0.4)}" height="${Math.max(1, (x / mx) * h)}" fill="var(--${x > 3 * E.healthy_sd ? "orange" : "axis"})"/>`; });
    s += `<text x="0" y="${h + 13}" class="faint">filter 1</text><text x="${w}" y="${h + 13}" text-anchor="end" class="faint">filter ${v.length}</text>`;
    $("#edit-std").innerHTML = s + "</svg>";
  });
  const draw = () => {
    const B = E.behaviour[mode];
    $("#edit-img").src = `data/edit_${mode}.png`;
    $("#edit-acc").textContent = pct(B.accuracy);
    $("#edit-acc").style.color = mode === "damaged" ? "var(--orange)" : "var(--blue)";
    $("#edit-samples").innerHTML = B.sample_pred.map((p, k) => `<div class="thumb ${p === D.net.labels[k] ? "" : "wrong"}" style="${thumbStyle(k, "rgb")}" data-tip="True class: ${NAMES[D.net.labels[k]]}\nNetwork says: ${NAMES[p]} (${pct(B.sample_conf[k], 0)} sure)"></div>`).join("");
    const ok = B.sample_pred.filter((p, k) => p === D.net.labels[k]).length;
    $("#edit-count").textContent = `${ok} of ${B.sample_pred.length} correct. Tap an image for details.`;
    setM(B.confusion); std();
  };
  seg($("#edit-mode"), (v) => { mode = v; draw(); });
  draw();
  const M = E.methods.A.methods;
  const rows = [["Original network, for reference", E.original_accuracy, "axis"], ["Damaged, no repair", M.corrupted, "orange"],
    ["Low-rank cleaning (suggested in the prompt)", M.global_svd, "axis"], ["Clip the largest weights (suggested)", M.clip_range, "axis"],
    ["Switch the damaged filters off", M.zero_rows, "axis"], ["Rescale them to a healthy size (suggested)", M.norm_match, "axis"],
    ["Replace with the average healthy filter · the model's repair", M.submitted, "blue"]];
  $("#methods-chart").innerHTML = `<div class="bars" style="font-size:13.5px;gap:6px">${rows.map(([n, v, col]) => `<div class="row methods-row" data-tip="${n}: ${pct(v, 2)}">
    <span class="name" style="white-space:normal;${col === "blue" ? "font-weight:700;color:var(--ink)" : ""}">${n}</span><span class="track" style="height:18px"><span class="fill" style="display:block;width:${v * 100}%;background:var(--${col})"></span></span><span class="val">${pct(v)}</span></div>`).join("")}</div>`;
  stepper($("#steps-edit"), "How the model reasoned", [
    { time: "Step 1 · 19:29", title: "Find the damage", text: "It measured the spread of the weights in each filter. Healthy filters sat between 0.025 and 0.040, and 32 filters sat near 0.17. A simple rule, more than three times the median, picks out exactly those 32 (the orange bars in the figure).", quote: "Next I'll load only the corrupted checkpoint and run weight-only diagnostics on the damaged layer before designing the repair." },
    { time: "Step 2", title: "Measure how bad it is", text: "Comparing damaged and healthy filters gave the noise level: the noise carries about 35 times the energy of the original weights. Each damaged filter is roughly 97% noise." },
    { time: "Step 3 · 19:30", title: "Test the suggested fixes before using them", text: "The prompt suggested low-rank cleaning, clipping and rescaling. Low-rank cleaning only works if healthy filters share a few common patterns. They do not: their spectrum is as flat as that of pure randomness. Clipping cannot help either, because no healthy weight is large enough to stand out from the noise." },
    { time: "Step 4 · 19:31", title: "Decide", text: "With no pattern to recover, the best estimate of each damaged filter is the average healthy one. The repair neutralises the broken filters with a sensible default instead of pretending to reconstruct them.", quote: "The clean filters form a flat Marchenko–Pastur bulk, so the 32 damaged ones can't be recovered from the tensor and I'll replace them with the mean clean filter." },
    { time: "Step 5", title: "Predict, then score", text: "Before running the evaluation it predicted 88 to 94% accuracy, noting that some loss was unavoidable. The measured result was slightly above its range.", quote: "Stage 2 came in at 95.46% accuracy and 93.3% recovery, beating my predicted 88–94%." },
  ]);
}

// ── pilot WRITE ──────────────────────────────────────────────────────────────
function pilotWrite() {
  const P = D.pilot_write, el = $("#filters");
  el.innerHTML = P.filters.map((_, i) => `<canvas width="3" height="3" data-i="${i}" tabindex="0" aria-label="Filter ${i + 1}"></canvas>`).join("");
  $$("canvas", el).forEach((cv, i) => {
    const f = P.filters[i], mx = Math.max(...f.flat(2).map(Math.abs)), ctx = cv.getContext("2d"), img = ctx.createImageData(3, 3);
    for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
      for (let ch = 0; ch < 3; ch++) img.data[(y * 3 + x) * 4 + ch] = 127.5 + 127.5 * f[ch][y][x] / mx;
      img.data[(y * 3 + x) * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });
  const pick = (i) => {
    $$("canvas", el).forEach((c) => c.classList.toggle("on", +c.dataset.i === i));
    $("#filter-note").textContent = `Filter ${i + 1}. ${P.notes[i].text}`;
  };
  el.addEventListener("click", (e) => { if (e.target.dataset.i) pick(+e.target.dataset.i); });
  el.addEventListener("pointerover", (e) => { if (e.target.dataset.i) pick(+e.target.dataset.i); });
  pick(0);
  dotRows($("#pw-dots"), () => ({ rows: [["kaiming", "Random first layer (Kaiming)", "var(--slate)"], ["fable", "Hand-written first layer", "var(--blue)"]]
    .map(([k, label, color]) => ({ label, color, values: P.runs.filter((r) => r.init === k).sort((a, b) => a.seed - b.seed).map((r) => r.test) })) }));
}

// ── page chrome ──────────────────────────────────────────────────────────────
function bands() {
  const col = { B01: "#5b6ee1", B02: "#2a78d6", B03: "#2aa568", B04: "#d5352f", B05: "#b02a35", B06: "#8e2433", B07: "#70212f", B08: "#55202c", B8A: "#47202a", B09: "#7d848a", B10: "#959ba1", B11: "#aeb3b8", B12: "#c9cdd1" };
  $("#bands").innerHTML = [...D.write.bands].sort((p, q) => p.nm - q.nm).map((x) => `<div data-tip="${x.id}: ${x.name}\n${x.nm} nm, ${x.res} m pixels"><i style="background:${col[x.id]}"></i><span>${x.nm}</span></div>`).join("");
}
function theme() {
  const btn = $("#theme"), root = document.documentElement;
  const isDark = () => (root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  const label = () => { btn.textContent = isDark() ? "Light mode" : "Dark mode"; };
  btn.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* private window */ }
    label();
  });
  label();
}
function chrome() {
  theme();
  const bar = $("#progress");
  const onScroll = () => { bar.style.width = (100 * window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight)) + "%"; };
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  const links = $$("#nav .links a");
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main section").forEach((s) => io.observe(s));
}

async function main() {
  chrome();
  await Promise.all(["write", "spectra", "net", "init", "read", "edit", "pilot_write"].map(async (n) => {
    D[n] = await (await fetch(`data/${n}.json`)).json();
  }));
  const F = computeFacts(D);
  $$("[data-n]").forEach((el) => { if (F[el.dataset.n] !== undefined) el.textContent = F[el.dataset.n]; });
  D.net.rounds.forEach((r) => nets.push(prepare(r)));
  bands(); hero(); gallery(); arch();
  const redrawLive = live();
  rounds(); spectra(); recall(); init();
  const redrawTemp = temperature();
  read(); edit(); pilotWrite();
  const buf = await (await fetch("data/samples.bin")).arrayBuffer();
  const raw = new Uint16Array(buf), per = D.net.used_bands.length * 4096;
  samples = D.net.labels.map((_, k) => Float32Array.from(raw.subarray(k * per, (k + 1) * per), (v) => v / 10000));
  redrawLive(); redrawTemp();
  // for screenshots and testing: index.html?click=<selector>,<selector> presses those controls after loading
  const clicks = new URLSearchParams(location.search).get("click");
  if (clicks) clicks.split(",").forEach((sel) => $(sel)?.click());
}
main().catch((e) => { console.error(e); document.body.insertAdjacentHTML("afterbegin", `<p style="padding:12px;background:#fdf3ec;color:#0b0b0b">The interactive figures could not load (${esc(e.message)}). The text below is complete.</p>`); });
