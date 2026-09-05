// Main thread: file handling, drawing, run control, plots. Heavy CPU work goes to worker.js;
// the FDTD engine (WebGPU or the JS reference) runs here.
import { Model, CpuEngine, GpuEngine, run, gaussPulse, sParams } from "./fdtd.js";
import { nf2ffSurface, cispr32B } from "./nf2ff.js";
import { pointInPoly, polyBBox } from "./geom.js";

const $ = id => document.getElementById(id);
const LAYER_COL = { "F.Cu": "#e0603a", "In1.Cu": "#4caf50", "In2.Cu": "#3f8cff", "In3.Cu": "#c77dff", "In4.Cu": "#ff9f43", "B.Cu": "#3fb8c8" };
const ID_COL = ["#ffd54a", "#ff6ec7", "#5ce1ff", "#a3ff5c", "#ff9f43", "#c77dff", "#ffffff", "#ff5c5c"];
let board = null, setup = null, vox = null, result = null, therm = null;
let snaps = [], curSnap = -1, playing = false, running = false, stopFlag = false;
let view = { s: 8, ox: 40, oy: 40 };
const cv = $("board"), ctx = cv.getContext("2d");
const layerOn = {};
let gpuOk = false;

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map(); let rpcId = 0;
const hooks = { snap: null, progress: null };
worker.onmessage = e => {
  const m = e.data;
  if (m.progress) { status(m.progress); return; }
  if (m.snap) { if (hooks.snap) hooks.snap(m.snap.n, m.snap.data); return; }
  if (m.runProgress) { if (hooks.progress) hooks.progress(m.runProgress); return; }
  const p = pending.get(m.id); if (!p) return; pending.delete(m.id);
  m.ok ? p.resolve(m) : p.reject(new Error(m.error));
};
const call = (msg) => new Promise((resolve, reject) => { const id = ++rpcId; pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }); });
function status(s) { $("status").textContent = s; }
function log(s) { const el = $("log"); el.textContent = (el.textContent + "\n" + s).split("\n").slice(-12).join("\n").trim(); }

// ---------------------------------------------------------------- loading
async function loadText(text, name, ref = null) {
  status("reading " + name + "…");
  try {
    const r = await call({ type: "load", text, ref });
    setBoard(r.view, name);
    status(`${name}: ${board.w.toFixed(1)} x ${board.h.toFixed(1)} mm, ${board.layers.length} copper layers, ${board.vias.length} vias, ${board.nets.length} nets (${r.ms.toFixed(0)} ms)`);
  } catch (e) { status("could not read the board: " + e.message); }
}
function setBoard(v, name) {
  board = v; board.name = name; setup = null; vox = null; result = null; therm = null; snaps = []; curSnap = -1;
  $("drop").style.display = "none";
  $("nets").innerHTML = board.nets.filter(Boolean).map(n => `<option>${esc(n)}</option>`).join("");
  $("stats").textContent = board.statsText.replace(/^browser/, name);
  const cands = board.stats.ref_candidates.map(c => c.net); if (board.refNet && !cands.includes(board.refNet)) cands.unshift(board.refNet);
  $("refSel").innerHTML = cands.map(n => `<option ${n === board.refNet ? "selected" : ""}>${esc(n)}</option>`).join("") + board.nets.filter(n => n && !cands.includes(n)).map(n => `<option>${esc(n)}</option>`).join("");
  $("pairSel").innerHTML = '<option value="">detected pairs…</option>' + board.stats.diff_pairs.map(p => `<option value="${esc(p[0])}|${esc(p[1])}">${esc(p[0])}  /  ${esc(p[1])}</option>`).join("");
  const bar = $("layerBar"); [...bar.querySelectorAll(".lyr")].forEach(e => e.remove());
  board.layers.forEach(L => {
    layerOn[L.name] = layerOn[L.name] ?? true;
    const lab = document.createElement("label"); lab.className = "lyr";
    lab.innerHTML = `<input type="checkbox" ${layerOn[L.name] ? "checked" : ""}><span class="sw" style="background:${LAYER_COL[L.name] || "#aaa"}"></span>${L.name}`;
    lab.querySelector("input").onchange = e => { layerOn[L.name] = e.target.checked; draw(); };
    bar.insertBefore(lab, $("showMesh").parentElement);
  });
  $("edgeLayer").innerHTML = board.layers.map(L => `<option>${L.name}</option>`).join("");
  const fps = board.footprints.filter(f => /^[UQDL]/.test(f.ref));
  $("fpHint").textContent = fps.length ? "e.g. " + fps.slice(0, 10).map(f => `${f.ref} (${f.value})`).join(", ") : "";
  $("setupBtn").disabled = false; $("thermBtn").disabled = false; $("runBtn").disabled = true;
  $("setupRep").textContent = ""; $("resRep").textContent = ""; $("thermRep").textContent = ""; $("showTherm").checked = false;
  fit();
}
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
$("openBtn").onclick = () => $("file").click();
$("file").onchange = async e => { const f = e.target.files[0]; if (f) loadText(await f.text(), f.name); };
// examples with presets (docs/examples/index.json): loading one fills every panel
let EXAMPLES = [];
fetch("examples/index.json").then(r => r.json()).then(list => { EXAMPLES = list; $("exampleSel").innerHTML = '<option value="">Examples…</option>' + list.map(e => `<option value="${e.id}">${esc(e.title)}</option>`).join(""); const want = new URLSearchParams(location.search).get("example"); if (want) loadExample(want); }).catch(() => {});
$("exampleSel").onchange = e => { if (e.target.value) loadExample(e.target.value); };
async function loadExample(id) {
  const ex = EXAMPLES.find(x => x.id === id); if (!ex) return;
  const r = await fetch(ex.file); await loadText(await r.text(), ex.file.split("/").pop());
  if (!board) return;
  [...$("nets").options].forEach(o => o.selected = ex.nets.includes(o.text));
  $("ports").value = ex.ports; $("tie").value = ex.tie || "";
  $("res").value = ex.res; $("base").value = ex.base; $("fmax").value = ex.fmax; $("tmax").value = ex.tmax;
  $("emcOn").checked = !!ex.emc;
  if (ex.emc) { $("emcClk").value = ex.emc.clk; $("emcA").value = ex.emc.A; $("emcTr").value = ex.emc.tr; $("emcDuty").value = ex.emc.duty; $("emcGap").value = ex.emc.gap; $("emcNf").value = ex.emc.nf; }
  $("powers").value = ex.powers || ""; $("tcell").value = ex.tcell || 0.25;
  $("dcNet").value = ex.dc?.net || ""; $("dcFrom").value = ex.dc?.from || ""; $("dcTo").value = ex.dc?.to || ""; $("dcI").value = ex.dc?.current || 0;
  status(`${ex.title}: nets, ports and settings filled in; press Set up and check`); draw();
}
document.addEventListener("dragover", e => { e.preventDefault(); });
document.addEventListener("drop", async e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadText(await f.text(), f.name); });
$("refSel").onchange = async e => { if (!board) return; const r = await call({ type: "ref", ref: e.target.value }); setBoard(r.view, board.name); };
$("netFilter").oninput = e => { const f = e.target.value.toLowerCase(); [...$("nets").options].forEach(o => o.hidden = f && !o.text.toLowerCase().includes(f)); };
$("nets").onchange = draw;
$("pairSel").onchange = e => { if (!e.target.value) return; const [p, n] = e.target.value.split("|"); [...$("nets").options].forEach(o => o.selected = (o.text === p || o.text === n)); $("ports").value = ""; draw(); };
$("ports").oninput = draw;
const selectedNets = () => [...$("nets").selectedOptions].map(o => o.text);

// ---------------------------------------------------------------- view
function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; draw(); }
window.addEventListener("resize", resize);
function fit() {
  if (!board) return;
  const r = cv.getBoundingClientRect();
  view.s = Math.min((r.width - 40) / board.w, (r.height - 40) / board.h);
  view.ox = (r.width - view.s * board.w) / 2; view.oy = (r.height - view.s * board.h) / 2;
  draw();
}
function zoomWindow() {
  if (!setup) return;
  const [x0, y0, x1, y1] = setup.window; const r = cv.getBoundingClientRect();
  view.s = Math.min(r.width / (x1 - x0), r.height / (y1 - y0)) * 0.95;
  view.ox = r.width / 2 - view.s * (x0 + x1) / 2; view.oy = r.height / 2 + view.s * (y0 + y1) / 2; draw();
}
$("fitBtn").onclick = fit; $("winBtn").onclick = zoomWindow;
const W2S = (x, y) => [view.ox + view.s * x, view.oy - view.s * y];
const S2W = (X, Y) => [(X - view.ox) / view.s, (view.oy - Y) / view.s];
cv.addEventListener("wheel", e => { e.preventDefault(); const r = cv.getBoundingClientRect(); const [wx, wy] = S2W(e.clientX - r.left, e.clientY - r.top);
  view.s *= e.deltaY < 0 ? 1.15 : 1 / 1.15; view.ox = e.clientX - r.left - view.s * wx; view.oy = e.clientY - r.top + view.s * wy; draw(); }, { passive: false });
let drag = null;
cv.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy, moved: false }; });
window.addEventListener("mousemove", e => {
  if (drag) { view.ox = drag.ox + e.clientX - drag.x; view.oy = drag.oy + e.clientY - drag.y; if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 3) drag.moved = true; draw(); return; }
  hover(e);
});
window.addEventListener("mouseup", e => { if (drag && !drag.moved) clickPad(e); drag = null; });
function nearestPad(e) {
  if (!board) return null;
  const r = cv.getBoundingClientRect(); const [wx, wy] = S2W(e.clientX - r.left, e.clientY - r.top);
  let best = null, bd = 0.6;
  for (const p of board.pads) { const d = Math.hypot(p.x - wx, p.y - wy); if (d < bd) { bd = d; best = p; } }
  return best;
}
function hover(e) {
  const p = nearestPad(e); const t = $("tip");
  if (!p) { t.style.display = "none"; return; }
  t.style.display = "block"; t.style.left = (e.clientX + 12) + "px"; t.style.top = (e.clientY + 12) + "px";
  t.textContent = `${p.ref}.${p.num}  ${p.net}  [${p.layers.join(",")}]`;
}
function clickPad(e) {
  const p = nearestPad(e); if (!p) return;
  if (!selectedNets().includes(p.net)) { status(`${p.ref}.${p.num} is on ${p.net}, not a selected net`); return; }
  const cur = $("ports").value.trim(); $("ports").value = (cur ? cur + " " : "") + `${p.ref}.${p.num}`; draw();
}
function poly(pts) { ctx.moveTo(...W2S(pts[0][0], pts[0][1])); for (let i = 1; i < pts.length; i++) ctx.lineTo(...W2S(pts[i][0], pts[i][1])); ctx.closePath(); }
function draw() {
  if (!board) return;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  const r = cv.getBoundingClientRect(); ctx.clearRect(0, 0, r.width, r.height);
  const sel = new Set(selectedNets());
  ctx.beginPath(); poly(board.outline); ctx.fillStyle = "#1b1f26"; ctx.fill(); ctx.strokeStyle = "#5a6473"; ctx.lineWidth = 1; ctx.stroke();
  for (let li = board.layers.length - 1; li >= 0; li--) {
    const L = board.layers[li]; if (!layerOn[L.name]) continue;
    const col = LAYER_COL[L.name] || "#aaa";
    for (const P of L.polys) {
      const mine = sel.has(P.net);
      ctx.beginPath(); poly(P.ext); for (const h of P.holes) poly(h);
      ctx.fillStyle = col; ctx.globalAlpha = mine ? 0.9 : (P.kind === "zone" ? 0.18 : 0.45); ctx.fill("evenodd");
      if (mine) { ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); }
    }
  }
  ctx.globalAlpha = 1;
  for (const v of board.vias) { const [X, Y] = W2S(v.x, v.y); ctx.beginPath(); ctx.arc(X, Y, Math.max(1.5, v.r * view.s), 0, 7); ctx.fillStyle = sel.has(v.net) ? "#fff" : "#6b7480"; ctx.fill(); }
  if ($("showField").checked && curSnap >= 0 && snaps[curSnap] && setup) drawField(snaps[curSnap]);
  if ($("showTherm").checked && therm) drawTherm();
  if (setup) {
    const [x0, y0, x1, y1] = setup.window; const [X0, Y0] = W2S(x0, y1), [X1, Y1] = W2S(x1, y0);
    ctx.strokeStyle = "#e0b04a"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1; ctx.strokeRect(X0, Y0, X1 - X0, Y1 - Y0); ctx.setLineDash([]);
    if ($("showMesh").checked && view.s * 0.05 > 0.6) {
      ctx.strokeStyle = "#ffffff22"; ctx.lineWidth = 1; ctx.beginPath();
      for (const x of setup.lines.x) { const [X] = W2S(x, 0); if (X < 0 || X > r.width) continue; ctx.moveTo(X, Y0); ctx.lineTo(X, Y1); }
      for (const y of setup.lines.y) { const [, Y] = W2S(0, y); if (Y < 0 || Y > r.height) continue; ctx.moveTo(X0, Y); ctx.lineTo(X1, Y); }
      ctx.stroke();
    }
    if ($("showEdges").checked) drawEdges();
    if ($("showPorts").checked) for (const s of setup.sites) {
      const [X, Y] = W2S(s.x, s.y); ctx.strokeStyle = "#ffd54a"; ctx.lineWidth = 2; ctx.strokeRect(X - 5, Y - 5, 10, 10);
      ctx.fillStyle = "#ffd54a"; ctx.font = "11px system-ui"; ctx.fillText(`${s.name} (${s.sig}/${s.plane})`, X + 8, Y - 6);
    }
    for (const s of setup.shorts) { const [X, Y] = W2S(s[2], s[3]); ctx.strokeStyle = "#e5533d"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X, Y, 7, 0, 7); ctx.stroke(); }
  } else {
    for (const p of board.pads) if (sel.has(p.net)) { const [X, Y] = W2S(p.x, p.y); ctx.fillStyle = "#ffd54a"; ctx.fillRect(X - 3, Y - 3, 6, 6); }
  }
  for (const rp of $("ports").value.trim().split(/\s+/).filter(Boolean)) { const [ref, num] = rp.split("."); const p = board.pads.find(q => q.ref === ref && q.num === num); if (!p) continue; const [X, Y] = W2S(p.x, p.y); ctx.strokeStyle = "#ffd54a"; ctx.lineWidth = 1; ctx.strokeRect(X - 6, Y - 6, 12, 12); }
}
function drawEdges() {
  const A = setup.assign.find(a => a.layer === $("edgeLayer").value); if (!A) return;
  const xs = setup.lines.x, ys = setup.lines.y, nx = xs.length, ny = ys.length, r = cv.getBoundingClientRect(), byId = {};
  for (let i = 0; i < nx - 1; i++) { const [X0] = W2S(xs[i], 0), [X1] = W2S(xs[i + 1], 0); if (X1 < 0 || X0 > r.width) continue;
    for (let j = 0; j < ny; j++) { const id = A.ex[i * ny + j]; if (id < 0) continue; const [, Y] = W2S(0, ys[j]); if (Y < 0 || Y > r.height) continue; (byId[id] = byId[id] || []).push(X0, Y, X1, Y); } }
  for (let i = 0; i < nx; i++) { const [X] = W2S(xs[i], 0); if (X < 0 || X > r.width) continue;
    for (let j = 0; j < ny - 1; j++) { const id = A.ey[i * (ny - 1) + j]; if (id < 0) continue; const [, Y0] = W2S(0, ys[j]), [, Y1] = W2S(0, ys[j + 1]); if (Y0 < 0 || Y1 > r.height) continue; (byId[id] = byId[id] || []).push(X, Y0, X, Y1); } }
  ctx.lineWidth = 1.2;
  for (const id in byId) { ctx.strokeStyle = ID_COL[id % ID_COL.length]; ctx.beginPath(); const s = byId[id]; for (let q = 0; q < s.length; q += 4) { ctx.moveTo(s[q], s[q + 1]); ctx.lineTo(s[q + 2], s[q + 3]); } ctx.stroke(); }
  let X = 12, Y = r.height - 14; ctx.font = "11px system-ui";
  setup.names.forEach((n, id) => { ctx.fillStyle = ID_COL[id % ID_COL.length]; ctx.fillRect(X, Y - 8, 10, 10); ctx.fillStyle = "#d7dbe0"; ctx.fillText(n, X + 14, Y); X += 24 + ctx.measureText(n).width; });
}
function drawField(sn) {
  const xs = setup.lines.x, ys = setup.lines.y, nx = xs.length, ny = ys.length, d = sn.data;
  let mx = 0; for (let q = 0; q < d.length; q++) { const a = Math.abs(d[q]); if (a > mx) mx = a; }
  if (!(mx > 0)) return;
  const floor = parseFloat($("floor").value) || -40, levels = 32, buckets = Array.from({ length: 2 * levels }, () => []), r = cv.getBoundingClientRect();
  for (let i = 0; i < nx - 1; i++) { const [X0] = W2S(xs[i], 0), [X1] = W2S(xs[i + 1], 0); if (X1 < 0 || X0 > r.width) continue;
    for (let j = 0; j < ny - 1; j++) { const v = d[i * ny + j]; const db = 20 * Math.log10(Math.abs(v) / mx + 1e-12); if (db < floor) continue;
      const [, Y0] = W2S(0, ys[j + 1]), [, Y1] = W2S(0, ys[j]); if (Y1 < 0 || Y0 > r.height) continue;
      const lv = Math.min(levels - 1, Math.floor((db - floor) / -floor * levels)); buckets[(v > 0 ? levels : 0) + lv].push(X0, Y0, X1 - X0 + 0.5, Y1 - Y0 + 0.5); } }
  buckets.forEach((b, q) => { if (!b.length) return; const pos = q >= levels, lv = q % levels, a = 0.15 + 0.85 * lv / levels;
    ctx.fillStyle = pos ? `rgba(255,${Math.round(140 - 100 * lv / levels)},60,${a})` : `rgba(60,${Math.round(160 - 100 * lv / levels)},255,${a})`;
    ctx.beginPath(); for (let k = 0; k < b.length; k += 4) ctx.rect(b[k], b[k + 1], b[k + 2], b[k + 3]); ctx.fill(); });
  ctx.fillStyle = "#d7dbe0"; ctx.font = "11px system-ui"; ctx.fillText(`Ez at z ${sn.z.toFixed(3)} mm, t ${(sn.t * 1e9).toFixed(2)} ns, step ${sn.n}, ${floor} dB floor`, 12, 16);
}
function heat(u) { const r = Math.min(1, u * 2.2), g = Math.max(0, Math.min(1, u * 2.2 - 0.8)), b = Math.max(0, Math.min(1, u * 4 - 3.2)); return `rgb(${Math.round(255 * r)},${Math.round(255 * g)},${Math.round(255 * b)})`; }
function drawTherm() {
  const kind = $("thermKind").value, layer = $("thermLayer").value;
  const src = kind === "T" ? therm.T : (therm.dc ? therm.dc[kind] : null); if (!src || !src[layer]) return;
  const A = src[layer], xs = therm.x, ys = therm.y, nx = xs.length - 1, ny = ys.length - 1;
  let lo = Infinity, hi = -Infinity;
  for (let q = 0; q < A.length; q++) { const v = A[q]; if (kind === "V" && v < 0) continue; if (kind === "J" && v <= 0) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!(hi > lo)) return;
  const r = cv.getBoundingClientRect(), levels = 24, buckets = Array.from({ length: levels }, () => []);
  for (let i = 0; i < nx; i++) { const [X0] = W2S(xs[i], 0), [X1] = W2S(xs[i + 1], 0); if (X1 < 0 || X0 > r.width) continue;
    for (let j = 0; j < ny; j++) { const v = A[i * ny + j]; if (kind !== "T" && v <= (kind === "V" ? -0.5 : 0)) continue;
      const [, Y0] = W2S(0, ys[j + 1]), [, Y1] = W2S(0, ys[j]); if (Y1 < 0 || Y0 > r.height) continue;
      const lv = Math.min(levels - 1, Math.floor((v - lo) / (hi - lo) * levels)); buckets[lv].push(X0, Y0, X1 - X0 + 0.5, Y1 - Y0 + 0.5); } }
  ctx.globalAlpha = 0.75;
  buckets.forEach((b, q) => { if (!b.length) return; ctx.fillStyle = heat((q + 0.5) / levels); ctx.beginPath(); for (let k = 0; k < b.length; k += 4) ctx.rect(b[k], b[k + 1], b[k + 2], b[k + 3]); ctx.fill(); });
  ctx.globalAlpha = 1;
  const bx = r.width - 28, by = 40, bh = 160;
  for (let q = 0; q < bh; q++) { ctx.fillStyle = heat(1 - q / bh); ctx.fillRect(bx, by + q, 14, 1); }
  ctx.fillStyle = "#d7dbe0"; ctx.font = "11px system-ui"; ctx.textAlign = "right";
  const unit = kind === "T" ? " °C" : kind === "J" ? " A/mm²" : " V";
  ctx.fillText(hi.toFixed(kind === "V" ? 3 : 1) + unit, bx - 4, by + 10); ctx.fillText(lo.toFixed(kind === "V" ? 3 : 1) + unit, bx - 4, by + bh); ctx.textAlign = "left";
  ctx.fillText(`${kind === "T" ? "temperature" : kind === "J" ? "current density" : "DC voltage"} on ${layer}`, 12, 32);
}
["showMesh", "showEdges", "showPorts", "showField", "edgeLayer", "floor", "showTherm", "thermKind", "thermLayer"].forEach(id => $(id).onchange = draw);

// ---------------------------------------------------------------- setup
$("setupBtn").onclick = async () => {
  const nets = selectedNets(); if (!nets.length) return status("select one or two nets");
  const ports = $("ports").value.trim().split(/\s+/).filter(Boolean); if (!ports.length) return status("give the port pads");
  $("setupBtn").disabled = true; status("meshing and checking…"); $("setupRep").textContent = "";
  try {
    const r = await call({ type: "setup", nets, ports, res: +$("res").value, base: +$("base").value, margin: +$("margin").value, fmax: +$("fmax").value * 1e9, z0: +$("z0").value, tie: $("tie").value.split(/\s+/).filter(Boolean), airGap: $("emcOn").checked ? +$("emcGap").value : 0, wholeBoard: $("emcOn").checked });
    setup = r.setup; vox = r.vox; result = null; snaps = []; curSnap = -1; $("snapSl").max = 0;
    const steps = Math.round(+$("tmax").value * 1e-9 / setup.dt);
    $("setupRep").innerHTML = esc(setup.log.slice(0, -1).join("\n")) + `\n${steps} steps for ${$("tmax").value} ns\n` + (setup.nShorts ? `<span class="bad">MERGE CHECK: ${setup.nShorts} node(s) join the pair to another conductor (red rings)</span>` : `<span class="ok">MERGE CHECK: clean</span>`);
    $("runBtn").disabled = setup.nShorts > 0;
    status(setup.nShorts ? "shorted model: fix the geometry or the ports" : `ready to run on ${gpuOk ? "WebGPU" : "the CPU (slow: keep the mesh small)"}`);
    zoomWindow();
  } catch (e) { status(e.message); $("setupRep").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  $("setupBtn").disabled = false;
};

// ---------------------------------------------------------------- run
$("runBtn").onclick = async () => {
  if (!setup || running) return;
  running = true; stopFlag = false; $("runBtn").disabled = true; $("stopBtn").disabled = false; $("setupBtn").disabled = true; result = null; snaps = []; curSnap = -1; $("snapSl").max = 0;
  let engine = null;
  try {
    const pair = setup.nets.length === 2;
    const z = setup.lines.z, zmid = (setup.sites[0].z0 + setup.sites[0].z1) / 2;
    let kz = 0; for (let k = 0; k < z.length - 1; k++) if (Math.abs((z[k] + z[k + 1]) / 2 - zmid) < Math.abs((z[kz] + z[kz + 1]) / 2 - zmid)) kz = k;
    const snapZ = (z[kz] + z[kz + 1]) / 2;
    const tmax = +$("tmax").value * 1e-9;
    const onSnap = (n, data, dt) => { snaps.push({ n, t: n * dt, data, z: snapZ }); $("snapSl").max = snaps.length - 1; if (!playing) { $("snapSl").value = snaps.length - 1; showSnap(snaps.length - 1); } };
    const onProgress = p => { $("prog").style.width = (100 * p.step / p.nsteps) + "%"; $("runRep").textContent = `step ${p.step} / ${p.nsteps}  energy ${p.energyDb.toFixed(1)} dB  ${p.elapsed.toFixed(0)} s  ${(p.step * setup.cells / p.elapsed / 1e6).toFixed(0)} Mcell/s`; };
    let r, S, f = []; for (let q = 0; q < 300; q++) f.push(100e6 + (setup.fmax - 100e6) * q / 299);
    const emc = $("emcOn").checked ? { clk: +$("emcClk").value * 1e6, A: +$("emcA").value, tr: +$("emcTr").value * 1e-9, duty: +$("emcDuty").value, nf: Math.max(3, +$("emcNf").value), spacing: 0.5e-3, distance: 3.0 } : null;
    // a radiating structure keeps energy long after the pulse: the far field needs the full ring-down
    const endCrit = emc ? Math.min(+$("endc").value, 1e-6) : +$("endc").value;
    let tmaxRun = tmax;
    if (emc && tmax < 40e-9) { tmaxRun = 40e-9; log("emissions: t max raised to 40 ns so the ring-down is captured (the power balance is reported)"); }
    let emcResult = null;
    $("emcRep").textContent = ""; $("emcBox").hidden = true;
    if (gpuOk) {
      const erOfZ = zz => { for (const [zt, zb, er] of setup.diel) if (zb <= zz && zz <= zt) return er; return 1.0; };
      const ol = { outline: setup.outline, holes: [], bbox: polyBBox(setup.outline) };
      const M = new Model(setup.lines, vox, erOfZ, { insideXY: (x, y) => pointInPoly(ol, x, y) });
      const excOf = pair ? [1, 0, -1, 0] : [1, 0, 0, 0];
      setup.boxes.forEach(([s, e], i) => M.addPort(s[0], e[0], s[1], e[1], s[2], e[2], setup.z0, i < excOf.length ? excOf[i] : 0));
      engine = new GpuEngine(M); await engine.init();
      const exc = gaussPulse(setup.fmax / 2, setup.fmax / 2);
      const nsteps = Math.floor(tmaxRun / M.dt);
      let nf2ff = null;
      if (emc) { if (!setup.airGap) log("note: set up with the emissions box ticked to add the air gap the far field needs"); const surf = nf2ffSurface(M, { spacing: emc.spacing, inset: setup.gapInset }); nf2ff = { surf, nf: emc.nf, f0: 0.2e9, df: (setup.fmax - 0.2e9) / (emc.nf - 1) }; log(`far-field box: ${surf.npts} points, ${emc.nf} frequencies`); }
      log(`[${pair ? "odd" : "single"}] WebGPU engine, ${nsteps} steps for ${$("tmax").value} ns`);
      r = await run(M, engine, exc.f, nsteps, { end: endCrit, snapEvery: +$("snapEvery").value, snapK: kz, batch: 100, nf2ff, onSnap: (n, d) => onSnap(n, d, M.dt), onProgress, shouldStop: () => stopFlag });
      S = sParams(M.spectra(f), pair);
      if (nf2ff) {
        status("far-field transform…");
        const freqs = []; for (let i = 0; i < nf2ff.nf; i++) freqs.push(nf2ff.f0 + i * nf2ff.df);
        const spec = M.spectra(freqs).map(s => ({ V: Array.from(s.V), I: Array.from(s.I), R: s.R }));
        const surf = { pts: null, pos: nf2ff.surf.pos, nrm: nf2ff.surf.nrm, area: nf2ff.surf.area, npts: nf2ff.surf.npts };
        emcResult = (await call({ type: "farfield", surf, acc: r.acc, nf: nf2ff.nf, f0: nf2ff.f0, df: nf2ff.df, spec, emc, fmax: setup.fmax, dt: M.dt })).emc;
      }
    } else {
      log(`[${pair ? "odd" : "single"}] CPU engine in the worker (slow)`);
      let dt = setup.dt;
      hooks.snap = (n, d) => onSnap(n, d, dt); hooks.progress = onProgress;
      const rr = await call({ type: "runCpu", nets: setup.nets, boxes: setup.boxes, z0: setup.z0, fmax: setup.fmax, tmax: tmaxRun, end: endCrit, snapEvery: +$("snapEvery").value, snapK: kz, emc });
      hooks.snap = hooks.progress = null;
      r = rr.result; S = rr.result; f = rr.result.f; emcResult = rr.result.emc || null;
    }
    if (emcResult) { $("emcRep").textContent = emcResult.report; $("emcBox").hidden = false; plotEmc(emcResult); log(emcResult.worst ? (emcResult.worst.margin >= 0 ? `emissions: passes CISPR 32 B by ${emcResult.worst.margin.toFixed(1)} dB` : `emissions: FAILS CISPR 32 B by ${(-emcResult.worst.margin).toFixed(1)} dB at ${(emcResult.worst.f / 1e6).toFixed(0)} MHz`) : "emissions: no harmonics in band"); }
    const speed = r.steps * setup.cells / r.wall / 1e6;
    log(`${r.steps} steps in ${r.wall.toFixed(0)} s (${speed.toFixed(0)} Mcell-updates/s)`);
    result = { f, S11_dB: S.S11_dB, S21_dB: S.S21_dB, Z: S.Z_re || S.Z, pair, steps: r.steps, wall: r.wall, speed };
    status(`done: ${r.steps} steps in ${r.wall.toFixed(0)} s (${speed.toFixed(0)} Mcell/s)`); plots();
  } catch (e) { status("run failed: " + e.message); log("error: " + e.message); hooks.snap = hooks.progress = null; }
  if (engine) engine.destroy();
  running = false; $("runBtn").disabled = false; $("stopBtn").disabled = true; $("setupBtn").disabled = false;
};
$("stopBtn").onclick = () => { stopFlag = true; worker.postMessage({ type: "stop" }); };
function showSnap(i) { if (i < 0 || !snaps[i]) return; curSnap = i; $("snapLbl").textContent = `snapshot ${i + 1}/${snaps.length}, ${(snaps[i].t * 1e9).toFixed(2)} ns`; draw(); }
$("snapSl").oninput = e => showSnap(+e.target.value);
$("playBtn").onclick = () => { playing = !playing; $("playBtn").textContent = playing ? "pause" : "play"; if (playing) step(); };
function step() { if (!playing) return; const i = (+$("snapSl").value + 1) % Math.max(1, snaps.length); $("snapSl").value = i; showSnap(i); setTimeout(step, 120); }

// ---------------------------------------------------------------- thermal
$("thermBtn").onclick = async () => {
  if (!board) return;
  const powers = {};
  $("powers").value.split(/\n|,/).map(s => s.trim()).filter(Boolean).forEach(s => { const [r, w] = s.split("="); if (r && w) powers[r.trim()] = +w; });
  $("thermBtn").disabled = true; status("solving thermal…"); $("thermRep").textContent = "";
  try {
    const r = await call({ type: "thermal", cell: +$("tcell").value, h: +$("hconv").value, tamb: +$("tamb").value, powers, dc: { net: $("dcNet").value.trim(), src: $("dcFrom").value.trim(), sink: $("dcTo").value.trim(), current: +$("dcI").value } });
    therm = r.thermal;
    $("thermLayer").innerHTML = therm.layers.map(l => `<option>${l}</option>`).join("");
    $("thermRep").textContent = therm.report; $("showTherm").checked = true; $("showField").checked = false;
    status(`thermal done in ${(therm.ms / 1000).toFixed(1)} s: max ${therm.Tmax.toFixed(1)} °C`); draw();
  } catch (e) { status(e.message); $("thermRep").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  $("thermBtn").disabled = false;
};

// ---------------------------------------------------------------- plots
function plot(canvas, series, yl, ylabel, xlabel) {
  const c = canvas.getContext("2d"); const r = canvas.getBoundingClientRect(); canvas.width = r.width * devicePixelRatio; canvas.height = r.height * devicePixelRatio;
  c.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); c.clearRect(0, 0, r.width, r.height);
  const m = { l: 42, r: 10, t: 10, b: 24 }, w = r.width - m.l - m.r, h = r.height - m.t - m.b;
  const xs = series[0].x, x0 = xs[0], x1 = xs[xs.length - 1];
  const X = x => m.l + (x - x0) / (x1 - x0) * w, Y = y => m.t + (yl[1] - y) / (yl[1] - yl[0]) * h;
  c.strokeStyle = "#2c323b"; c.fillStyle = "#8a93a0"; c.font = "10px system-ui"; c.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const y = yl[0] + (yl[1] - yl[0]) * g / 4; c.beginPath(); c.moveTo(m.l, Y(y)); c.lineTo(m.l + w, Y(y)); c.stroke(); c.fillText(y.toFixed(0), 4, Y(y) + 3); }
  for (let g = 0; g <= 5; g++) { const x = x0 + (x1 - x0) * g / 5; c.beginPath(); c.moveTo(X(x), m.t); c.lineTo(X(x), m.t + h); c.stroke(); c.fillText((x / 1e9).toFixed(1), X(x) - 8, r.height - 8); }
  c.fillText(ylabel, m.l + 4, m.t + 10); c.fillText(xlabel, m.l + w - 30, r.height - 8);
  let lx = m.l + 60;
  for (const s of series) { c.strokeStyle = s.col; c.lineWidth = 1.5; c.beginPath();
    s.x.forEach((x, i) => { const y = Math.max(yl[0], Math.min(yl[1], s.y[i])); i ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)); }); c.stroke();
    c.fillStyle = s.col; c.fillText(s.name, lx, m.t + 10); lx += c.measureText(s.name).width + 14; }
}
function plots() {
  if (!result) return;
  const f = result.f;
  plot($("plotS"), [{ x: f, y: result.S11_dB, col: "#ff6ec7", name: "S11" }, { x: f, y: result.S21_dB, col: "#5ce1ff", name: "S21" }], [-40, 0], "dB", "GHz");
  const Z = result.Z;
  plot($("plotZ"), [{ x: f, y: Z, col: "#ffd54a", name: result.pair ? "Zdiff (odd mode)" : "Zin" }], [0, 150], "ohm", "GHz");
  const sel = f.map((v, i) => i).filter(i => f[i] > 2e8 && f[i] < 2e9);
  const med = a => { const s = sel.map(i => a[i]).sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
  const at = (arr, fq) => arr[f.reduce((b, v, i) => Math.abs(v - fq) < Math.abs(f[b] - fq) ? i : b, 0)];
  $("resRep").textContent = `${result.pair ? "Zdiff" : "Zin"} median 0.2-2 GHz: ${med(Z).toFixed(0)} ohm\n` +
    [480e6, 1e9, 2e9].filter(fq => fq <= f[f.length - 1]).map(fq => `${(fq / 1e6).toFixed(0)} MHz: S21 ${at(result.S21_dB, fq).toFixed(2)} dB  S11 ${at(result.S11_dB, fq).toFixed(1)} dB  Z ${at(Z, fq).toFixed(0)} ohm`).join("\n");
}

function plotEmc(e) {
  const canvas = $("plotE"), c = canvas.getContext("2d"); const r = canvas.getBoundingClientRect(); canvas.width = r.width * devicePixelRatio; canvas.height = r.height * devicePixelRatio;
  c.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); c.clearRect(0, 0, r.width, r.height);
  const m = { l: 42, r: 10, t: 10, b: 24 }, w = r.width - m.l - m.r, h = r.height - m.t - m.b;
  const fmax = e.freqs[e.freqs.length - 1], x0 = 1e8, yl = [0, 90];
  const X = f => m.l + Math.log10(f / x0) / Math.log10(fmax / x0) * w, Y = y => m.t + (yl[1] - y) / (yl[1] - yl[0]) * h;
  c.strokeStyle = "#2c323b"; c.fillStyle = "#8a93a0"; c.font = "10px system-ui"; c.lineWidth = 1;
  for (let g = 0; g <= 3; g++) { const y = yl[0] + (yl[1] - yl[0]) * g / 3; c.beginPath(); c.moveTo(m.l, Y(y)); c.lineTo(m.l + w, Y(y)); c.stroke(); c.fillText(y.toFixed(0), 4, Y(y) + 3); }
  for (const f of [1e8, 2e8, 5e8, 1e9, 2e9, 3e9]) if (f <= fmax) { c.beginPath(); c.moveTo(X(f), m.t); c.lineTo(X(f), m.t + h); c.stroke(); c.fillText(f >= 1e9 ? (f / 1e9) + " G" : (f / 1e6) + " M", X(f) - 10, r.height - 8); }
  c.fillText("dBuV/m at 3 m", m.l + 4, m.t + 10);
  // limit line
  c.strokeStyle = "#e5533d"; c.lineWidth = 1.5; c.beginPath();
  let first = true; for (let f = x0; f <= fmax; f *= 1.02) { const y = Y(cispr32B(f)); first ? c.moveTo(X(f), y) : c.lineTo(X(f), y); first = false; } c.stroke();
  c.fillStyle = "#e5533d"; c.fillText("CISPR 32 B (peak)", m.l + w - 100, Y(cispr32B(fmax)) - 4);
  // 1 V broadband transfer
  c.strokeStyle = "#8a93a0"; c.setLineDash([3, 3]); c.beginPath();
  e.freqs.forEach((f, i) => { const y = Y(Math.max(yl[0], 20 * Math.log10(e.T[i] * 1e6 + 1e-30))); i ? c.lineTo(X(f), y) : c.moveTo(X(f), y); }); c.stroke(); c.setLineDash([]);
  c.fillStyle = "#8a93a0"; c.fillText("1 V sine per frequency", m.l + 4, m.t + 22);
  // harmonics
  for (const hm of e.harm) { const y = Y(Math.max(yl[0], Math.min(yl[1], hm.dBuV))); c.fillStyle = hm.margin >= 0 ? "#5fbf7a" : "#e5533d"; c.beginPath(); c.arc(X(hm.f), y, 4, 0, 7); c.fill(); }
}

// ---------------------------------------------------------------- boot
(async () => {
  resize();
  const a = await GpuEngine.available();
  gpuOk = !!a;
  const info = a && a.info ? [a.info.vendor, a.info.architecture, a.info.description].filter(Boolean).join(" ") : "";
  $("engine").textContent = gpuOk ? `WebGPU ${info}`.trim() : "no WebGPU: CPU engine (slow; use Chrome or Edge for the GPU)";
})();
window.kf = { get board() { return board; }, get setup() { return setup; }, get result() { return result; }, get therm() { return therm; }, loadText, call, zoomWindow, draw, showSnap };
