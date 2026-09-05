// Board model in the solver frame (port of kicad_fdtd/geometry.py):
// x right, y UP, z = 0 at the top copper, origin at the board's bounding-box corner. mm.
export const PORT_W = 0.16;
export const DEFAULT_ER = 4.5, DEFAULT_TAND = 0.02, DEFAULT_TOTAL = 1.6, DEFAULT_CU = 0.035;
const PAIR_SUFFIXES = [["_P", "_N"], ["+", "-"], ["P", "N"], ["_DP", "_DM"], ["DP", "DM"], ["_H", "_L"], ["_p", "_n"]];

// ---------------------------------------------------------------- polygon utilities
export function polyArea(pts) {
  let a = 0;
  for (let i = 0, n = pts.length; i < n; i++) { const p = pts[i], q = pts[(i + 1) % n]; a += p[0] * q[1] - q[0] * p[1]; }
  return Math.abs(a) / 2;
}
export function polyBBox(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
  return [x0, y0, x1, y1];
}
export function pointInRing(pts, x, y) {          // even-odd ray cast
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
export function pointInPoly(P, x, y) {           // P = {outline, holes, bbox}
  const b = P.bbox;
  if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) return false;
  if (!pointInRing(P.outline, x, y)) return false;
  for (const h of P.holes) if (pointInRing(h, x, y)) return false;
  return true;
}
export function distPointSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
  let t = L2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
export function distPointPoly(P, x, y) {         // 0 inside, else distance to the boundary
  if (pointInPoly(P, x, y)) return 0;
  let d = Infinity;
  const rings = [P.outline, ...P.holes];
  for (const r of rings) for (let i = 0, n = r.length; i < n; i++) { const a = r[i], b = r[(i + 1) % n]; const dd = distPointSeg(x, y, a[0], a[1], b[0], b[1]); if (dd < d) d = dd; }
  return d;
}

// Scanline rasteriser: for each row y in rowYs, which columns x in colXs are inside the polygon
// (even-odd over outline + holes). Calls mark(i, j) for inside samples. O(rows * (edges + cols)).
export function scanPoly(P, colXs, rowYs, mark) {
  const b = P.bbox;
  const edges = [];
  for (const ring of [P.outline, ...P.holes]) for (let i = 0, n = ring.length; i < n; i++) { const a = ring[i], c = ring[(i + 1) % n]; if (a[1] !== c[1]) edges.push(a[1] < c[1] ? [a[0], a[1], c[0], c[1]] : [c[0], c[1], a[0], a[1]]); }
  edges.sort((e, f) => e[1] - f[1]);
  // column index range of the bbox
  let c0 = lowerBound(colXs, b[0]), c1 = upperBound(colXs, b[2]);
  if (c0 >= c1) return;
  const xs = new Float64Array(64 + edges.length);
  let j0 = lowerBound(rowYs, b[1]), j1 = upperBound(rowYs, b[3]);
  let start = 0;
  for (let j = j0; j < j1; j++) {
    const y = rowYs[j];
    let nx = 0;
    for (let e = start; e < edges.length; e++) {
      const E = edges[e];
      if (E[1] > y) break;
      if (E[3] <= y) { if (e === start) start++; continue; }    // half-open rule y0 <= y < y1
      xs[nx++] = E[0] + (y - E[1]) * (E[2] - E[0]) / (E[3] - E[1]);
    }
    if (nx < 2) continue;
    const cross = Array.prototype.slice.call(xs, 0, nx).sort((p, q) => p - q);
    for (let k = 0; k + 1 < cross.length; k += 2) {
      const xa = cross[k], xb = cross[k + 1];
      for (let i = lowerBound(colXs, xa, c0, c1); i < c1 && colXs[i] < xb; i++) mark(i, j);
    }
  }
}
export function lowerBound(arr, v, lo = 0, hi = arr.length) {   // first index with arr[i] >= v
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}
export function upperBound(arr, v, lo = 0, hi = arr.length) {   // first index with arr[i] > v
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; }
  return lo;
}

// ---------------------------------------------------------------- Board
export class Board {
  constructor(g, refNet = null) {
    this.g = g;
    this.layers = g.layers;
    [this.x0, this.y0, this.w, this.h] = g.bbox;
    this._buildStackup(g.stackup || []);
    const ol = (g.outline || []).length >= 3 ? this.xy(g.outline) : [[0, -this.h], [this.w, -this.h], [this.w, 0], [0, 0]];
    this.outline = { outline: ol, holes: [], bbox: polyBBox(ol) };
    this.pads = g.pads;
    this._polyCache = {};
    this.refNet = refNet || this.guessRefNet();
  }
  _buildStackup(stack) {
    const cu = stack.filter(s => s.type === "copper"), dl = stack.filter(s => s.type === "prepreg" || s.type === "core");
    const n = this.layers.length;
    const usable = cu.length === n && dl.length === n - 1 && dl.every(d => d.thickness_mm);
    let cuT, dlT;
    if (usable) {
      this.stackupSource = "file";
      cuT = cu.map(c => c.thickness_mm || DEFAULT_CU);
      dlT = dl.map(d => [d.thickness_mm, d.er || DEFAULT_ER, d.tand || DEFAULT_TAND]);
    } else {
      this.stackupSource = `default (no stackup in the file: ${DEFAULT_TOTAL} mm FR4, er ${DEFAULT_ER})`;
      cuT = new Array(n).fill(DEFAULT_CU);
      const t = (DEFAULT_TOTAL - n * DEFAULT_CU) / Math.max(1, n - 1);
      dlT = new Array(n - 1).fill([t, DEFAULT_ER, DEFAULT_TAND]);
    }
    this.cuThickness = {}; this.layers.forEach((l, i) => this.cuThickness[l] = cuT[i]);
    let z = 0; this.z = {}; this.diel = [];
    this.layers.forEach((layer, i) => {
      this.z[layer] = z;
      if (i < n - 1) { const [t, er, tand] = dlT[i]; this.diel.push([z, z - t, er, tand]); z -= t; }
    });
    this.zBot = z;
  }
  xy(pts) { return pts.map(p => [p[0] - this.x0, -(p[1] - this.y0)]); }
  pt(x, y) { return [x - this.x0, -(y - this.y0)]; }
  pad(ref, num) {
    const p = this.pads.find(p => p.ref === ref && p.num === String(num));
    if (!p) throw new Error(`no pad ${ref}.${num}`);
    return [p, this.pt(p.x, p.y)];
  }
  polys(layer) {
    if (!this._polyCache[layer]) {
      const out = [];
      for (const poly of (this.g.copper[layer] || [])) {
        const outline = this.xy(poly.outline);
        if (outline.length < 3) continue;
        const holes = poly.holes.map(h => this.xy(h));
        out.push({ net: poly.net, kind: poly.kind, outline, holes, bbox: polyBBox(outline), area: polyArea(outline) - holes.reduce((s, h) => s + polyArea(h), 0) });
      }
      this._polyCache[layer] = out;
    }
    return this._polyCache[layer];
  }
  vias() {
    if (!this._vias) this._vias = this.g.vias.map(v => { const [x, y] = this.pt(v.x, v.y); return { x, y, r: v.dia / 2, zt: this.z[v.top] ?? 0, zb: this.z[v.bottom] ?? this.zBot, net: v.net, kind: v.kind, top: v.top, bottom: v.bottom, drill: v.drill || 0 }; });
    return this._vias;
  }
  copperAt(layer, x, y) {
    const s = new Set();
    for (const P of this.polys(layer)) if (pointInPoly(P, x, y)) s.add(P.net);
    return s;
  }
  netBBox(nets, margin = 2.0) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const layer of this.layers) for (const P of this.polys(layer)) if (nets.includes(P.net)) { const b = P.bbox; x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]); }
    for (const v of this.vias()) if (nets.includes(v.net)) { x0 = Math.min(x0, v.x); y0 = Math.min(y0, v.y); x1 = Math.max(x1, v.x); y1 = Math.max(y1, v.y); }
    if (!isFinite(x0)) throw new Error("no copper on nets " + nets.join(", "));
    return [x0 - margin, y0 - margin, x1 + margin, y1 + margin];
  }
  zoneAreaByNet() {
    const area = {};
    for (const layer of this.layers) for (const P of this.polys(layer)) if (P.kind === "zone" && P.net) area[P.net] = (area[P.net] || 0) + P.area;
    return area;
  }
  guessRefNet() {
    const area = this.zoneAreaByNet();
    const keys = Object.keys(area);
    if (!keys.length) return (this.g.nets || []).includes("GND") ? "GND" : null;
    return keys.reduce((a, b) => area[a] >= area[b] ? a : b);
  }
  diffPairs() {
    const nets = new Set(this.g.nets || []);
    const out = new Set();
    for (const n of [...nets].sort()) for (const [sp, sn] of PAIR_SUFFIXES) if (n.endsWith(sp) && n.length > sp.length && nets.has(n.slice(0, -sp.length) + sn)) out.add(JSON.stringify([n, n.slice(0, -sp.length) + sn]));
    return [...out].map(s => JSON.parse(s)).sort();
  }
  // copper area per layer by sampling (union without a geometry library): fraction grid at `cell` mm
  copperAreaSampled(layer, cell = 0.1) {
    const nx = Math.max(1, Math.ceil(this.w / cell)), ny = Math.max(1, Math.ceil(this.h / cell));
    const xs = new Float64Array(nx), ys = new Float64Array(ny);
    for (let i = 0; i < nx; i++) xs[i] = (i + 0.5) * cell;
    for (let j = 0; j < ny; j++) ys[j] = -this.h + (j + 0.5) * cell;
    const hit = new Uint8Array(nx * ny);
    for (const P of this.polys(layer)) scanPoly(P, xs, ys, (i, j) => { hit[i * ny + j] = 1; });
    let n = 0; for (let k = 0; k < hit.length; k++) n += hit[k];
    return n * cell * cell;
  }
  stats() {
    const g = this.g;
    const boardArea = polyArea(this.outline.outline);
    const perLayer = this.layers.map(layer => {
      const ps = this.polys(layer);
      const kinds = {}; for (const P of ps) kinds[P.kind] = (kinds[P.kind] || 0) + 1;
      const cu = this.copperAreaSampled(layer);
      return { layer, z_mm: this.z[layer], polygons: ps.length, kinds, copper_mm2: cu, fill: boardArea ? cu / boardArea : 0 };
    });
    const vias = this.vias();
    const area = this.zoneAreaByNet();
    const refCands = Object.entries(area).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([net, a]) => ({ net, zone_mm2: a }));
    return {
      board: g.board, kicad: g.kicad, size_mm: [this.w, this.h], board_area_mm2: boardArea, outline_points: (g.outline || []).length,
      layers: this.layers, thickness_mm: -this.zBot + Object.values(this.cuThickness).reduce((a, b) => a + b, 0), stackup_source: this.stackupSource,
      dielectrics: this.diel.map(([zt, zb, er, tand]) => ({ from: zt, to: zb, thickness_mm: zt - zb, er, tand })),
      per_layer: perLayer, footprints: (g.footprints || []).length, pads: this.pads.length,
      pth: vias.filter(v => v.kind === "pth").length, vias: vias.filter(v => v.kind === "via").length,
      nets: (g.nets || []).length, ref_net: this.refNet, ref_candidates: refCands, diff_pairs: this.diffPairs(),
    };
  }
  portLayerOptions(ref, num) {
    const [p] = this.pad(ref, num);
    const idx = {}; this.layers.forEach((l, i) => idx[l] = i);
    const n = this.layers.length;
    const on = p.layers.filter(l => l in idx);
    if (!on.length) return [];
    const sides = [];
    if (on.includes(this.layers[0])) sides.push([this.layers[0], [...Array(n - 1).keys()].map(k => k + 1)]);
    if (on.includes(this.layers[n - 1]) && n > 1) sides.push([this.layers[n - 1], [...Array(n - 1).keys()].map(k => n - 2 - k)]);
    for (const l of on) { const i = idx[l]; if (i > 0 && i < n - 1) sides.push([l, [...Array(n).keys()].filter(j => j !== i).sort((a, b) => Math.abs(a - i) - Math.abs(b - i) || a - b)]); }
    const out = [];
    for (const [sig, planes] of sides) for (const j of planes) out.push([sig, this.layers[j]]);
    return out;
  }
  portSite(ref, num, tie = {}, maxShift = 3.5, w0 = PORT_W, refNet = null) {
    // the port face must sit entirely on the signal copper: try the standard face first, then
    // smaller faces so that thin traces (0.15 mm) still get a port
    for (const w of [w0, 0.1, 0.06]) { const s = this._portSite(ref, num, tie, maxShift, w, refNet); if (s) { s.w = w; return s; } }
    return null;
  }
  _portSite(ref, num, tie, maxShift, w, refNet) {
    const pnet = refNet || this.refNet;
    const [p, [x0, y0]] = this.pad(ref, num);
    const net = p.net;
    for (const [sig, plane] of this.portLayerOptions(ref, num)) {
      const ok = (x, y) => {
        for (const [dx, dy] of [[0, 0], [-w / 2, -w / 2], [w / 2, -w / 2], [-w / 2, w / 2], [w / 2, w / 2]]) {
          if (!this.copperAt(sig, x + dx, y + dy).has(net)) return false;
          const under = new Set([...this.copperAt(plane, x + dx, y + dy)].map(n => tie[n] || n));
          if (under.size !== 1 || !under.has(pnet)) return false;
        }
        return true;
      };
      const tracks = this.polys(sig).filter(P => P.kind === "track" && P.net === net);
      const cands = tracks.map(P => { const cx = (P.bbox[0] + P.bbox[2]) / 2, cy = (P.bbox[1] + P.bbox[3]) / 2; return [Math.hypot(cx - x0, cy - y0), cx, cy]; }).sort((a, b) => a[0] - b[0]);
      const dirs = cands.slice(0, 3).filter(c => c[0] > 1e-6).map(([d, cx, cy]) => [(cx - x0) / d, (cy - y0) / d]);
      for (let shift = 0; shift <= maxShift + 1e-9; shift += 0.1)
        for (const [dx, dy] of (dirs.length ? dirs : [[0, 0]])) { const x = x0 + dx * shift, y = y0 + dy * shift; if (ok(x, y)) return { x, y, sig, plane, ref: pnet, shift }; }
      for (let shift = 0.2; shift <= maxShift + 1e-9; shift += 0.1)
        for (let ang = 0; ang < 360; ang += 20) { const x = x0 + shift * Math.cos(ang * Math.PI / 180), y = y0 + shift * Math.sin(ang * Math.PI / 180); if (ok(x, y)) return { x, y, sig, plane, ref: pnet, shift }; }
    }
    return null;
  }
}

export function formatStats(s) {
  const L = [`${s.board}  (KiCad ${s.kicad || "?"})`,
    `size ${s.size_mm[0].toFixed(1)} x ${s.size_mm[1].toFixed(1)} mm, outline area ${s.board_area_mm2.toFixed(0)} mm2, ${s.layers.length} copper layers, thickness ${s.thickness_mm.toFixed(2)} mm`,
    `stackup: ${s.stackup_source}`];
  for (const d of s.dielectrics) L.push(`   dielectric ${d.thickness_mm.toFixed(3)} mm  er ${d.er.toFixed(2)}  tand ${d.tand.toFixed(3)}`);
  L.push("layer      z mm    copper   fill  polygons");
  for (const p of s.per_layer) L.push(`${p.layer.padEnd(8)} ${p.z_mm.toFixed(3).padStart(8)} ${p.copper_mm2.toFixed(0).padStart(7)} mm2 ${(100 * p.fill).toFixed(0).padStart(4)}%  ${Object.entries(p.kinds).map(([k, v]) => `${v} ${k}`).join(", ")}`);
  L.push(`${s.footprints} footprints, ${s.pads} pads (${s.pth} plated holes), ${s.vias} vias, ${s.nets} nets`);
  L.push(`reference net: ${s.ref_net}   (candidates by zone area: ${s.ref_candidates.map(c => `${c.net} ${c.zone_mm2.toFixed(0)} mm2`).join(", ")})`);
  if (s.diff_pairs.length) L.push("differential pairs by name: " + s.diff_pairs.map(p => p.join("/")).join(", "));
  return L.join("\n");
}
