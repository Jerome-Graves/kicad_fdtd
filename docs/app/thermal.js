// Steady-state thermal and DC conduction on the board's exact copper (port of kicad_fdtd/thermal.py).
// Finite volumes on a whole-board grid; Jacobi-preconditioned conjugate gradient in plain JS.
import { scanPoly, pointInPoly } from "./geom.js";

export const K_CU = 385.0, K_FR4_XY = 0.8, K_FR4_Z = 0.3, K_AIR = 0.03, RHO_CU = 1.68e-8, VIA_WALL = 0.025;

export class Grid {
  constructor(bd, cell = 0.25, dzMax = 0.4) {
    this.bd = bd; this.cell = cell;
    const nx = Math.max(2, Math.ceil(bd.w / cell) + 1), ny = Math.max(2, Math.ceil(bd.h / cell) + 1);
    this.x = new Float64Array(nx); for (let i = 0; i < nx; i++) this.x[i] = bd.w * i / (nx - 1);
    this.y = new Float64Array(ny); for (let j = 0; j < ny; j++) this.y[j] = -bd.h + bd.h * j / (ny - 1);
    const zs = [0.0], kind = [];
    bd.layers.forEach((layer, i) => {
      const t = bd.cuThickness[layer] ?? 0.035;
      zs.push(zs[zs.length - 1] - t); kind.push(["cu", layer]);
      if (i < bd.layers.length - 1) { const [zt, zb] = bd.diel[i]; const span = zt - zb, n = Math.max(1, Math.ceil(span / dzMax)); for (let j = 0; j < n; j++) { zs.push(zs[zs.length - 1] - span / n); kind.push(["diel", i]); } }
    });
    this.z = Float64Array.from(zs); this.kind = kind;
    this.nx = nx - 1; this.ny = ny - 1; this.nz = zs.length - 1;
    this.dx = new Float64Array(this.nx); for (let i = 0; i < this.nx; i++) this.dx[i] = this.x[i + 1] - this.x[i];
    this.dy = new Float64Array(this.ny); for (let j = 0; j < this.ny; j++) this.dy[j] = this.y[j + 1] - this.y[j];
    this.dz = new Float64Array(this.nz); for (let k = 0; k < this.nz; k++) this.dz[k] = this.z[k] - this.z[k + 1];
    this.cx = new Float64Array(this.nx); for (let i = 0; i < this.nx; i++) this.cx[i] = (this.x[i] + this.x[i + 1]) / 2;
    this.cy = new Float64Array(this.ny); for (let j = 0; j < this.ny; j++) this.cy[j] = (this.y[j] + this.y[j + 1]) / 2;
    this.kCu = {}; kind.forEach(([kd, layer], k) => { if (kd === "cu") this.kCu[layer] = k; });
    this._frac = {};
    this._buildMaterials();
  }
  idx(i, j, k) { return (i * this.ny + j) * this.nz + k; }
  // fraction of each cell covered by copper on a layer (all nets or one net), 3 x 3 samples
  copperFraction(layer, net = null) {
    const key = layer + "|" + (net ?? "*");
    if (!this._frac[key]) {
      const nx = this.nx, ny = this.ny;
      const sx = new Float64Array(nx * 3), sy = new Float64Array(ny * 3);
      const o = [-1 / 3, 0, 1 / 3];
      for (let i = 0; i < nx; i++) for (let a = 0; a < 3; a++) sx[i * 3 + a] = this.cx[i] + o[a] * this.dx[i];
      for (let j = 0; j < ny; j++) for (let b = 0; b < 3; b++) sy[j * 3 + b] = this.cy[j] + o[b] * this.dy[j];
      const hit = new Uint8Array(nx * 3 * ny * 3);
      for (const P of this.bd.polys(layer)) if (net === null || P.net === net) scanPoly(P, sx, sy, (a, b) => { hit[a * ny * 3 + b] = 1; });
      const f = new Float64Array(nx * ny);
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { let n = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) n += hit[(i * 3 + a) * ny * 3 + j * 3 + b]; f[i * ny + j] = n / 9; }
      this._frac[key] = f;
    }
    return this._frac[key];
  }
  insideOutline() {
    if (!this._in) {
      this._in = new Uint8Array(this.nx * this.ny);
      scanPoly(this.bd.outline, this.cx, this.cy, (i, j) => { this._in[i * this.ny + j] = 1; });
    }
    return this._in;
  }
  cellOf(x, y) {
    let i = 0; while (i < this.nx - 1 && this.x[i + 1] <= x) i++;
    let j = 0; while (j < this.ny - 1 && this.y[j + 1] <= y) j++;
    return [i, j];
  }
  viaCopperArea(net = null) {
    const A = new Float64Array(this.nx * this.ny * this.nz);
    for (const v of this.bd.vias()) {
      if (net !== null && v.net !== net) continue;
      if (v.x < 0 || v.x > this.bd.w || v.y < -this.bd.h || v.y > 0) continue;
      const [i, j] = this.cellOf(v.x, v.y);
      const ri = v.drill ? v.drill / 2 : Math.max(0, v.r - VIA_WALL), ro = ri + VIA_WALL;
      const area = Math.PI * (ro * ro - ri * ri);
      const kt = this.kCu[v.top] ?? 0, kb = this.kCu[v.bottom] ?? this.nz - 1;
      for (let k = Math.min(kt, kb); k <= Math.max(kt, kb); k++) A[this.idx(i, j, k)] += area;
    }
    return A;
  }
  _buildMaterials() {
    const n = this.nx * this.ny * this.nz;
    this.kx = new Float64Array(n).fill(K_FR4_XY); this.kz = new Float64Array(n).fill(K_FR4_Z);
    const inside = this.insideOutline();
    this.kind.forEach(([kd, layer], k) => {
      if (kd !== "cu") return;
      const f = this.copperFraction(layer);
      for (let i = 0; i < this.nx; i++) for (let j = 0; j < this.ny; j++) { const q = this.idx(i, j, k), ff = f[i * this.ny + j]; this.kx[q] = ff * K_CU + (1 - ff) * K_FR4_XY; this.kz[q] = ff * K_CU + (1 - ff) * K_FR4_Z; }
    });
    const A = this.viaCopperArea();
    for (let i = 0; i < this.nx; i++) for (let j = 0; j < this.ny; j++) { const cellA = this.dx[i] * this.dy[j]; for (let k = 0; k < this.nz; k++) { const q = this.idx(i, j, k); this.kz[q] += A[q] / cellA * K_CU; if (!inside[i * this.ny + j]) { this.kx[q] = K_AIR; this.kz[q] = K_AIR; } } }
  }
  padCells(ref, num = null, layer = null) {
    const out = [];
    for (const p of this.bd.pads) {
      if (p.ref !== ref || (num !== null && p.num !== String(num)) || !p.layers.length) continue;
      const L = this.bd.layers;
      const lay = layer || (p.layers.includes(L[0]) ? L[0] : (p.layers.includes(L[L.length - 1]) ? L[L.length - 1] : p.layers[0]));
      const [x, y] = this.bd.pt(p.x, p.y);
      const [i, j] = this.cellOf(x, y);
      out.push([i, j, this.kCu[lay]]);
    }
    return out;
  }
}

// links: conductances between neighbouring cells along the three axes. Returns {a, b, G} flat indices.
function links(g, kx, ky, kz, active = null) {
  const nx = g.nx, ny = g.ny, nz = g.nz;
  const A = [], B = [], G = [];
  const dx = g.dx, dy = g.dy, dz = g.dz;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    const q = g.idx(i, j, k);
    if (active && !active[q]) continue;
    // x neighbour
    if (i + 1 < nx) { const r = g.idx(i + 1, j, k); if (!active || active[r]) { const area = dy[j] * dz[k] * 1e-6, ka = kx[q], kb = kx[r]; if (ka > 0 && kb > 0) { const gg = area / (dx[i] * 1e-3 / (2 * ka) + dx[i + 1] * 1e-3 / (2 * kb)); if (gg > 0) { A.push(q); B.push(r); G.push(gg); } } } }
    if (j + 1 < ny) { const r = g.idx(i, j + 1, k); if (!active || active[r]) { const area = dx[i] * dz[k] * 1e-6, ka = ky[q], kb = ky[r]; if (ka > 0 && kb > 0) { const gg = area / (dy[j] * 1e-3 / (2 * ka) + dy[j + 1] * 1e-3 / (2 * kb)); if (gg > 0) { A.push(q); B.push(r); G.push(gg); } } } }
    if (k + 1 < nz) { const r = g.idx(i, j, k + 1); if (!active || active[r]) { const area = dx[i] * dy[j] * 1e-6, ka = kz[q], kb = kz[r]; if (ka > 0 && kb > 0) { const gg = area / (dz[k] * 1e-3 / (2 * ka) + dz[k + 1] * 1e-3 / (2 * kb)); if (gg > 0) { A.push(q); B.push(r); G.push(gg); } } } }
  }
  return { a: Int32Array.from(A), b: Int32Array.from(B), G: Float64Array.from(G) };
}

// CSR from links + diagonal; SPD system (diag - offdiag) x = rhs; Jacobi PCG
export function cgSolve(n, a, b, G, diag, rhs, { tol = 1e-9, maxIter = 20000, onIter = null } = {}) {
  const cnt = new Int32Array(n + 1);
  for (let q = 0; q < a.length; q++) { cnt[a[q] + 1]++; cnt[b[q] + 1]++; }
  for (let i = 0; i < n; i++) cnt[i + 1] += cnt[i];
  const col = new Int32Array(cnt[n]), val = new Float64Array(cnt[n]), fill = cnt.slice(0, n);
  for (let q = 0; q < a.length; q++) { col[fill[a[q]]] = b[q]; val[fill[a[q]]++] = -G[q]; col[fill[b[q]]] = a[q]; val[fill[b[q]]++] = -G[q]; }
  const matvec = (x, y) => { for (let i = 0; i < n; i++) { let s = diag[i] * x[i]; for (let p = cnt[i]; p < cnt[i + 1]; p++) s += val[p] * x[col[p]]; y[i] = s; } };
  const x = new Float64Array(n), r = Float64Array.from(rhs), z = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
  const Minv = new Float64Array(n); for (let i = 0; i < n; i++) Minv[i] = diag[i] > 0 ? 1 / diag[i] : 0;
  let rz = 0, r0 = 0;
  for (let i = 0; i < n; i++) { z[i] = Minv[i] * r[i]; p[i] = z[i]; rz += r[i] * z[i]; r0 += r[i] * r[i]; }
  r0 = Math.sqrt(r0); if (r0 === 0) return x;
  let it = 0;
  for (; it < maxIter; it++) {
    matvec(p, Ap);
    let pAp = 0; for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    const alpha = rz / pAp;
    let rr = 0;
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; rr += r[i] * r[i]; }
    if (Math.sqrt(rr) < tol * r0) break;
    let rz2 = 0; for (let i = 0; i < n; i++) { z[i] = Minv[i] * r[i]; rz2 += r[i] * z[i]; }
    const beta = rz2 / rz; rz = rz2;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    if (onIter && it % 50 === 0) onIter(it, Math.sqrt(rr) / r0);
  }
  if (it >= maxIter) throw new Error("CG did not converge");
  x.iterations = it;
  return x;
}

export function solveDC(g, net, src, sink, current, onIter = null) {
  const nx = g.nx, ny = g.ny, nz = g.nz, n = nx * ny * nz;
  const sig = 1 / RHO_CU;
  const sx = new Float64Array(n), sz = new Float64Array(n), active = new Uint8Array(n);
  g.kind.forEach(([kd, layer], k) => {
    if (kd !== "cu") return;
    const f = g.copperFraction(layer, net);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const q = g.idx(i, j, k), ff = f[i * ny + j]; sx[q] = sig * ff; if (ff > 1e-3) active[q] = 1; }
  });
  const A = g.viaCopperArea(net);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const cellA = g.dx[i] * g.dy[j]; for (let k = 0; k < nz; k++) { const q = g.idx(i, j, k); if (A[q] > 0) { sz[q] = sig * A[q] / cellA; active[q] = 1; } } }
  for (let q = 0; q < n; q++) if (!active[q]) { sx[q] = 0; sz[q] = 0; }
  const L = links(g, sx, sx, sz, active);
  const srcCells = g.padCells(...src).filter(c => active[g.idx(...c)]);
  const sinkCells = g.padCells(...sink).filter(c => active[g.idx(...c)]);
  if (!srcCells.length || !sinkCells.length) throw new Error(`source or sink pad has no copper of net ${net} under it at this cell size`);
  // connected component of the source (union-find)
  const parent = new Int32Array(n); for (let q = 0; q < n; q++) parent[q] = q;
  const find = q => { while (parent[q] !== q) { parent[q] = parent[parent[q]]; q = parent[q]; } return q; };
  for (let q = 0; q < L.a.length; q++) { const ra = find(L.a[q]), rb = find(L.b[q]); if (ra !== rb) parent[ra] = rb; }
  const comp = find(g.idx(...srcCells[0]));
  if (sinkCells.some(c => find(g.idx(...c)) !== comp)) throw new Error(`source and sink are not connected through copper of ${net} on this grid (try a smaller cell)`);
  const loc = new Int32Array(n).fill(-1); let m = 0;
  for (let q = 0; q < n; q++) if (active[q] && find(q) === comp) loc[q] = m++;
  const ra = [], rb = [], rG = [];
  for (let q = 0; q < L.a.length; q++) if (loc[L.a[q]] >= 0 && loc[L.b[q]] >= 0) { ra.push(loc[L.a[q]]); rb.push(loc[L.b[q]]); rG.push(L.G[q]); }
  const diag = new Float64Array(m);
  for (let q = 0; q < ra.length; q++) { diag[ra[q]] += rG[q]; diag[rb[q]] += rG[q]; }
  const rhs = new Float64Array(m);
  for (const c of srcCells) rhs[loc[g.idx(...c)]] += current / srcCells.length;
  for (const c of sinkCells) rhs[loc[g.idx(...c)]] -= current / sinkCells.length;
  const gnd = loc[g.idx(...sinkCells[0])];
  let dmax = 0; for (let i = 0; i < m; i++) dmax = Math.max(dmax, diag[i]);
  diag[gnd] += 1e3 * dmax;
  const xs = cgSolve(m, Int32Array.from(ra), Int32Array.from(rb), Float64Array.from(rG), diag, rhs, { onIter });
  const V = new Float64Array(n).fill(NaN), J = new Float64Array(n), joule = new Float64Array(n);
  for (let q = 0; q < n; q++) if (loc[q] >= 0) V[q] = xs[loc[q]] - xs[gnd];
  for (let q = 0; q < ra.length; q++) {
    const dv = xs[ra[q]] - xs[rb[q]], I = rG[q] * dv, P = 0.5 * rG[q] * dv * dv;
    const qa = L.a[q], qb = L.b[q];
    joule[qa] += P; joule[qb] += P;
    const ka = qa % nz, kb = qb % nz;
    if (ka === kb) {                                   // in-plane: J = I / (t f width)
      const ia = Math.floor(qa / nz / ny), ja = Math.floor(qa / nz) % ny, ib = Math.floor(qb / nz / ny);
      const f = Math.max(sx[qa], sx[qb]) / sig, t = g.dz[ka] * 1e-3;
      const width = (ia !== ib ? g.dy[ja] : g.dx[ia]) * 1e-3;
      if (f > 0) { const Jl = Math.abs(I) / (t * f * width) / 1e6; if (Jl > J[qa]) J[qa] = Jl; if (Jl > J[qb]) J[qb] = Jl; }
    }
  }
  let Vs = 0; for (const c of srcCells) Vs += V[g.idx(...c)]; Vs /= srcCells.length;
  let jw = 0, Jmax = 0; for (let q = 0; q < n; q++) { jw += joule[q]; if (J[q] > Jmax) Jmax = J[q]; }
  return { net, current, drop_V: Vs, R_ohm: current ? Vs / current : 0, V, J, joule, joule_W: jw, Jmax, cells: m, iterations: xs.iterations };
}

export function solveThermal(g, powers, { h = 10, tamb = 25, extraQ = null, onIter = null } = {}) {
  const nx = g.nx, ny = g.ny, nz = g.nz, n = nx * ny * nz;
  const L = links(g, g.kx, g.kx, g.kz);
  const diag = new Float64Array(n);
  for (let q = 0; q < L.a.length; q++) { diag[L.a[q]] += L.G[q]; diag[L.b[q]] += L.G[q]; }
  const q = new Float64Array(n);
  const placed = {};
  for (const [ref, P] of Object.entries(powers)) {
    const cells = g.padCells(ref);
    if (!cells.length) throw new Error("no pads for footprint " + ref);
    for (const c of cells) q[g.idx(...c)] += P / cells.length;
    placed[ref] = cells.length;
  }
  if (extraQ) for (let i = 0; i < n; i++) q[i] += extraQ[i];
  const rhs = new Float64Array(n);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const faceA = g.dx[i] * g.dy[j] * 1e-6;
    for (const k of [0, nz - 1]) { const id = g.idx(i, j, k); diag[id] += h * faceA; rhs[id] += h * faceA * tamb; }
  }
  for (let i = 0; i < n; i++) rhs[i] += q[i];
  const T = cgSolve(n, L.a, L.b, L.G, diag, rhs, { onIter });
  let Tmax = -Infinity, Ptot = 0; for (let i = 0; i < n; i++) { if (T[i] > Tmax) Tmax = T[i]; Ptot += q[i]; }
  return { T, q, h, tamb, P_total: Ptot, Tmax, placed, iterations: T.iterations };
}

export function report(g, th, dc = null) {
  const bd = g.bd, T = th.T;
  const L = [`thermal: ${th.P_total.toFixed(2)} W in, h ${th.h.toFixed(0)} W/m2K both faces, T_amb ${th.tamb.toFixed(0)} C, grid ${g.nx} x ${g.ny} x ${g.nz} cells of ${g.cell} mm (${th.iterations} CG iterations)`];
  let qm = 0; for (let q = 1; q < T.length; q++) if (T[q] > T[qm]) qm = q;
  const k = qm % g.nz, j = Math.floor(qm / g.nz) % g.ny, i = Math.floor(qm / g.nz / g.ny);
  L.push(`max ${T[qm].toFixed(1)} C (+${(T[qm] - th.tamb).toFixed(1)} K) at (${g.cx[i].toFixed(1)}, ${g.cy[j].toFixed(1)}) mm on ${g.kind[k][0] === "cu" ? g.kind[k][1] : "dielectric"}`);
  const inside = g.insideOutline();
  for (const [layer, kk] of Object.entries(g.kCu)) {
    let mx = -Infinity, s = 0, c = 0;
    for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) { const v = T[g.idx(i, j, kk)]; if (v > mx) mx = v; if (inside[i * g.ny + j]) { s += v; c++; } }
    L.push(`   ${layer.padEnd(7)} max ${mx.toFixed(1)} C  mean ${(s / c).toFixed(1)} C`);
  }
  const rows = [];
  for (const fp of (bd.g.footprints || [])) { const cells = g.padCells(fp.ref); if (cells.length) rows.push([Math.max(...cells.map(c => T[g.idx(...c)])), fp.ref, fp.value]); }
  rows.sort((a, b) => b[0] - a[0]);
  L.push("hottest footprints: " + rows.slice(0, 8).map(([t, r, v]) => `${r} (${v}) ${t.toFixed(1)} C`).join(", "));
  if (dc) L.push(`DC ${dc.net}: ${dc.current.toFixed(2)} A from source to sink, drop ${(1e3 * dc.drop_V).toFixed(1)} mV (${(1e3 * dc.R_ohm).toFixed(1)} mohm), Joule ${dc.joule_W.toFixed(3)} W, max current density ${dc.Jmax.toFixed(1)} A/mm2 over ${dc.cells} copper cells`);
  return L.join("\n");
}

// per-layer 2D slices for the view
export function layerSlices(g, arr, nanTo = null) {
  const out = {};
  for (const [layer, k] of Object.entries(g.kCu)) {
    const s = new Float32Array(g.nx * g.ny);
    for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) { const v = arr[g.idx(i, j, k)]; s[i * g.ny + j] = (nanTo !== null && Number.isNaN(v)) ? nanTo : v; }
    out[layer] = s;
  }
  return out;
}
