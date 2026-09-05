// Yee FDTD on a non-uniform grid with CPML and lumped ports (port of kicad_fdtd/solver.py +
// kernels.py). Two engines with one interface: WebGPU (six fused compute kernels, CPML and
// ports inside them, port V/I read on the device) and a plain-JS reference for machines
// without WebGPU (same arithmetic, ~100x slower).
import { accumulateCpu } from "./nf2ff.js";
export const C0 = 299792458.0, EPS0 = 8.8541878128e-12, MU0 = 1.25663706212e-6, ETA0 = 376.730313668;

export function gaussPulse(f0, fc) {
  const tau = 0.5 / fc, t0 = 3.5 * tau;
  return { f: t => Math.exp(-(((t - t0) / tau) ** 2)) * Math.cos(2 * Math.PI * f0 * (t - t0)), t0, tau };
}

// ---------------------------------------------------------------- model (coefficients on the CPU)
export class Model {
  // erOfZ(z_mm): permittivity of the board's dielectric layers; insideXY(x_mm, y_mm): true inside the board
  // outline (outside it the layers are air, so a Huygens surface beyond the board sits in free space)
  constructor(lines, vox, erOfZ, { npml = 8, cfl = 0.99, insideXY = null } = {}) {
    const x = Float64Array.from(lines.x, v => v * 1e-3), y = Float64Array.from(lines.y, v => v * 1e-3), z = Float64Array.from(lines.z, v => v * 1e-3);
    this.x = x; this.y = y; this.z = z;
    const nx = x.length, ny = y.length, nz = z.length;
    this.nx = nx; this.ny = ny; this.nz = nz; this.npml = npml;
    const dx = diff(x), dy = diff(y), dz = diff(z);
    const ddx = dual(dx), ddy = dual(dy), ddz = dual(dz);
    this.dx = dx; this.dy = dy; this.dz = dz; this.ddx = ddx; this.ddy = ddy; this.ddz = ddz;
    this.dt = cfl / (C0 * Math.sqrt(1 / min(dx) ** 2 + 1 / min(dy) ** 2 + 1 / min(dz) ** 2));
    const erCell = new Float64Array(nz - 1); for (let k = 0; k < nz - 1; k++) erCell[k] = erOfZ((z[k] + z[k + 1]) / 2 * 1e3);
    this.erCell = erCell;
    const erPlane = new Float64Array(nz); for (let k = 0; k < nz; k++) erPlane[k] = (k < nz - 1 ? erCell[k] : 1) / 2 + (k > 0 ? erCell[k - 1] : 1) / 2;
    const cEplane = Float32Array.from(erPlane, e => this.dt / (EPS0 * e)), cEcell = Float32Array.from(erCell, e => this.dt / (EPS0 * e));
    const cAir = this.dt / EPS0;
    // in-plane mask: which (x, y) cells and nodes lie inside the board outline (all of them when no outline is given)
    const cellIn = new Uint8Array((nx - 1) * (ny - 1)).fill(1), nodeIn = new Uint8Array(nx * ny).fill(1);
    if (insideXY) {
      for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) cellIn[i * (ny - 1) + j] = insideXY((x[i] + x[i + 1]) / 2 * 1e3, (y[j] + y[j + 1]) / 2 * 1e3) ? 1 : 0;
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) nodeIn[i * ny + j] = insideXY(x[i] * 1e3, y[j] * 1e3) ? 1 : 0;
    }
    // an edge bordering the outline gets the average of its neighbouring cells' permittivity
    const exIn = (i, j) => (((j > 0 ? cellIn[i * (ny - 1) + j - 1] : 1) + (j < ny - 1 ? cellIn[i * (ny - 1) + j] : 1)) / 2);
    const eyIn = (i, j) => (((i > 0 ? cellIn[(i - 1) * (ny - 1) + j] : 1) + (i < nx - 1 ? cellIn[i * (ny - 1) + j] : 1)) / 2);
    const mix = (cIn, frac) => 1 / (frac / cIn + (1 - frac) / cAir);     // coefficient of the averaged permittivity (c = dt / eps)
    this.cEx = new Float32Array((nx - 1) * ny * nz); this.cEy = new Float32Array(nx * (ny - 1) * nz); this.cEz = new Float32Array(nx * ny * (nz - 1));
    for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) { const fr = exIn(i, j); for (let k = 0; k < nz; k++) { const q = (i * ny + j) * nz + k; this.cEx[q] = vox.ex[q] >= 0 ? 0 : (fr >= 1 ? cEplane[k] : mix(cEplane[k], fr)); } }
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) { const fr = eyIn(i, j); for (let k = 0; k < nz; k++) { const q = (i * (ny - 1) + j) * nz + k; this.cEy[q] = vox.ey[q] >= 0 ? 0 : (fr >= 1 ? cEplane[k] : mix(cEplane[k], fr)); } }
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const fr = nodeIn[i * ny + j]; for (let k = 0; k < nz - 1; k++) { const q = (i * ny + j) * (nz - 1) + k; this.cEz[q] = vox.ez[q] >= 0 ? 0 : (fr ? cEcell[k] : cAir); } }
    this.cH = this.dt / MU0;
    this.idx = Float32Array.from(dx, v => 1 / v); this.idy = Float32Array.from(dy, v => 1 / v); this.idz = Float32Array.from(dz, v => 1 / v);
    this.iddx = Float32Array.from(ddx, v => 1 / v); this.iddy = Float32Array.from(ddy, v => 1 / v); this.iddz = Float32Array.from(ddz, v => 1 / v);
    this._setupCpml(dx, dy, dz);
    this.ports = [];
  }
  _setupCpml(dx, dy, dz) {
    const n = this.npml, m = 3.0, amax = 0.05, dt = this.dt;
    this.pml = {};                         // (axis, kind) -> {b, c} concatenated lo (n) + hi reversed (n)
    if (n === 0) { for (const ax of "xyz") for (const kd of "he") this.pml[ax + kd] = { b: new Float32Array(0), c: new Float32Array(0) }; return; }
    for (const [axis, d] of [["x", dx], ["y", dy], ["z", dz]]) {
      const coef = (rho, smax) => {
        const b = new Float32Array(rho.length), c = new Float32Array(rho.length);
        rho.forEach((r, i) => { r = Math.max(0, Math.min(1, r)); const sig = smax * r ** m, a = amax * (1 - r); const bb = Math.exp(-(sig + a) * dt / EPS0); b[i] = bb; c[i] = sig + a > 0 ? sig * (bb - 1) / (sig + a + 1e-30) : 0; });
        return [b, c];
      };
      const parts = { h: [], e: [] };
      for (const side of ["lo", "hi"]) {
        const cells = side === "lo" ? d.slice(0, n) : d.slice(d.length - n);
        const smax = 0.8 * (m + 1) / (ETA0 * mean(cells));
        const rhoE = [...Array(n + 1).keys()].map(i => (n - i) / n), rhoH = [...Array(n).keys()].map(i => (n - i - 0.5) / n);
        let [be, ce] = coef(rhoE, smax); let [bh, ch] = coef(rhoH, smax);
        be = be.slice(1, n + 1); ce = ce.slice(1, n + 1);
        if (side === "hi") { be = be.reverse(); ce = ce.reverse(); bh = bh.reverse(); ch = ch.reverse(); }
        parts.h.push([bh, ch]); parts.e.push([be, ce]);
      }
      for (const kind of ["h", "e"]) this.pml[axis + kind] = { b: concat(parts[kind][0][0], parts[kind][1][0]), c: concat(parts[kind][0][1], parts[kind][1][1]) };
    }
  }
  addPort(x0, x1, y0, y1, z0, z1, R, excite = 0) {           // mm
    const xl = this.x, yl = this.y, zl = this.z;
    let ix = range(xl, x0 * 1e-3 - 1e-12, x1 * 1e-3 + 1e-12), jy = range(yl, y0 * 1e-3 - 1e-12, y1 * 1e-3 + 1e-12);
    const kz = []; for (let k = 0; k < zl.length; k++) if (zl[k] >= z0 * 1e-3 - 1e-12 && zl[k] < z1 * 1e-3 - 1e-12) kz.push(k);
    // on a coarse mesh the 0.16 mm face may hold no line: use the nearest one (a single Ez column)
    const nearest = (arr, v) => { let b = 0; for (let i = 1; i < arr.length; i++) if (Math.abs(arr[i] - v) < Math.abs(arr[b] - v)) b = i; return [b]; };
    if (!ix.length) ix = nearest(xl, (x0 + x1) / 2 * 1e-3);
    if (!jy.length) jy = nearest(yl, (y0 + y1) / 2 * 1e-3);
    if (!kz.length) throw new Error("port box contains no Ez edge between the two copper layers");
    if (ix[0] <= 0 || jy[0] <= 0 || ix[ix.length - 1] >= this.nx - 1 || jy[jy.length - 1] >= this.ny - 1) throw new Error("port touches the boundary");
    const L = zl[kz[kz.length - 1] + 1] - zl[kz[0]];
    let area = 0; for (const i of ix) for (const j of jy) area += this.ddx[i] * this.ddy[j];
    const sigma = L / (R * area);
    const kmid = kz[kz.length >> 1];
    const eps = EPS0 * this.erCell[kmid];
    const f = this.dt * sigma / (2 * eps), lb = 1 / (1 + f), src = this.dt * sigma / eps;
    const p = { nr: this.ports.length, R, excite, i0: ix[0], i1: ix[ix.length - 1], j0: jy[0], j1: jy[jy.length - 1], k0: kz[0], k1: kz[kz.length - 1],
      ic: ix[ix.length >> 1], jc: jy[jy.length >> 1], kmid, L, lb, f, srcScale: excite ? src * (-excite / L) : 0, V: null, I: null };
    this.ports.push(p);
    return p;
  }
  // port tables for the kernels
  portTables() {
    const NP = Math.max(1, this.ports.length);
    const pb = new Int32Array(NP * 6), pc = new Float32Array(NP * 3), pg = new Int32Array(NP * 9);
    this.ports.forEach((p, q) => { pb.set([p.i0, p.i1, p.j0, p.j1, p.k0, p.k1], q * 6); pc.set([p.lb, p.f, p.srcScale], q * 3); pg.set([p.ic, p.jc, p.k0, p.k1, p.kmid, p.i0, p.i1, p.j0, p.j1], q * 9); });
    return { pb, pc, pg, NP };
  }
  spectra(freqs) {
    const out = [];
    for (const p of this.ports) {
      const N = p.V.length, Vf = new Float64Array(2 * freqs.length), If = new Float64Array(2 * freqs.length);
      freqs.forEach((f, q) => {
        let vr = 0, vi = 0, ir = 0, ii = 0; const w = 2 * Math.PI * f * this.dt;
        for (let n = 0; n < N; n++) { const c = Math.cos(w * n), s = -Math.sin(w * n); vr += p.V[n] * c; vi += p.V[n] * s; ir += p.I[n] * c; ii += p.I[n] * s; }
        Vf[2 * q] = vr * this.dt; Vf[2 * q + 1] = vi * this.dt; If[2 * q] = ir * this.dt; If[2 * q + 1] = ii * this.dt;
      });
      out.push({ V: Vf, I: If, R: p.R });
    }
    return out;
  }
}
const diff = a => { const d = new Float64Array(a.length - 1); for (let i = 0; i < d.length; i++) d[i] = a[i + 1] - a[i]; return d; };
const dual = d => { const o = new Float64Array(d.length + 1); o[0] = d[0]; o[d.length] = d[d.length - 1]; for (let i = 1; i < d.length; i++) o[i] = (d[i - 1] + d[i]) / 2; return o; };
const min = a => { let m = Infinity; for (const v of a) if (v < m) m = v; return m; };
const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
const concat = (a, b) => { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
const range = (arr, lo, hi) => { const o = []; for (let i = 0; i < arr.length; i++) if (arr[i] >= lo && arr[i] <= hi) o.push(i); return o; };

// incident / reflected wave magnitudes per port at the given spectra, and the power the ports
// deliver into the structure (amplitude convention: P = |V|^2 / 2R)
export function portPower(spec) {
  const n = spec[0].V.length / 2, Pacc = new Float64Array(n), Vinc = new Float64Array(n);
  for (let q = 0; q < n; q++) {
    let p = 0;
    spec.forEach((s, i) => {
      const V = [s.V[2 * q], s.V[2 * q + 1]], I = [s.I[2 * q], s.I[2 * q + 1]], R = s.R;
      const inc2 = ((V[0] + R * I[0]) ** 2 + (V[1] + R * I[1]) ** 2) / 4, ref2 = ((V[0] - R * I[0]) ** 2 + (V[1] - R * I[1]) ** 2) / 4;
      if (i === 0) { Vinc[q] = Math.sqrt(inc2); p += inc2 / (2 * R); }
      p -= ref2 / (2 * R);
    });
    Pacc[q] = p;
  }
  return { Pacc, Vinc };
}

// complex helpers for S-parameters
export function sParams(spec, pair) {
  const div = (a, b) => { const d = b[0] * b[0] + b[1] * b[1] + 1e-60; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
  const n = spec[0].V.length / 2;
  const S11 = new Float64Array(n), S21 = new Float64Array(n), Z = new Float64Array(n);
  for (let q = 0; q < n; q++) {
    const V0 = [spec[0].V[2 * q], spec[0].V[2 * q + 1]], I0 = [spec[0].I[2 * q], spec[0].I[2 * q + 1]], R = spec[0].R;
    const inc = [(V0[0] + R * I0[0]) / 2, (V0[1] + R * I0[1]) / 2], ref = [(V0[0] - R * I0[0]) / 2, (V0[1] - R * I0[1]) / 2];
    const s11 = div(ref, inc); S11[q] = 20 * Math.log10(Math.hypot(s11[0], s11[1]) + 1e-12);
    if (spec.length > 1) { const V1 = [spec[1].V[2 * q], spec[1].V[2 * q + 1]], I1 = [spec[1].I[2 * q], spec[1].I[2 * q + 1]]; const ref1 = [(V1[0] - spec[1].R * I1[0]) / 2, (V1[1] - spec[1].R * I1[1]) / 2]; const s21 = div(ref1, inc); S21[q] = 20 * Math.log10(Math.hypot(s21[0], s21[1]) + 1e-12); }
    const z = div(V0, I0); Z[q] = (pair ? 2 : 1) * z[0];
  }
  return { S11_dB: S11, S21_dB: S21, Z_re: Z };
}

// ---------------------------------------------------------------- CPU reference engine
export class CpuEngine {
  constructor(M) {
    this.M = M;
    const { nx, ny, nz } = M;
    this.Ex = new Float32Array((nx - 1) * ny * nz); this.Ey = new Float32Array(nx * (ny - 1) * nz); this.Ez = new Float32Array(nx * ny * (nz - 1));
    this.Hx = new Float32Array(nx * (ny - 1) * (nz - 1)); this.Hy = new Float32Array((nx - 1) * ny * (nz - 1)); this.Hz = new Float32Array((nx - 1) * (ny - 1) * nz);
    const n2 = 2 * M.npml;
    this.psi = { Hx_y: new Float32Array(nx * n2 * (nz - 1)), Hx_z: new Float32Array(nx * (ny - 1) * n2), Hy_z: new Float32Array((nx - 1) * ny * n2), Hy_x: new Float32Array(n2 * ny * (nz - 1)),
      Hz_x: new Float32Array(n2 * (ny - 1) * nz), Hz_y: new Float32Array((nx - 1) * n2 * nz), Ex_y: new Float32Array((nx - 1) * n2 * nz), Ex_z: new Float32Array((nx - 1) * ny * n2),
      Ey_z: new Float32Array(nx * (ny - 1) * n2), Ey_x: new Float32Array(n2 * (ny - 1) * nz), Ez_x: new Float32Array(n2 * ny * (nz - 1)), Ez_y: new Float32Array(nx * n2 * (nz - 1)) };
    this.tables = M.portTables();
  }
  async init() {}
  slabH(q, N) { const n = this.M.npml; return q < n ? q : (q >= N - n ? q - (N - n) + n : -1); }
  slabE(q, N) { const n = this.M.npml; return (q >= 1 && q <= n) ? q - 1 : ((q >= N - 1 - n && q <= N - 2) ? q - (N - 1 - n) + n : -1); }
  step(nStep, ex, VI, nsteps) {
    const M = this.M, { nx, ny, nz, npml: n } = M, n2 = 2 * n, cH = Math.fround(M.cH);
    const { Ex, Ey, Ez, Hx, Hy, Hz, psi } = this;
    const P = M.pml;
    // H
    for (let ii = 0; ii < nx; ii++) for (let j = 0; j < ny - 1; j++) for (let k = 0; k < nz - 1; k++) {
      let dEz_dy = (Ez[(ii * ny + j + 1) * (nz - 1) + k] - Ez[(ii * ny + j) * (nz - 1) + k]) * M.idy[j];
      let dEy_dz = (Ey[(ii * (ny - 1) + j) * nz + k + 1] - Ey[(ii * (ny - 1) + j) * nz + k]) * M.idz[k];
      let s = this.slabH(j, ny - 1); if (s >= 0) { const q = (ii * n2 + s) * (nz - 1) + k; const p = P.yh.b[s] * psi.Hx_y[q] + P.yh.c[s] * dEz_dy; psi.Hx_y[q] = p; dEz_dy += p; }
      s = this.slabH(k, nz - 1); if (s >= 0) { const q = (ii * (ny - 1) + j) * n2 + s; const p = P.zh.b[s] * psi.Hx_z[q] + P.zh.c[s] * dEy_dz; psi.Hx_z[q] = p; dEy_dz += p; }
      const q = (ii * (ny - 1) + j) * (nz - 1) + k; Hx[q] = Math.fround(Hx[q] - cH * (dEz_dy - dEy_dz));
    }
    for (let ii = 0; ii < nx - 1; ii++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz - 1; k++) {
      let dEx_dz = (Ex[(ii * ny + j) * nz + k + 1] - Ex[(ii * ny + j) * nz + k]) * M.idz[k];
      let dEz_dx = (Ez[((ii + 1) * ny + j) * (nz - 1) + k] - Ez[(ii * ny + j) * (nz - 1) + k]) * M.idx[ii];
      let s = this.slabH(k, nz - 1); if (s >= 0) { const q = (ii * ny + j) * n2 + s; const p = P.zh.b[s] * psi.Hy_z[q] + P.zh.c[s] * dEx_dz; psi.Hy_z[q] = p; dEx_dz += p; }
      s = this.slabH(ii, nx - 1); if (s >= 0) { const q = (s * ny + j) * (nz - 1) + k; const p = P.xh.b[s] * psi.Hy_x[q] + P.xh.c[s] * dEz_dx; psi.Hy_x[q] = p; dEz_dx += p; }
      const q = (ii * ny + j) * (nz - 1) + k; Hy[q] = Math.fround(Hy[q] - cH * (dEx_dz - dEz_dx));
    }
    for (let ii = 0; ii < nx - 1; ii++) for (let j = 0; j < ny - 1; j++) for (let k = 0; k < nz; k++) {
      let dEy_dx = (Ey[((ii + 1) * (ny - 1) + j) * nz + k] - Ey[(ii * (ny - 1) + j) * nz + k]) * M.idx[ii];
      let dEx_dy = (Ex[(ii * ny + j + 1) * nz + k] - Ex[(ii * ny + j) * nz + k]) * M.idy[j];
      let s = this.slabH(ii, nx - 1); if (s >= 0) { const q = (s * (ny - 1) + j) * nz + k; const p = P.xh.b[s] * psi.Hz_x[q] + P.xh.c[s] * dEy_dx; psi.Hz_x[q] = p; dEy_dx += p; }
      s = this.slabH(j, ny - 1); if (s >= 0) { const q = (ii * n2 + s) * nz + k; const p = P.yh.b[s] * psi.Hz_y[q] + P.yh.c[s] * dEx_dy; psi.Hz_y[q] = p; dEx_dy += p; }
      const q = (ii * (ny - 1) + j) * nz + k; Hz[q] = Math.fround(Hz[q] - cH * (dEy_dx - dEx_dy));
    }
    // E
    for (let ii = 0; ii < nx - 1; ii++) for (let j = 1; j <= ny - 2; j++) for (let k = 1; k <= nz - 2; k++) {
      let dHz_dy = (Hz[(ii * (ny - 1) + j) * nz + k] - Hz[(ii * (ny - 1) + j - 1) * nz + k]) * M.iddy[j];
      let dHy_dz = (Hy[(ii * ny + j) * (nz - 1) + k] - Hy[(ii * ny + j) * (nz - 1) + k - 1]) * M.iddz[k];
      let s = this.slabE(j, ny); if (s >= 0) { const q = (ii * n2 + s) * nz + k; const p = P.ye.b[s] * psi.Ex_y[q] + P.ye.c[s] * dHz_dy; psi.Ex_y[q] = p; dHz_dy += p; }
      s = this.slabE(k, nz); if (s >= 0) { const q = (ii * ny + j) * n2 + s; const p = P.ze.b[s] * psi.Ex_z[q] + P.ze.c[s] * dHy_dz; psi.Ex_z[q] = p; dHy_dz += p; }
      const q = (ii * ny + j) * nz + k; Ex[q] = Math.fround(Ex[q] + M.cEx[q] * (dHz_dy - dHy_dz));
    }
    for (let ii = 1; ii <= nx - 2; ii++) for (let j = 0; j < ny - 1; j++) for (let k = 1; k <= nz - 2; k++) {
      let dHx_dz = (Hx[(ii * (ny - 1) + j) * (nz - 1) + k] - Hx[(ii * (ny - 1) + j) * (nz - 1) + k - 1]) * M.iddz[k];
      let dHz_dx = (Hz[(ii * (ny - 1) + j) * nz + k] - Hz[((ii - 1) * (ny - 1) + j) * nz + k]) * M.iddx[ii];
      let s = this.slabE(k, nz); if (s >= 0) { const q = (ii * (ny - 1) + j) * n2 + s; const p = P.ze.b[s] * psi.Ey_z[q] + P.ze.c[s] * dHx_dz; psi.Ey_z[q] = p; dHx_dz += p; }
      s = this.slabE(ii, nx); if (s >= 0) { const q = (s * (ny - 1) + j) * nz + k; const p = P.xe.b[s] * psi.Ey_x[q] + P.xe.c[s] * dHz_dx; psi.Ey_x[q] = p; dHz_dx += p; }
      const q = (ii * (ny - 1) + j) * nz + k; Ey[q] = Math.fround(Ey[q] + M.cEy[q] * (dHx_dz - dHz_dx));
    }
    const { pb, pc, NP } = this.tables, npt = M.ports.length;
    for (let ii = 1; ii <= nx - 2; ii++) for (let j = 1; j <= ny - 2; j++) for (let k = 0; k < nz - 1; k++) {
      let dHy_dx = (Hy[(ii * ny + j) * (nz - 1) + k] - Hy[((ii - 1) * ny + j) * (nz - 1) + k]) * M.iddx[ii];
      let dHx_dy = (Hx[(ii * (ny - 1) + j) * (nz - 1) + k] - Hx[(ii * (ny - 1) + j - 1) * (nz - 1) + k]) * M.iddy[j];
      let s = this.slabE(ii, nx); if (s >= 0) { const q = (s * ny + j) * (nz - 1) + k; const p = P.xe.b[s] * psi.Ez_x[q] + P.xe.c[s] * dHy_dx; psi.Ez_x[q] = p; dHy_dx += p; }
      s = this.slabE(j, ny); if (s >= 0) { const q = (ii * n2 + s) * (nz - 1) + k; const p = P.ye.b[s] * psi.Ez_y[q] + P.ye.c[s] * dHx_dy; psi.Ez_y[q] = p; dHx_dy += p; }
      const q = (ii * ny + j) * (nz - 1) + k;
      const e0 = Ez[q]; let e1 = e0 + M.cEz[q] * (dHy_dx - dHx_dy);
      for (let pq = 0; pq < npt; pq++) if (ii >= pb[pq * 6] && ii <= pb[pq * 6 + 1] && j >= pb[pq * 6 + 2] && j <= pb[pq * 6 + 3] && k >= pb[pq * 6 + 4] && k <= pb[pq * 6 + 5]) { e1 = pc[pq * 3] * (e1 - pc[pq * 3 + 1] * e0) + pc[pq * 3] * pc[pq * 3 + 2] * ex; break; }
      Ez[q] = Math.fround(e1);
    }
    // ports V, I
    M.ports.forEach((p, pq) => {
      let V = 0; for (let k = p.k0; k <= p.k1; k++) V -= Ez[(p.ic * ny + p.jc) * (nz - 1) + k] * M.dz[k];
      let I = 0;
      for (let a = p.i0; a <= p.i1; a++) I += (Hx[(a * (ny - 1) + p.j0 - 1) * (nz - 1) + p.kmid] - Hx[(a * (ny - 1) + p.j1) * (nz - 1) + p.kmid]) * M.ddx[a];
      for (let b = p.j0; b <= p.j1; b++) I += (Hy[(p.i1 * ny + b) * (nz - 1) + p.kmid] - Hy[((p.i0 - 1) * ny + b) * (nz - 1) + p.kmid]) * M.ddy[b];
      VI[pq * nsteps + nStep] = V; VI[NP * nsteps + pq * nsteps + nStep] = I;
    });
  }
  async runSteps(n0, count, excArr, VI, nsteps) {
    for (let n = n0; n < n0 + count; n++) {
      this.step(n, excArr[n], VI, nsteps);
      if (this.nf) accumulateCpu(this.nf.surf, [this.Ex, this.Ey, this.Ez, this.Hx, this.Hy, this.Hz], this.nf.acc, this.nf.nf, this.nf.f0, this.nf.df, n * this.M.dt);
    }
  }
  prepareNf2ff(surf, nf, f0, df) { this.nf = { surf, nf, f0, df, acc: new Float64Array(surf.npts * nf * 8) }; }
  async readNf2ff() { const a = Float32Array.from(this.nf.acc); for (let i = 0; i < a.length; i++) a[i] *= this.M.dt; return a; }
  async energy() { let e = 0; for (const a of [this.Ex, this.Ey, this.Ez]) for (let i = 0; i < a.length; i++) e += a[i] * a[i]; return e; }
  // instantaneous Poynting flux (W) out through a nf2ff surface, from the same samples the transform uses
  boxFlux(surf) {
    const F = [this.Ex, this.Ey, this.Ez, this.Hx, this.Hy, this.Hz], { pts, nrm, area, npts } = surf;
    let flux = 0;
    for (let p = 0; p < npts; p++) {
      const b = p * 12, n = [nrm[3 * p], nrm[3 * p + 1], nrm[3 * p + 2]];
      const e1 = F[pts[b + 1]][pts[b]], e2 = F[pts[b + 3]][pts[b + 2]];
      const h1 = 0.5 * (F[pts[b + 5]][pts[b + 4]] + F[pts[b + 7]][pts[b + 6]]), h2 = 0.5 * (F[pts[b + 9]][pts[b + 8]] + F[pts[b + 11]][pts[b + 10]]);
      const ax = n[0] !== 0 ? [1, 2] : (n[1] !== 0 ? [0, 2] : [0, 1]);
      const E = [0, 0, 0], H = [0, 0, 0]; E[ax[0]] = e1; E[ax[1]] = e2; H[ax[0]] = h1; H[ax[1]] = h2;
      const S = [E[1] * H[2] - E[2] * H[1], E[2] * H[0] - E[0] * H[2], E[0] * H[1] - E[1] * H[0]];
      flux += (S[0] * n[0] + S[1] * n[1] + S[2] * n[2]) * area[p];
    }
    return flux;
  }
  // physical field energy (J): sum 1/2 eps E^2 dV + 1/2 mu H^2 dV on the Yee cells (eps from the E coefficients)
  physEnergy() {
    const M = this.M, { nx, ny, nz } = M, dx = M.dx, dy = M.dy, dz = M.dz, ddx = M.ddx, ddy = M.ddy, ddz = M.ddz;
    let We = 0, Wm = 0;
    for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) { const q = (i * ny + j) * nz + k; const c = M.cEx[q]; if (c > 0) We += 0.5 * (M.dt / c) * this.Ex[q] ** 2 * dx[i] * ddy[j] * ddz[k]; }
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) for (let k = 0; k < nz; k++) { const q = (i * (ny - 1) + j) * nz + k; const c = M.cEy[q]; if (c > 0) We += 0.5 * (M.dt / c) * this.Ey[q] ** 2 * ddx[i] * dy[j] * ddz[k]; }
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz - 1; k++) { const q = (i * ny + j) * (nz - 1) + k; const c = M.cEz[q]; if (c > 0) We += 0.5 * (M.dt / c) * this.Ez[q] ** 2 * ddx[i] * ddy[j] * dz[k]; }
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) for (let k = 0; k < nz - 1; k++) Wm += 0.5 * MU0 * this.Hx[(i * (ny - 1) + j) * (nz - 1) + k] ** 2 * ddx[i] * dy[j] * dz[k];
    for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz - 1; k++) Wm += 0.5 * MU0 * this.Hy[(i * ny + j) * (nz - 1) + k] ** 2 * dx[i] * ddy[j] * dz[k];
    for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) for (let k = 0; k < nz; k++) Wm += 0.5 * MU0 * this.Hz[(i * (ny - 1) + j) * nz + k] ** 2 * dx[i] * dy[j] * ddz[k];
    return { We, Wm, W: We + Wm };
  }
  async readVI(VI) { return VI; }
  async readSlice(kz) { const { nx, ny, nz } = this.M; const s = new Float32Array(nx * ny); for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) s[i * ny + j] = this.Ez[(i * ny + j) * (nz - 1) + kz]; return s; }
  destroy() {}
}

// ---------------------------------------------------------------- WebGPU engine
const WG = 256;
function wgsl(M) {
  const { nx, ny, nz, npml: n } = M;
  const n2 = 2 * n;
  // params buffer layout (f32): idx[nx-1] idy[ny-1] idz[nz-1] iddx[nx] iddy[ny] iddz[nz] then pml vectors (each 2n) xh.b xh.c xe.b xe.c yh.b yh.c ye.b ye.c zh.b zh.c ze.b ze.c, dz[nz-1] ddx[nx] ddy[ny]
  const off = {}; let o = 0;
  for (const [k, len] of [["idx", nx - 1], ["idy", ny - 1], ["idz", nz - 1], ["iddx", nx], ["iddy", ny], ["iddz", nz],
    ["xhb", n2], ["xhc", n2], ["xeb", n2], ["xec", n2], ["yhb", n2], ["yhc", n2], ["yeb", n2], ["yec", n2], ["zhb", n2], ["zhc", n2], ["zeb", n2], ["zec", n2],
    ["dz", nz - 1], ["ddx", nx], ["ddy", ny]]) { off[k] = o; o += len; }
  const paramsLen = o;
  // psi layout
  const po = {}; o = 0;
  for (const [k, len] of [["Hx_y", nx * n2 * (nz - 1)], ["Hx_z", nx * (ny - 1) * n2], ["Hy_z", (nx - 1) * ny * n2], ["Hy_x", n2 * ny * (nz - 1)], ["Hz_x", n2 * (ny - 1) * nz], ["Hz_y", (nx - 1) * n2 * nz],
    ["Ex_y", (nx - 1) * n2 * nz], ["Ex_z", (nx - 1) * ny * n2], ["Ey_z", nx * (ny - 1) * n2], ["Ey_x", n2 * (ny - 1) * nz], ["Ez_x", n2 * ny * (nz - 1)], ["Ez_y", nx * n2 * (nz - 1)]]) { po[k] = o; o += len; }
  const psiLen = Math.max(1, o);
  const head = `
const NX: i32 = ${nx}; const NY: i32 = ${ny}; const NZ: i32 = ${nz}; const NP: i32 = ${n}; const N2: i32 = ${n2};
struct U { cH: f32, step: u32, npt: u32, nsteps: u32, ex: f32, pad0: f32, dt: f32, pad2: f32, nf: u32, npts: u32, f0: f32, df: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> prm: array<f32>;
@group(0) @binding(2) var<storage, read_write> psi: array<f32>;
fn gid(w: vec3<u32>, l: u32) -> i32 { return i32((w.x + w.y * 65535u) * ${WG}u + l); }
fn slab_h(q: i32, N: i32) -> i32 { if (q < NP) { return q; } if (q >= N - NP) { return q - (N - NP) + NP; } return -1; }
fn slab_e(q: i32, N: i32) -> i32 { if (q >= 1 && q <= NP) { return q - 1; } if (q >= N - 1 - NP && q <= N - 2) { return q - (N - 1 - NP) + NP; } return -1; }
`;
  const K = {};
  K.Hx = head + `
@group(0) @binding(3) var<storage, read> Ey: array<f32>;
@group(0) @binding(4) var<storage, read> Ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> Hx: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NYm = NY - 1; let NZm = NZ - 1; if (i >= NX * NYm * NZm) { return; }
  let k = i % NZm; let j = (i / NZm) % NYm; let ii = i / (NZm * NYm);
  var dEz_dy = (Ez[(ii * NY + j + 1) * NZm + k] - Ez[(ii * NY + j) * NZm + k]) * prm[${off.idy} + j];
  var dEy_dz = (Ey[(ii * NYm + j) * NZ + k + 1] - Ey[(ii * NYm + j) * NZ + k]) * prm[${off.idz} + k];
  var s = slab_h(j, NYm); if (s >= 0) { let q = ${po.Hx_y} + (ii * N2 + s) * NZm + k; let p = prm[${off.yhb} + s] * psi[q] + prm[${off.yhc} + s] * dEz_dy; psi[q] = p; dEz_dy += p; }
  s = slab_h(k, NZm); if (s >= 0) { let q = ${po.Hx_z} + (ii * NYm + j) * N2 + s; let p = prm[${off.zhb} + s] * psi[q] + prm[${off.zhc} + s] * dEy_dz; psi[q] = p; dEy_dz += p; }
  Hx[i] = Hx[i] - u.cH * (dEz_dy - dEy_dz);
}`;
  K.Hy = head + `
@group(0) @binding(3) var<storage, read> Ex: array<f32>;
@group(0) @binding(4) var<storage, read> Ez: array<f32>;
@group(0) @binding(5) var<storage, read_write> Hy: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NXm = NX - 1; let NZm = NZ - 1; if (i >= NXm * NY * NZm) { return; }
  let k = i % NZm; let j = (i / NZm) % NY; let ii = i / (NZm * NY);
  var dEx_dz = (Ex[(ii * NY + j) * NZ + k + 1] - Ex[(ii * NY + j) * NZ + k]) * prm[${off.idz} + k];
  var dEz_dx = (Ez[((ii + 1) * NY + j) * NZm + k] - Ez[(ii * NY + j) * NZm + k]) * prm[${off.idx} + ii];
  var s = slab_h(k, NZm); if (s >= 0) { let q = ${po.Hy_z} + (ii * NY + j) * N2 + s; let p = prm[${off.zhb} + s] * psi[q] + prm[${off.zhc} + s] * dEx_dz; psi[q] = p; dEx_dz += p; }
  s = slab_h(ii, NXm); if (s >= 0) { let q = ${po.Hy_x} + (s * NY + j) * NZm + k; let p = prm[${off.xhb} + s] * psi[q] + prm[${off.xhc} + s] * dEz_dx; psi[q] = p; dEz_dx += p; }
  Hy[i] = Hy[i] - u.cH * (dEx_dz - dEz_dx);
}`;
  K.Hz = head + `
@group(0) @binding(3) var<storage, read> Ex: array<f32>;
@group(0) @binding(4) var<storage, read> Ey: array<f32>;
@group(0) @binding(5) var<storage, read_write> Hz: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NXm = NX - 1; let NYm = NY - 1; if (i >= NXm * NYm * NZ) { return; }
  let k = i % NZ; let j = (i / NZ) % NYm; let ii = i / (NZ * NYm);
  var dEy_dx = (Ey[((ii + 1) * NYm + j) * NZ + k] - Ey[(ii * NYm + j) * NZ + k]) * prm[${off.idx} + ii];
  var dEx_dy = (Ex[(ii * NY + j + 1) * NZ + k] - Ex[(ii * NY + j) * NZ + k]) * prm[${off.idy} + j];
  var s = slab_h(ii, NXm); if (s >= 0) { let q = ${po.Hz_x} + (s * NYm + j) * NZ + k; let p = prm[${off.xhb} + s] * psi[q] + prm[${off.xhc} + s] * dEy_dx; psi[q] = p; dEy_dx += p; }
  s = slab_h(j, NYm); if (s >= 0) { let q = ${po.Hz_y} + (ii * N2 + s) * NZ + k; let p = prm[${off.yhb} + s] * psi[q] + prm[${off.yhc} + s] * dEx_dy; psi[q] = p; dEx_dy += p; }
  Hz[i] = Hz[i] - u.cH * (dEy_dx - dEx_dy);
}`;
  K.Ex = head + `
@group(0) @binding(3) var<storage, read> Hy: array<f32>;
@group(0) @binding(4) var<storage, read> Hz: array<f32>;
@group(0) @binding(5) var<storage, read_write> Ex: array<f32>;
@group(0) @binding(6) var<storage, read> cE: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NXm = NX - 1; let NYm = NY - 1; let NZm = NZ - 1; if (i >= NXm * NY * NZ) { return; }
  let k = i % NZ; let j = (i / NZ) % NY; let ii = i / (NZ * NY);
  if (j < 1 || j > NY - 2 || k < 1 || k > NZ - 2) { return; }
  var dHz_dy = (Hz[(ii * NYm + j) * NZ + k] - Hz[(ii * NYm + j - 1) * NZ + k]) * prm[${off.iddy} + j];
  var dHy_dz = (Hy[(ii * NY + j) * NZm + k] - Hy[(ii * NY + j) * NZm + k - 1]) * prm[${off.iddz} + k];
  var s = slab_e(j, NY); if (s >= 0) { let q = ${po.Ex_y} + (ii * N2 + s) * NZ + k; let p = prm[${off.yeb} + s] * psi[q] + prm[${off.yec} + s] * dHz_dy; psi[q] = p; dHz_dy += p; }
  s = slab_e(k, NZ); if (s >= 0) { let q = ${po.Ex_z} + (ii * NY + j) * N2 + s; let p = prm[${off.zeb} + s] * psi[q] + prm[${off.zec} + s] * dHy_dz; psi[q] = p; dHy_dz += p; }
  Ex[i] = Ex[i] + cE[i] * (dHz_dy - dHy_dz);
}`;
  K.Ey = head + `
@group(0) @binding(3) var<storage, read> Hx: array<f32>;
@group(0) @binding(4) var<storage, read> Hz: array<f32>;
@group(0) @binding(5) var<storage, read_write> Ey: array<f32>;
@group(0) @binding(6) var<storage, read> cE: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NYm = NY - 1; let NZm = NZ - 1; if (i >= NX * NYm * NZ) { return; }
  let k = i % NZ; let j = (i / NZ) % NYm; let ii = i / (NZ * NYm);
  if (ii < 1 || ii > NX - 2 || k < 1 || k > NZ - 2) { return; }
  var dHx_dz = (Hx[(ii * NYm + j) * NZm + k] - Hx[(ii * NYm + j) * NZm + k - 1]) * prm[${off.iddz} + k];
  var dHz_dx = (Hz[(ii * NYm + j) * NZ + k] - Hz[((ii - 1) * NYm + j) * NZ + k]) * prm[${off.iddx} + ii];
  var s = slab_e(k, NZ); if (s >= 0) { let q = ${po.Ey_z} + (ii * NYm + j) * N2 + s; let p = prm[${off.zeb} + s] * psi[q] + prm[${off.zec} + s] * dHx_dz; psi[q] = p; dHx_dz += p; }
  s = slab_e(ii, NX); if (s >= 0) { let q = ${po.Ey_x} + (s * NYm + j) * NZ + k; let p = prm[${off.xeb} + s] * psi[q] + prm[${off.xec} + s] * dHz_dx; psi[q] = p; dHz_dx += p; }
  Ey[i] = Ey[i] + cE[i] * (dHx_dz - dHz_dx);
}`;
  K.Ez = head + `
@group(0) @binding(3) var<storage, read> Hx: array<f32>;
@group(0) @binding(4) var<storage, read> Hy: array<f32>;
@group(0) @binding(5) var<storage, read_write> Ez: array<f32>;
@group(0) @binding(6) var<storage, read> cE: array<f32>;
@group(0) @binding(7) var<storage, read> pt: array<i32>;      // pb (NP*6) then pc as bitcast (NP*3)
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); let NYm = NY - 1; let NZm = NZ - 1; if (i >= NX * NY * NZm) { return; }
  let k = i % NZm; let j = (i / NZm) % NY; let ii = i / (NZm * NY);
  if (ii < 1 || ii > NX - 2 || j < 1 || j > NY - 2) { return; }
  var dHy_dx = (Hy[(ii * NY + j) * NZm + k] - Hy[((ii - 1) * NY + j) * NZm + k]) * prm[${off.iddx} + ii];
  var dHx_dy = (Hx[(ii * NYm + j) * NZm + k] - Hx[(ii * NYm + j - 1) * NZm + k]) * prm[${off.iddy} + j];
  var s = slab_e(ii, NX); if (s >= 0) { let q = ${po.Ez_x} + (s * NY + j) * NZm + k; let p = prm[${off.xeb} + s] * psi[q] + prm[${off.xec} + s] * dHy_dx; psi[q] = p; dHy_dx += p; }
  s = slab_e(j, NY); if (s >= 0) { let q = ${po.Ez_y} + (ii * N2 + s) * NZm + k; let p = prm[${off.yeb} + s] * psi[q] + prm[${off.yec} + s] * dHx_dy; psi[q] = p; dHx_dy += p; }
  let e0 = Ez[i]; var e1 = e0 + cE[i] * (dHy_dx - dHx_dy);
  let npt = i32(u.npt);
  for (var q = 0; q < npt; q++) {
    if (ii >= pt[q * 6] && ii <= pt[q * 6 + 1] && j >= pt[q * 6 + 2] && j <= pt[q * 6 + 3] && k >= pt[q * 6 + 4] && k <= pt[q * 6 + 5]) {
      let lb = bitcast<f32>(pt[npt * 6 + q * 3]); let f = bitcast<f32>(pt[npt * 6 + q * 3 + 1]); let sc = bitcast<f32>(pt[npt * 6 + q * 3 + 2]);
      e1 = lb * (e1 - f * e0) + lb * sc * u.ex; break;
    }
  }
  Ez[i] = e1;
}`;
  K.VI = head + `
@group(0) @binding(3) var<storage, read> Ez: array<f32>;
@group(0) @binding(4) var<storage, read> Hx: array<f32>;
@group(0) @binding(5) var<storage, read> Hy: array<f32>;
@group(0) @binding(6) var<storage, read> pg: array<i32>;
@group(0) @binding(7) var<storage, read_write> VI: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) w: vec3<u32>) {
  let p = i32(w.x); let NYm = NY - 1; let NZm = NZ - 1;
  let ic = pg[p * 9]; let jc = pg[p * 9 + 1]; let k0 = pg[p * 9 + 2]; let k1 = pg[p * 9 + 3]; let km = pg[p * 9 + 4];
  let i0 = pg[p * 9 + 5]; let i1 = pg[p * 9 + 6]; let j0 = pg[p * 9 + 7]; let j1 = pg[p * 9 + 8];
  var V = 0.0; for (var k = k0; k <= k1; k++) { V -= Ez[(ic * NY + jc) * NZm + k] * prm[${off.dz} + k]; }
  var I = 0.0;
  for (var a = i0; a <= i1; a++) { I += (Hx[(a * NYm + j0 - 1) * NZm + km] - Hx[(a * NYm + j1) * NZm + km]) * prm[${off.ddx} + a]; }
  for (var b = j0; b <= j1; b++) { I += (Hy[(i1 * NY + b) * NZm + km] - Hy[((i0 - 1) * NY + b) * NZm + km]) * prm[${off.ddy} + b]; }
  let ns = i32(u.nsteps); let npt = i32(u.npt); let n = i32(u.step);
  VI[p * ns + n] = V; VI[npt * ns + p * ns + n] = I;
}`;
  K.reduce = head + `
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
var<workgroup> sh: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let n = arrayLength(&A); let g = w.x + w.y * 65535u; var s = 0.0;
  var i = g * ${WG}u * 8u + l;
  for (var r = 0u; r < 8u; r++) { if (i < n) { s += A[i] * A[i]; } i += ${WG}u; }
  sh[l] = s; workgroupBarrier();
  for (var st = ${WG / 2}u; st > 0u; st >>= 1u) { if (l < st) { sh[l] += sh[l + st]; } workgroupBarrier(); }
  if (l == 0u) { part[g] = sh[0]; }
}`;
  K.slice = head + `
@group(0) @binding(3) var<storage, read> Ez: array<f32>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = gid(w, l); if (i >= NX * NY) { return; }
  S[i] = Ez[i * (NZ - 1) + i32(u.pad0)];
}`;
  // near-to-far-field accumulation: one thread per (surface point, frequency)
  K.nf = `
struct U { cH: f32, step: u32, npt: u32, nsteps: u32, ex: f32, pad0: f32, dt: f32, pad2: f32, nf: u32, npts: u32, f0: f32, df: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> Ex: array<f32>;
@group(0) @binding(2) var<storage, read> Ey: array<f32>;
@group(0) @binding(3) var<storage, read> Ez: array<f32>;
@group(0) @binding(4) var<storage, read> Hx: array<f32>;
@group(0) @binding(5) var<storage, read> Hy: array<f32>;
@group(0) @binding(6) var<storage, read> Hz: array<f32>;
@group(0) @binding(7) var<storage, read> pts: array<i32>;
@group(0) @binding(8) var<storage, read_write> acc: array<f32>;
fn fld(a: i32, i: i32) -> f32 { switch a { case 0: { return Ex[i]; } case 1: { return Ey[i]; } case 2: { return Ez[i]; } case 3: { return Hx[i]; } case 4: { return Hy[i]; } default: { return Hz[i]; } } }
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let id = i32((w.x + w.y * 65535u) * ${WG}u + l); let nf = i32(u.nf); if (id >= i32(u.npts) * nf) { return; }
  let p = id / nf; let q = id % nf; let b = p * 12;
  let e1 = fld(pts[b + 1], pts[b]); let e2 = fld(pts[b + 3], pts[b + 2]);
  let h1 = 0.5 * (fld(pts[b + 5], pts[b + 4]) + fld(pts[b + 7], pts[b + 6])); let h2 = 0.5 * (fld(pts[b + 9], pts[b + 8]) + fld(pts[b + 11], pts[b + 10]));
  let wt = 6.283185307179586 * (u.f0 + f32(q) * u.df) * f32(u.step) * u.dt; let c = cos(wt); let s = -sin(wt);
  let o = id * 8;
  acc[o] += e1 * c; acc[o + 1] += e1 * s; acc[o + 2] += e2 * c; acc[o + 3] += e2 * s;
  acc[o + 4] += h1 * c; acc[o + 5] += h1 * s; acc[o + 6] += h2 * c; acc[o + 7] += h2 * s;
}`;
  // explicit bind group layouts (an "auto" layout drops bindings a kernel does not read)
  const U = "uniform", R = "read-only-storage", S = "storage";
  const LAY = { Hx: [U, R, S, R, R, S], Hy: [U, R, S, R, R, S], Hz: [U, R, S, R, R, S], Ex: [U, R, S, R, R, S, R], Ey: [U, R, S, R, R, S, R], Ez: [U, R, S, R, R, S, R, R],
    VI: [U, R, S, R, R, R, R, S], reduce: [U, R, S, R, S], slice: [U, R, S, R, S], nf: [U, R, R, R, R, R, R, R, S] };
  return { K, LAY, off, po, paramsLen, psiLen };
}

export class GpuEngine {
  constructor(M) { this.M = M; this.tables = M.portTables(); }
  static async available() { if (!navigator.gpu) return null; try { return await navigator.gpu.requestAdapter(); } catch (e) { return null; } }
  async init() {
    const M = this.M;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const need = Math.max(M.cEx.byteLength, M.cEy.byteLength, M.cEz.byteLength);
    const dev = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(134217728, need)), maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(268435456, need)) } });
    this.dev = dev;
    this.adapterInfo = adapter.info || null;
    const g = wgsl(M); this.g = g;
    const mk = (size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) => dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    const up = (buf, arr) => dev.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    const { nx, ny, nz } = M;
    this.B = {
      Ex: mk(M.cEx.byteLength), Ey: mk(M.cEy.byteLength), Ez: mk(M.cEz.byteLength),
      Hx: mk(4 * nx * (ny - 1) * (nz - 1)), Hy: mk(4 * (nx - 1) * ny * (nz - 1)), Hz: mk(4 * (nx - 1) * (ny - 1) * nz),
      cEx: mk(M.cEx.byteLength), cEy: mk(M.cEy.byteLength), cEz: mk(M.cEz.byteLength),
      prm: mk(4 * g.paramsLen), psi: mk(4 * g.psiLen), pt: mk(4 * (this.tables.NP * 9)), pg: mk(4 * this.tables.NP * 9),
      u: dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    up(this.B.cEx, M.cEx); up(this.B.cEy, M.cEy); up(this.B.cEz, M.cEz);
    const prm = new Float32Array(g.paramsLen);
    const put = (k, arr) => prm.set(arr, g.off[k]);
    put("idx", M.idx); put("idy", M.idy); put("idz", M.idz); put("iddx", M.iddx); put("iddy", M.iddy); put("iddz", M.iddz);
    for (const ax of "xyz") for (const kd of "he") { put(ax + kd + "b", M.pml[ax + kd].b); put(ax + kd + "c", M.pml[ax + kd].c); }
    put("dz", Float32Array.from(M.dz)); put("ddx", Float32Array.from(M.ddx)); put("ddy", Float32Array.from(M.ddy));
    up(this.B.prm, prm);
    const { pb, pc, pg, NP } = this.tables;
    const pt = new Int32Array(NP * 9); pt.set(pb, 0); pt.set(new Int32Array(pc.buffer), NP * 6); up(this.B.pt, pt); up(this.B.pg, pg);
    this.pipes = {};
    this.errors = [];
    dev.addEventListener("uncapturederror", e => { this.errors.push(e.error.message); console.error("WebGPU:", e.error.message); });
    for (const [name, code] of Object.entries(g.K)) {
      const module = dev.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const bad = info.messages.filter(m => m.type === "error");
      if (bad.length) throw new Error(`WGSL ${name}: ` + bad.map(m => `line ${m.lineNum}: ${m.message}`).join("; "));
      const bgl = dev.createBindGroupLayout({ entries: g.LAY[name].map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: t } })) });
      try { this.pipes[name] = await dev.createComputePipelineAsync({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: "main" } }); }
      catch (e) { throw new Error(`pipeline ${name}: ${e.message}`); }
    }
    const bg = (name, entries) => dev.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const B = this.B;
    this.bgs = {
      Hx: bg("Hx", [B.u, B.prm, B.psi, B.Ey, B.Ez, B.Hx]), Hy: bg("Hy", [B.u, B.prm, B.psi, B.Ex, B.Ez, B.Hy]), Hz: bg("Hz", [B.u, B.prm, B.psi, B.Ex, B.Ey, B.Hz]),
      Ex: bg("Ex", [B.u, B.prm, B.psi, B.Hy, B.Hz, B.Ex, B.cEx]), Ey: bg("Ey", [B.u, B.prm, B.psi, B.Hx, B.Hz, B.Ey, B.cEy]),
    };
    this.sizes = { Hx: nx * (ny - 1) * (nz - 1), Hy: (nx - 1) * ny * (nz - 1), Hz: (nx - 1) * (ny - 1) * nz, Ex: (nx - 1) * ny * nz, Ey: nx * (ny - 1) * nz, Ez: nx * ny * (nz - 1) };
    this.uni = new ArrayBuffer(48); this.uf = new Float32Array(this.uni); this.uu = new Uint32Array(this.uni);
    this.uf[0] = M.cH; this.uf[6] = M.dt;
  }
  prepareNf2ff(surf, nf, f0, df) {
    const dev = this.dev, B = this.B;
    B.pts = dev.createBuffer({ size: Math.max(16, surf.pts.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(B.pts, 0, surf.pts.buffer, surf.pts.byteOffset, surf.pts.byteLength);
    B.acc = dev.createBuffer({ size: Math.max(16, 4 * 8 * surf.npts * nf), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    B.accRead = dev.createBuffer({ size: B.acc.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.bgs.nf = dev.createBindGroup({ layout: this.pipes.nf.getBindGroupLayout(0), entries: [B.u, B.Ex, B.Ey, B.Ez, B.Hx, B.Hy, B.Hz, B.pts, B.acc].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    this.nfCount = surf.npts * nf;
    this.uu[8] = nf; this.uu[9] = surf.npts; this.uf[10] = f0; this.uf[11] = df;
  }
  async readNf2ff() { const a = await this._readback(this.B.acc, this.B.accRead, 4 * 8 * this.nfCount); for (let i = 0; i < a.length; i++) a[i] *= this.M.dt; return a; }
  dispatch(pass, name, count) {
    const groups = Math.ceil(count / WG); pass.setPipeline(this.pipes[name]);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
  }
  prepareRun(nsteps) {
    const { NP } = this.tables, dev = this.dev, B = this.B;
    B.VI = dev.createBuffer({ size: Math.max(16, 4 * 2 * NP * nsteps), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    B.VIread = dev.createBuffer({ size: B.VI.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bg = (name, entries) => dev.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    this.bgs.Ez = bg("Ez", [B.u, B.prm, B.psi, B.Hx, B.Hy, B.Ez, B.cEz, B.pt]);
    this.bgs.VI = bg("VI", [B.u, B.prm, B.psi, B.Ez, B.Hx, B.Hy, B.pg, B.VI]);
    const ngroups = Math.max(...["Ex", "Ey", "Ez"].map(k => Math.ceil(this.sizes[k] / (WG * 8))));
    B.part = dev.createBuffer({ size: Math.max(16, 4 * ngroups), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    B.partRead = dev.createBuffer({ size: B.part.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.bgs.red = { Ex: bg("reduce", [B.u, B.prm, B.psi, B.Ex, B.part]), Ey: bg("reduce", [B.u, B.prm, B.psi, B.Ey, B.part]), Ez: bg("reduce", [B.u, B.prm, B.psi, B.Ez, B.part]) };
    B.S = dev.createBuffer({ size: 4 * this.M.nx * this.M.ny, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    B.Sread = dev.createBuffer({ size: B.S.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.bgs.slice = bg("slice", [B.u, B.prm, B.psi, B.Ez, B.S]);
    this.nsteps = nsteps; this.ngroups = ngroups;
    this.uu[2] = this.M.ports.length; this.uu[3] = nsteps;
  }
  async runSteps(n0, count, excArr) {
    // one command buffer per step (the uniform changes each step); submitted in a batch
    const dev = this.dev, S = this.sizes;
    for (let n = n0; n < n0 + count; n++) {
      this.uu[1] = n; this.uf[4] = excArr[n];
      dev.queue.writeBuffer(this.B.u, 0, this.uni);
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const k of ["Hx", "Hy", "Hz"]) { pass.setBindGroup(0, this.bgs[k]); this.dispatch(pass, k, S[k]); }
      for (const k of ["Ex", "Ey", "Ez"]) { pass.setBindGroup(0, this.bgs[k]); this.dispatch(pass, k, S[k]); }
      pass.setPipeline(this.pipes.VI); pass.setBindGroup(0, this.bgs.VI); pass.dispatchWorkgroups(Math.max(1, this.M.ports.length));
      if (this.bgs.nf) { pass.setBindGroup(0, this.bgs.nf); this.dispatch(pass, "nf", this.nfCount); }
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    await dev.queue.onSubmittedWorkDone();
  }
  async _readback(src, dst, bytes) {
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, dst, 0, bytes); this.dev.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ); const out = new Float32Array(dst.getMappedRange().slice(0, bytes)); dst.unmap(); return out;
  }
  async energy() {
    let e = 0;
    for (const k of ["Ex", "Ey", "Ez"]) {
      const enc = this.dev.createCommandEncoder(); const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.reduce); pass.setBindGroup(0, this.bgs.red[k]);
      const groups = Math.ceil(this.sizes[k] / (WG * 8)); pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535)); pass.end();
      this.dev.queue.submit([enc.finish()]);
      const part = await this._readback(this.B.part, this.B.partRead, 4 * groups);
      for (let i = 0; i < groups; i++) e += part[i];
    }
    return e;
  }
  async readVI() { return this._readback(this.B.VI, this.B.VIread, 4 * 2 * this.tables.NP * this.nsteps); }
  async readSlice(kz) {
    this.uf[5] = kz; this.dev.queue.writeBuffer(this.B.u, 0, this.uni);
    const enc = this.dev.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes.slice); pass.setBindGroup(0, this.bgs.slice); const groups = Math.ceil(this.M.nx * this.M.ny / WG); pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535)); pass.end();
    this.dev.queue.submit([enc.finish()]);
    return this._readback(this.B.S, this.B.Sread, 4 * this.M.nx * this.M.ny);
  }
  destroy() { try { for (const b of Object.values(this.B)) b.destroy && b.destroy(); this.dev.destroy(); } catch (e) {} }
}

// ---------------------------------------------------------------- run loop (either engine)
export async function run(M, engine, exc, nsteps, { end = 1e-3, minSteps = 1000, onProgress = null, snapEvery = 0, snapK = 0, onSnap = null, batch = 100, shouldStop = null, nf2ff = null } = {}) {
  const excArr = new Float32Array(nsteps); for (let n = 0; n < nsteps; n++) excArr[n] = exc(n * M.dt);
  const NP = Math.max(1, M.ports.length);
  let VI = null;
  if (engine instanceof CpuEngine) VI = new Float32Array(2 * NP * nsteps); else engine.prepareRun(nsteps);
  if (nf2ff) engine.prepareNf2ff(nf2ff.surf, nf2ff.nf, nf2ff.f0, nf2ff.df);
  const t0 = performance.now();
  let emax = 0, n = 0, energyDb = 0;
  for (n = 0; n < nsteps; n += batch) {
    const count = Math.min(batch, nsteps - n);
    if (engine instanceof CpuEngine) await engine.runSteps(n, count, excArr, VI, nsteps); else await engine.runSteps(n, count, excArr);
    const done = n + count;
    const e = await engine.energy();
    emax = Math.max(emax, e); energyDb = emax > 0 ? 10 * Math.log10(e / emax + 1e-30) : 0;
    if (snapEvery && onSnap && Math.floor(done / snapEvery) !== Math.floor(n / snapEvery)) onSnap(done, await engine.readSlice(snapK));
    if (onProgress) onProgress({ step: done, nsteps, energyDb, elapsed: (performance.now() - t0) / 1000 });
    if (engine instanceof CpuEngine) await new Promise(res => setTimeout(res, 0));     // let a worker receive its stop message
    if (shouldStop && shouldStop()) { n = done; break; }
    if (done > minSteps && emax > 0 && e < end * emax) { n = done; break; }
    if (done >= nsteps) { n = done; break; }
  }
  const stepsDone = Math.min(n, nsteps);
  const vi = await engine.readVI(VI);
  M.ports.forEach((p, q) => { p.V = Float64Array.from(vi.subarray(q * nsteps, q * nsteps + stepsDone)); p.I = Float64Array.from(vi.subarray(NP * nsteps + q * nsteps, NP * nsteps + q * nsteps + stepsDone)); });
  const acc = nf2ff ? await engine.readNf2ff() : null;
  return { steps: stepsDone, wall: (performance.now() - t0) / 1000, energyDb, acc };
}
