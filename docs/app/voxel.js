// Exact Yee-edge assignment (port of kicad_fdtd/voxel.py): every E edge whose midpoint lies
// inside a copper polygon on that layer's z plane belongs to that conductor; via barrels
// assign the edges inside their cylinder on every plane and the z edges they cross.
// merge_check: nodes touched by edges of two different conductors = shorts in the solver.
import { scanPoly, lowerBound, upperBound } from "./geom.js";

export function assign(bd, lines, tie = {}, skipViaNear = [], window = null) {
  const x = lines.x, y = lines.y, z = lines.z;
  const nx = x.length, ny = y.length, nz = z.length;
  const mx = new Float64Array(nx - 1), my = new Float64Array(ny - 1), mz = new Float64Array(nz - 1);
  for (let i = 0; i < nx - 1; i++) mx[i] = (x[i] + x[i + 1]) / 2;
  for (let j = 0; j < ny - 1; j++) my[j] = (y[j] + y[j + 1]) / 2;
  for (let k = 0; k < nz - 1; k++) mz[k] = (z[k] + z[k + 1]) / 2;
  const ex = new Int16Array((nx - 1) * ny * nz).fill(-1);      // index (i*ny + j)*nz + k
  const ey = new Int16Array(nx * (ny - 1) * nz).fill(-1);      // (i*(ny-1) + j)*nz + k
  const ez = new Int16Array(nx * ny * (nz - 1)).fill(-1);      // (i*ny + j)*(nz-1) + k
  const names = [];
  const nid = net => { net = tie[net] || net; let i = names.indexOf(net); if (i < 0) { names.push(net); i = names.length - 1; } return i; };
  const zidx = {};
  for (const layer of bd.layers) {
    let best = 0; for (let k = 1; k < nz; k++) if (Math.abs(z[k] - bd.z[layer]) < Math.abs(z[best] - bd.z[layer])) best = k;
    if (Math.abs(z[best] - bd.z[layer]) > 1e-6) throw new Error(`copper plane ${layer} is not on a z line`);
    zidx[layer] = best;
  }
  const inWin = P => !window || !(P.bbox[2] < window[0] || P.bbox[0] > window[2] || P.bbox[3] < window[1] || P.bbox[1] > window[3]);
  for (const layer of bd.layers) {
    const k = zidx[layer];
    for (const P of bd.polys(layer)) {
      if (!inWin(P)) continue;
      const id = nid(P.net);
      scanPoly(P, mx, y, (i, j) => { ex[(i * ny + j) * nz + k] = id; });
      scanPoly(P, x, my, (i, j) => { ey[(i * (ny - 1) + j) * nz + k] = id; });
    }
  }
  let nVia = 0;
  for (const v of bd.vias()) {
    if (window && !(v.x >= window[0] && v.x <= window[2] && v.y >= window[1] && v.y <= window[3])) continue;
    if (skipViaNear.some(([px, py]) => Math.hypot(v.x - px, v.y - py) < 0.6)) continue;
    const id = nid(v.net), r = v.r;
    const zb = Math.min(v.zb, v.zt), zt = Math.max(v.zb, v.zt);
    const kz = [], kzm = [];
    for (let k = 0; k < nz; k++) if (z[k] >= zb - 1e-9 && z[k] <= zt + 1e-9) kz.push(k);
    for (let k = 0; k < nz - 1; k++) if (mz[k] >= zb - 1e-9 && mz[k] <= zt + 1e-9) kzm.push(k);
    // ex edges: midpoints (mx[i], y[j]) inside the circle
    for (let i = lowerBound(mx, v.x - r); i < upperBound(mx, v.x + r); i++)
      for (let j = lowerBound(y, v.y - r); j < upperBound(y, v.y + r); j++)
        if (Math.hypot(mx[i] - v.x, y[j] - v.y) <= r) for (const k of kz) ex[(i * ny + j) * nz + k] = id;
    for (let i = lowerBound(x, v.x - r); i < upperBound(x, v.x + r); i++)
      for (let j = lowerBound(my, v.y - r); j < upperBound(my, v.y + r); j++)
        if (Math.hypot(x[i] - v.x, my[j] - v.y) <= r) for (const k of kz) ey[(i * (ny - 1) + j) * nz + k] = id;
    for (let i = lowerBound(x, v.x - r); i < upperBound(x, v.x + r); i++)
      for (let j = lowerBound(y, v.y - r); j < upperBound(y, v.y + r); j++)
        if (Math.hypot(x[i] - v.x, y[j] - v.y) <= r) for (const k of kzm) ez[(i * ny + j) * (nz - 1) + k] = id;
    nVia++;
  }
  return { ex, ey, ez, names, nVia, nx, ny, nz };
}

export function mergeCheck(vox, lines, netsOfInterest, tie = {}) {
  const { ex, ey, ez, names, nx, ny, nz } = vox;
  const x = lines.x, y = lines.y, z = lines.z;
  const watch = new Set(netsOfInterest.map(n => tie[n] || n));
  const out = [];
  const seen = new Set();
  const consider = (i, j, k, ids) => {
    let mn = 32000, mx_ = -1;
    for (const v of ids) if (v >= 0) { if (v < mn) mn = v; if (v > mx_) mx_ = v; }
    if (mx_ < 0 || mn === mx_) return;
    const s = [...new Set(ids.filter(v => v >= 0))].sort((a, b) => a - b).map(v => names[v]);
    if (s.some(n => watch.has(n))) { const key = i + "," + j + "," + k; if (!seen.has(key)) { seen.add(key); out.push([s[0], s[1], x[i], y[j], z[k]]); } }
  };
  const gx = (i, j, k) => (i < 0 || i >= nx - 1) ? -1 : ex[(i * ny + j) * nz + k];
  const gy = (i, j, k) => (j < 0 || j >= ny - 1) ? -1 : ey[(i * (ny - 1) + j) * nz + k];
  const gz = (i, j, k) => (k < 0 || k >= nz - 1) ? -1 : ez[(i * ny + j) * (nz - 1) + k];
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    const a = gx(i - 1, j, k), b = gx(i, j, k), c = gy(i, j - 1, k), d = gy(i, j, k), e = gz(i, j, k - 1), f = gz(i, j, k);
    if (a < 0 && b < 0 && c < 0 && d < 0 && e < 0 && f < 0) continue;
    consider(i, j, k, [a, b, c, d, e, f]);
  }
  return out;
}
