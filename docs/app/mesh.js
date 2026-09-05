// Rectilinear Yee mesh for a board slice (port of kicad_fdtd/mesh.py).
// Lines on the copper edges of the nets of interest and of every neighbour within `near`
// of them, plus a base grid; graded air cells outward (the outer ones are the PML).
// z: every copper plane exactly, n cells per dielectric, graded air.
import { distPointPoly, polyBBox } from "./geom.js";

export const C0 = 299792458.0;

export function thin(lines, minGap) {
  const out = [lines[0]];
  for (let i = 1; i < lines.length; i++) if (lines[i] - out[out.length - 1] >= minGap) out.push(lines[i]);
  if (out[out.length - 1] !== lines[lines.length - 1]) out[out.length - 1] = lines[lines.length - 1];
  return out;
}
export function smooth(lines, maxRes) {
  let L = lines.slice(), changed = true;
  while (changed) {
    changed = false;
    const out = [L[0]];
    for (let i = 1; i < L.length; i++) {
      const v = L[i], gap = v - out[out.length - 1];
      if (gap > maxRes * 1.001) {
        const n = Math.ceil(gap / maxRes), a = out[out.length - 1];
        for (let k = 1; k <= n; k++) out.push(a + (v - a) * k / n);
        changed = true;
      } else out.push(v);
    }
    L = out;
  }
  return L;
}
export function airLines(edge, direction, first, n, grow) {
  const out = []; let step = first, pos = edge;
  for (let k = 0; k < n; k++) { pos += direction * step; out.push(pos); step *= grow; }
  return out;
}
// sorted, with values closer than 1 nm merged (floating-point twins like 18.9 and 18.900000000000002)
const uniqSorted = arr => { const s = Array.from(arr); s.sort((a, b) => a - b); const o = []; for (const v of s) { if (!o.length || v - o[o.length - 1] > 1e-9) o.push(v); } return o; };

export function buildMesh(bd, nets, window, { res = 0.1, base = 0.2, near = 0.6, tie = {}, airCells = 14, grow = 1.25, zCell = 0.1, portBoxes = [] } = {}) {
  const [xmin, ymin, xmax, ymax] = window;
  const xl = new Set([xmin, xmax]), yl = new Set([ymin, ymax]);
  const addPt = (x, y) => {
    if (x >= xmin && x <= xmax) xl.add(Math.round(x / 0.01) * 0.01);
    if (y >= ymin && y <= ymax) yl.add(Math.round(y / 0.01) * 0.01);
  };
  const inWin = b => !(b[2] < xmin || b[0] > xmax || b[3] < ymin || b[1] > ymax);
  const mine = [];                       // polygons of the nets of interest (vias as squares-ish circles)
  for (const layer of bd.layers) for (const P of bd.polys(layer)) if (nets.includes(P.net) && inWin(P.bbox)) { mine.push(P); for (const [x, y] of P.outline) addPt(x, y); }
  for (const v of bd.vias()) if (nets.includes(v.net) && v.x >= xmin && v.x <= xmax && v.y >= ymin && v.y <= ymax) {
    const c = []; for (let k = 0; k < 24; k++) c.push([v.x + v.r * Math.cos(2 * Math.PI * k / 24), v.y + v.r * Math.sin(2 * Math.PI * k / 24)]);
    mine.push({ outline: c, holes: [], bbox: polyBBox(c) });
    addPt(v.x - v.r, v.y - v.r); addPt(v.x + v.r, v.y + v.r); addPt(v.x, v.y);
  }
  if (mine.length) {
    // bounding box of the neighbourhood
    let nb = [Infinity, Infinity, -Infinity, -Infinity];
    for (const P of mine) nb = [Math.min(nb[0], P.bbox[0] - near), Math.min(nb[1], P.bbox[1] - near), Math.max(nb[2], P.bbox[2] + near), Math.max(nb[3], P.bbox[3] + near)];
    nb = [Math.max(nb[0], xmin), Math.max(nb[1], ymin), Math.min(nb[2], xmax), Math.min(nb[3], ymax)];
    const nearMine = (x, y) => {
      if (x < nb[0] || x > nb[2] || y < nb[1] || y > nb[3]) return false;
      for (const P of mine) { const b = P.bbox; if (x < b[0] - near || x > b[2] + near || y < b[1] - near || y > b[3] + near) continue; if (distPointPoly(P, x, y) <= near) return true; }
      return false;
    };
    const step = res * 0.7;
    for (const layer of bd.layers) for (const P of bd.polys(layer)) {
      if (nets.includes(P.net) || !inWin(P.bbox)) continue;
      const b = P.bbox;
      if (b[2] < nb[0] || b[0] > nb[2] || b[3] < nb[1] || b[1] > nb[3]) continue;
      for (const ring of [P.outline, ...P.holes]) for (let i = 0, n = ring.length; i < n; i++) {
        const a = ring[i], c = ring[(i + 1) % n];
        const L = Math.hypot(c[0] - a[0], c[1] - a[1]);
        const m = Math.max(1, Math.ceil(L / step));
        for (let k = 0; k <= m; k++) { const x = a[0] + (c[0] - a[0]) * k / m, y = a[1] + (c[1] - a[1]) * k / m; if (nearMine(x, y)) addPt(x, y); }
      }
    }
    for (const v of bd.vias()) if (!nets.includes(v.net)) {
      let close = false;
      for (let k = 0; k < 12 && !close; k++) close = nearMine(v.x + v.r * Math.cos(k * Math.PI / 6), v.y + v.r * Math.sin(k * Math.PI / 6));
      if (close) { addPt(v.x - v.r, v.y - v.r); addPt(v.x + v.r, v.y + v.r); }
    }
  }
  for (const [s, e] of portBoxes) { addPt(s[0], s[1]); addPt(e[0], e[1]); }
  const lines = {};
  for (const [name, Lset, lo, hi] of [["x", xl, xmin, xmax], ["y", yl, ymin, ymax]]) {
    const fixed = thin(uniqSorted([...Lset]), res * 0.4);
    const grid = [];
    for (let gv = lo; gv <= hi + base / 2 + 1e-12; gv += base) { let dmin = Infinity; for (const f of fixed) dmin = Math.min(dmin, Math.abs(gv - f)); if (dmin > res * 0.5) grid.push(gv); }
    let L2 = smooth(uniqSorted([...fixed, ...grid]), base);
    L2 = uniqSorted([...L2, ...airLines(lo, -1, res, airCells, grow), ...airLines(hi, +1, res, airCells, grow)]);
    lines[name] = Float64Array.from(L2);
  }
  let zl = [...Object.values(bd.z), bd.zBot];
  for (const [zt, zb] of bd.diel) { const n = Math.max(1, Math.round((zt - zb) / zCell)); for (let k = 0; k <= n; k++) zl.push(zb + (zt - zb) * k / n); }
  zl = uniqSorted(zl.map(v => Math.round(v * 1e6) / 1e6));
  zl = smooth(zl, 0.4);
  for (const zc of [...Object.values(bd.z), bd.zBot]) { let bi = 0; for (let i = 1; i < zl.length; i++) if (Math.abs(zl[i] - zc) < Math.abs(zl[bi] - zc)) bi = i; zl[bi] = zc; }
  zl = uniqSorted([...zl, ...airLines(0.0, +1, zCell, airCells, grow), ...airLines(bd.zBot, -1, zCell, airCells, grow)]);
  lines.z = Float64Array.from(zl);
  return lines;
}

export function cflDt(lines, safety = 0.99) {
  const d = ["x", "y", "z"].map(k => { let m = Infinity; const L = lines[k]; for (let i = 1; i < L.length; i++) m = Math.min(m, L[i] - L[i - 1]); return m * 1e-3; });
  return [safety / (C0 * Math.sqrt(d.reduce((s, v) => s + 1 / (v * v), 0))), d];
}
