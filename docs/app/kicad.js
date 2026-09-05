// .kicad_pcb reader: S-expression text -> the same geometry object the Python
// exporter (kicad_fdtd/export.py) writes, so every downstream module is shared.
//   { layers, bbox:[x,y,w,h], outline:[[x,y]..], stackup:[{name,type,thickness_mm,er,tand}],
//     copper:{layer:[{net,kind,outline,holes}]}, vias:[{x,y,drill,dia,top,bottom,net,kind}],
//     pads:[{ref,num,net,x,y,layers}], footprints:[{ref,value,x,y,layer}], nets:[...] }
// Coordinates in mm, KiCad's frame (y down). Pads, tracks and arcs become polygons here;
// zone fills are read as KiCad stored them (fractured, hole-free polygons).

const ARC_SEGS = 16;            // segments per quarter circle for round shapes

// ---------------------------------------------------------------- s-expressions
export function parseSexpr(text) {
  // returns nested arrays; atoms are strings (quoted strings unquoted, numbers kept as strings)
  const root = [];
  const stack = [root];
  const n = text.length;
  let i = 0;
  while (i < n) {
    const c = text.charCodeAt(i);
    if (c === 40) {                         // (
      const node = [];
      stack[stack.length - 1].push(node);
      stack.push(node);
      i++;
    } else if (c === 41) {                  // )
      stack.pop();
      i++;
    } else if (c === 34) {                  // "
      let j = i + 1, s = "";
      while (j < n) {
        const d = text.charCodeAt(j);
        if (d === 92) { s += text[j + 1]; j += 2; continue; }
        if (d === 34) break;
        s += text[j]; j++;
      }
      stack[stack.length - 1].push(s);
      i = j + 1;
    } else if (c <= 32) {
      i++;
    } else {
      let j = i + 1;
      while (j < n) { const d = text.charCodeAt(j); if (d <= 32 || d === 40 || d === 41) break; j++; }
      stack[stack.length - 1].push(text.slice(i, j));
      i = j;
    }
  }
  return root[0];
}

const num = v => (v === undefined ? 0 : parseFloat(v));
const child = (node, key) => node.find(x => Array.isArray(x) && x[0] === key);
const children = (node, key) => node.filter(x => Array.isArray(x) && x[0] === key);
const at = node => { const a = child(node, "at"); return a ? [num(a[1]), num(a[2]), num(a[3])] : [0, 0, 0]; };

// ---------------------------------------------------------------- geometry helpers
const rot = (x, y, deg) => {              // KiCad: positive angle = counter-clockwise on screen (y down)
  const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return [x * c + y * s, -x * s + y * c];
};
function circlePts(cx, cy, r, n = 4 * ARC_SEGS) {
  const out = [];
  for (let k = 0; k < n; k++) { const a = 2 * Math.PI * k / n; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
  return out;
}
function roundRect(w, h, r) {           // centred, corner radius r
  r = Math.min(r, w / 2, h / 2);
  if (r <= 1e-6) return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]];
  const out = [];
  const corners = [[w / 2 - r, h / 2 - r, 0], [-w / 2 + r, h / 2 - r, Math.PI / 2], [-w / 2 + r, -h / 2 + r, Math.PI], [w / 2 - r, -h / 2 + r, 3 * Math.PI / 2]];
  for (const [cx, cy, a0] of corners)
    for (let k = 0; k <= ARC_SEGS; k++) { const a = a0 + (Math.PI / 2) * k / ARC_SEGS; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
  return out;
}
function oval(w, h) {                    // stadium
  if (Math.abs(w - h) < 1e-9) return circlePts(0, 0, w / 2);
  return roundRect(w, h, Math.min(w, h) / 2);
}
function stroke(p0, p1, width) {         // segment with round caps -> polygon
  const [x0, y0] = p0, [x1, y1] = p1, r = width / 2;
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 1e-9) return circlePts(x0, y0, r);
  const a = Math.atan2(dy, dx), out = [];
  // cap at the end point sweeps the forward half, cap at the start point the backward half
  for (let k = 0; k <= 2 * ARC_SEGS; k++) { const t = a - Math.PI / 2 + Math.PI * k / (2 * ARC_SEGS); out.push([x1 + r * Math.cos(t), y1 + r * Math.sin(t)]); }
  for (let k = 0; k <= 2 * ARC_SEGS; k++) { const t = a + Math.PI / 2 + Math.PI * k / (2 * ARC_SEGS); out.push([x0 + r * Math.cos(t), y0 + r * Math.sin(t)]); }
  return out;
}
function arcCentre(s, m, e) {             // circle through three points
  const [ax, ay] = s, [bx, by] = m, [cx, cy] = e;
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-12) return null;
  const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
  const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
  return [ux, uy];
}
function arcPoints(s, m, e) {             // polyline along the arc start-mid-end
  const c = arcCentre(s, m, e);
  if (!c) return [s, e];
  const r = Math.hypot(s[0] - c[0], s[1] - c[1]);
  let a0 = Math.atan2(s[1] - c[1], s[0] - c[0]), am = Math.atan2(m[1] - c[1], m[0] - c[0]), a1 = Math.atan2(e[1] - c[1], e[0] - c[0]);
  // choose the sweep direction that passes through the mid point
  const norm = a => (a % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
  let ccw = norm(a1 - a0), mid = norm(am - a0);
  let sweep = mid <= ccw ? ccw : ccw - 2 * Math.PI;
  const n = Math.max(2, Math.ceil(Math.abs(sweep) / (Math.PI / 2) * ARC_SEGS));
  const out = [];
  for (let k = 0; k <= n; k++) { const a = a0 + sweep * k / n; out.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]); }
  return out;
}
function ptsOf(node) {                    // (pts (xy..) (arc (start)(mid)(end)) ...) -> [[x,y]...]
  const pts = child(node, "pts");
  if (!pts) return [];
  const out = [];
  for (const it of pts) {
    if (!Array.isArray(it)) continue;
    if (it[0] === "xy") out.push([num(it[1]), num(it[2])]);
    else if (it[0] === "arc") {
      const s = child(it, "start"), m = child(it, "mid"), e = child(it, "end");
      const ap = arcPoints([num(s[1]), num(s[2])], [num(m[1]), num(m[2])], [num(e[1]), num(e[2])]);
      out.push(...ap);
    }
  }
  return out;
}
const r4 = v => Math.round(v * 1e4) / 1e4;
const roundPts = pts => pts.map(p => [r4(p[0]), r4(p[1])]);

// ---------------------------------------------------------------- the reader
export function readKicadPcb(text) {
  const doc = parseSexpr(text);
  if (!doc || doc[0] !== "kicad_pcb") throw new Error("not a .kicad_pcb file");
  const version = child(doc, "version");
  // copper layers in stack order (the layers block lists them top to bottom)
  const layersNode = child(doc, "layers") || [];
  const CU = [];
  for (const L of layersNode) {
    if (!Array.isArray(L)) continue;
    const name = L[1], type = L[2];
    if (typeof name === "string" && name.endsWith(".Cu") && ["signal", "power", "mixed", "jumper"].includes(type)) CU.push(name);
  }
  const cuSet = new Set(CU);
  const copper = {}; for (const l of CU) copper[l] = [];
  const vias = [], pads = [], fps = [], nets = new Set();

  // nets: (net 1 "GND") declarations; items may refer to a net by number (KiCad <= 9) or by name (KiCad 10)
  const netByCode = {};
  for (const N of children(doc, "net")) { if (N[2] !== undefined) { nets.add(N[2]); netByCode[N[1]] = N[2]; } else if (typeof N[1] === "string" && isNaN(N[1])) nets.add(N[1]); }
  const netOf = node => {                 // (net 5 "GND") / (net 5) / (net "GND") / none
    const N = child(node, "net");
    if (!N) return "";
    if (N.length >= 3) return N[2];
    return isNaN(N[1]) ? N[1] : (netByCode[N[1]] || "");
  };
  const padLayers = (spec) => {           // ("F.Cu" "F.Mask") / ("*.Cu" ...) / ("F&B.Cu")
    const out = [];
    for (const s of spec) {
      if (typeof s !== "string") continue;
      if (s === "*.Cu" || s === "F&B.Cu") { out.push(...CU); }
      else if (s.endsWith(".Cu") && cuSet.has(s)) out.push(s);
    }
    return [...new Set(out)];
  };

  // --- footprints
  for (const F of children(doc, "footprint")) {
    const [fx, fy, frot] = at(F);
    const flayer = (child(F, "layer") || [])[1] || CU[0];
    let ref = "", value = "";
    for (const P of children(F, "property")) { if (P[1] === "Reference") ref = P[2]; if (P[1] === "Value") value = P[2]; }
    // KiCad 6/7 style
    for (const T of children(F, "fp_text")) { if (T[1] === "reference") ref = ref || T[2]; if (T[1] === "value") value = value || T[2]; }
    fps.push({ ref, value, x: r4(fx), y: r4(fy), layer: flayer });
    for (const P of children(F, "pad")) {
      const numStr = P[1], type = P[2], shape = P[3];
      const [dx, dy, pang] = at(P);
      const [rx, ry] = rot(dx, dy, frot);
      const px = fx + rx, py = fy + ry;
      const sz = child(P, "size"); const w = sz ? num(sz[1]) : 0, h = sz ? num(sz[2]) : 0;
      const net = netOf(P); if (net) nets.add(net);
      const lspec = child(P, "layers") || [];
      const on = padLayers(lspec.slice(1));
      pads.push({ ref, num: String(numStr), net, x: r4(px), y: r4(py), layers: on });
      if (!on.length) continue;
      // shape polygon at the origin, pad orientation, then to board
      let shp;
      if (shape === "circle") shp = circlePts(0, 0, w / 2);
      else if (shape === "oval") shp = oval(w, h);
      else if (shape === "roundrect") { const rr = child(P, "roundrect_rratio"); shp = roundRect(w, h, (rr ? num(rr[1]) : 0.25) * Math.min(w, h)); }
      else if (shape === "rect" || shape === "trapezoid") shp = roundRect(w, h, 0);
      else if (shape === "custom") {
        const prims = child(P, "primitives") || [];
        const polys = [];
        for (const G of prims) {
          if (!Array.isArray(G)) continue;
          if (G[0] === "gr_poly") polys.push(ptsOf(G));
          else if (G[0] === "gr_rect") { const s = child(G, "start"), e = child(G, "end"); polys.push([[num(s[1]), num(s[2])], [num(e[1]), num(s[2])], [num(e[1]), num(e[2])], [num(s[1]), num(e[2])]]); }
          else if (G[0] === "gr_circle") { const c = child(G, "center"), e = child(G, "end"); polys.push(circlePts(num(c[1]), num(c[2]), Math.hypot(num(e[1]) - num(c[1]), num(e[2]) - num(c[2])))); }
          else if (G[0] === "gr_line") { const s = child(G, "start"), e = child(G, "end"); const wd = child(G, "width"); polys.push(stroke([num(s[1]), num(s[2])], [num(e[1]), num(e[2])], wd ? num(wd[1]) : 0.1)); }
        }
        const anchor = (child(child(P, "options") || [], "anchor") || [])[1];
        polys.push(anchor === "rect" ? roundRect(w, h, 0) : circlePts(0, 0, w / 2));
        for (const poly of polys) {
          const placed = poly.map(([x, y]) => { const [qx, qy] = rot(x, y, pang); return [r4(px + qx), r4(py + qy)]; });
          for (const l of on) copper[l].push({ net, kind: "pad", outline: placed, holes: [] });
        }
        shp = null;
      } else shp = roundRect(w, h, 0);
      if (shp) {
        const placed = shp.map(([x, y]) => { const [qx, qy] = rot(x, y, pang); return [r4(px + qx), r4(py + qy)]; });
        for (const l of on) copper[l].push({ net, kind: "pad", outline: placed, holes: [] });
      }
      if (type === "thru_hole" || type === "np_thru_hole") {
        const D = child(P, "drill");
        let d = 0;
        if (D) { const vals = D.filter(v => typeof v === "string" && !isNaN(v)).map(Number); d = vals.length ? vals[0] : 0; }
        if (d > 0 && type === "thru_hole") vias.push({ x: r4(px), y: r4(py), drill: d, dia: r4(d + 0.05), top: CU[0], bottom: CU[CU.length - 1], net, kind: "pth" });
      }
    }
  }
  // --- tracks, arcs, vias
  for (const S of children(doc, "segment")) {
    const s = child(S, "start"), e = child(S, "end"), wd = num((child(S, "width") || [])[1]), l = (child(S, "layer") || [])[1];
    if (!cuSet.has(l)) continue;
    const net = netOf(S); if (net) nets.add(net);
    copper[l].push({ net, kind: "track", outline: roundPts(stroke([num(s[1]), num(s[2])], [num(e[1]), num(e[2])], wd)), holes: [] });
  }
  for (const A of children(doc, "arc")) {
    const s = child(A, "start"), m = child(A, "mid"), e = child(A, "end"), wd = num((child(A, "width") || [])[1]), l = (child(A, "layer") || [])[1];
    if (!cuSet.has(l)) continue;
    const net = netOf(A); if (net) nets.add(net);
    const pl = arcPoints([num(s[1]), num(s[2])], [num(m[1]), num(m[2])], [num(e[1]), num(e[2])]);
    for (let k = 0; k + 1 < pl.length; k++) copper[l].push({ net, kind: "track", outline: roundPts(stroke(pl[k], pl[k + 1], wd)), holes: [] });
  }
  for (const V of children(doc, "via")) {
    const [vx, vy] = at(V);
    const size = num((child(V, "size") || [])[1]), drill = num((child(V, "drill") || [])[1]);
    const ls = (child(V, "layers") || []).slice(1).filter(x => typeof x === "string");
    const net = netOf(V); if (net) nets.add(net);
    vias.push({ x: r4(vx), y: r4(vy), drill, dia: size, top: ls[0] || CU[0], bottom: ls[1] || CU[CU.length - 1], net, kind: "via" });
  }
  // --- zones (filled polygons as stored: fractured, no holes)
  for (const Z of children(doc, "zone")) {
    const net = (child(Z, "net_name") || [])[1] || netOf(Z); if (net) nets.add(net);
    for (const FP of children(Z, "filled_polygon")) {
      const l = (child(FP, "layer") || [])[1];
      if (!cuSet.has(l)) continue;
      const pts = roundPts(ptsOf(FP));
      if (pts.length >= 3) copper[l].push({ net, kind: "zone", outline: pts, holes: [] });
    }
  }
  // --- board edge: chain Edge.Cuts lines/arcs/rects/circles/polys
  const edges = [];
  for (const G of doc) {
    if (!Array.isArray(G)) continue;
    const l = (child(G, "layer") || [])[1];
    if (l !== "Edge.Cuts") continue;
    if (G[0] === "gr_line") { const s = child(G, "start"), e = child(G, "end"); edges.push([[num(s[1]), num(s[2])], [num(e[1]), num(e[2])]]); }
    else if (G[0] === "gr_arc") { const s = child(G, "start"), m = child(G, "mid"), e = child(G, "end"); const pl = arcPoints([num(s[1]), num(s[2])], [num(m[1]), num(m[2])], [num(e[1]), num(e[2])]); for (let k = 0; k + 1 < pl.length; k++) edges.push([pl[k], pl[k + 1]]); }
    else if (G[0] === "gr_rect") { const s = child(G, "start"), e = child(G, "end"); const a = [num(s[1]), num(s[2])], c = [num(e[1]), num(e[2])]; const b = [c[0], a[1]], d = [a[0], c[1]]; edges.push([a, b], [b, c], [c, d], [d, a]); }
    else if (G[0] === "gr_circle") { const c = child(G, "center"), e = child(G, "end"); const pl = circlePts(num(c[1]), num(c[2]), Math.hypot(num(e[1]) - num(c[1]), num(e[2]) - num(c[2]))); for (let k = 0; k < pl.length; k++) edges.push([pl[k], pl[(k + 1) % pl.length]]); }
    else if (G[0] === "gr_poly") { const pl = ptsOf(G); for (let k = 0; k < pl.length; k++) edges.push([pl[k], pl[(k + 1) % pl.length]]); }
  }
  let outline = chainEdges(edges);
  let bbox;
  if (outline.length >= 3) {
    const xs = outline.map(p => p[0]), ys = outline.map(p => p[1]);
    bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
  } else {                                 // no edge: bound the copper
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const l of CU) for (const P of copper[l]) for (const [x, y] of P.outline) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
    if (!isFinite(x0)) { x0 = y0 = 0; x1 = y1 = 10; }
    bbox = [x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2];
    outline = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]];
  }
  // --- stackup
  const stackup = [];
  const setup = child(doc, "setup");
  const st = setup && child(setup, "stackup");
  if (st) for (const L of children(st, "layer")) {
    const g = k => { const c = child(L, k); return c ? num(c[1]) : null; };
    stackup.push({ name: L[1], type: (child(L, "type") || [])[1], thickness_mm: g("thickness"), er: g("epsilon_r"), tand: g("loss_tangent") });
  }
  return { board: "browser", kicad: version ? "format " + version[1] : "?", layers: CU, bbox: bbox.map(r4), outline: roundPts(outline), stackup, copper, vias, pads, footprints: fps, nets: [...nets].sort() };
}

function chainEdges(edges) {
  if (!edges.length) return [];
  const used = new Array(edges.length).fill(false);
  const tol = 0.01;
  const same = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < tol;
  let best = [];
  for (let start = 0; start < edges.length; start++) {
    if (used[start]) continue;
    const loop = [edges[start][0], edges[start][1]];
    used[start] = true;
    let grown = true;
    while (grown) {
      grown = false;
      const tail = loop[loop.length - 1];
      for (let k = 0; k < edges.length; k++) {
        if (used[k]) continue;
        const [a, b] = edges[k];
        if (same(a, tail)) { loop.push(b); used[k] = true; grown = true; break; }
        if (same(b, tail)) { loop.push(a); used[k] = true; grown = true; break; }
      }
      if (same(loop[loop.length - 1], loop[0])) { loop.pop(); break; }
    }
    if (loop.length > best.length) best = loop;
  }
  return best;
}
