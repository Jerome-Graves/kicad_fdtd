// Compare the browser .kicad_pcb reader with the Python exporter's JSON (pcbnew ground truth).
//   node tests/web/check_parser.mjs out/boards/BugBot_Motion_Board.kicad_pcb out/BugBot_Motion_Board_geometry.json
import fs from "node:fs";
import { readKicadPcb } from "../../docs/app/kicad.js";

const [pcb, ref] = process.argv.slice(2);
const t0 = Date.now();
const g = readKicadPcb(fs.readFileSync(pcb, "utf8"));
const R = JSON.parse(fs.readFileSync(ref, "utf8"));
console.log("parsed in", Date.now() - t0, "ms");
console.log("layers", JSON.stringify(g.layers), "==", JSON.stringify(R.layers));
console.log("bbox  ", g.bbox.map(v => v.toFixed(3)).join(" "), "==", R.bbox.map(v => v.toFixed(3)).join(" "));
console.log("outline points", g.outline.length, "ref", R.outline.length);
console.log("stackup", g.stackup.length, "ref", R.stackup.length, JSON.stringify(g.stackup.filter(s => s.type === "copper" || s.thickness_mm).map(s => [s.name, s.thickness_mm, s.er])));
for (const l of R.layers) {
  const a = g.copper[l] || [], b = R.copper[l];
  const cnt = arr => { const k = {}; for (const p of arr) k[p.kind] = (k[p.kind] || 0) + 1; return JSON.stringify(k); };
  console.log("copper", l, cnt(a), "ref", cnt(b));
}
console.log("vias", g.vias.length, "ref", R.vias.length, " pth", g.vias.filter(v => v.kind === "pth").length, "ref", R.vias.filter(v => v.kind === "pth").length);
console.log("pads", g.pads.length, "ref", R.pads.length, " footprints", g.footprints.length, "ref", R.footprints.length, " nets", g.nets.length, "ref", R.nets.length);
// pad positions: match by ref+num
let worst = 0, bad = 0, layersMismatch = 0, netMismatch = 0;
const key = p => p.ref + "." + p.num;
const mine = new Map(g.pads.map(p => [key(p), p]));
for (const p of R.pads) {
  const q = mine.get(key(p));
  if (!q) { bad++; if (bad < 5) console.log("  missing pad", key(p)); continue; }
  const d = Math.hypot(p.x - q.x, p.y - q.y);
  if (d > worst) worst = d;
  if (d > 0.01 && bad < 8) { console.log("  pad", key(p), "off by", d.toFixed(3), "mm", p.x, p.y, "vs", q.x, q.y); bad++; }
  if (JSON.stringify([...p.layers].sort()) !== JSON.stringify([...q.layers].sort())) layersMismatch++;
  if (p.net !== q.net) { netMismatch++; if (netMismatch < 4) console.log("  net", key(p), JSON.stringify(p.net), "vs", JSON.stringify(q.net)); }
}
console.log("pad position worst", worst.toFixed(4), "mm; layer mismatches", layersMismatch, "; net mismatches", netMismatch);
// via positions
const vk = v => v.x.toFixed(2) + "," + v.y.toFixed(2);
const vs = new Set(g.vias.map(vk));
console.log("ref vias not found", R.vias.filter(v => !vs.has(vk(v))).length);
// copper area proxy per layer: sum of polygon areas (overlaps double count, same both sides)
const area = pts => { let a = 0; for (let i = 0; i < pts.length; i++) { const [x0, y0] = pts[i], [x1, y1] = pts[(i + 1) % pts.length]; a += x0 * y1 - x1 * y0; } return Math.abs(a) / 2; };
for (const l of R.layers) {
  const s = arr => arr.reduce((t, p) => t + area(p.outline) - p.holes.reduce((h, q) => h + area(q), 0), 0);
  console.log("area", l, s(g.copper[l] || []).toFixed(1), "ref", s(R.copper[l]).toFixed(1));
}
