// Mesh + edge assignment + merge check from the browser modules, to compare with
// `kicad-fdtd pair ... --setup-only` (Motion USB: mesh 357 x 332 x 43, 74 vias, clean at 0.15/0.3).
//   node tests/web/check_mesh.mjs out/boards/BugBot_Motion_Board.kicad_pcb /USB_DP /USB_DM J1.A6 J9.10 J1.A7 J9.9 [res base]
import fs from "node:fs";
import { readKicadPcb } from "../../docs/app/kicad.js";
import { Board, PORT_W } from "../../docs/app/geom.js";
import { buildMesh, cflDt } from "../../docs/app/mesh.js";
import { assign, mergeCheck } from "../../docs/app/voxel.js";

const a = process.argv.slice(2);
const pcb = a[0], nets = a.slice(1, 3), ports = a.slice(3, 7), res = +(a[7] || 0.15), base = +(a[8] || 0.3);
const bd = new Board(readKicadPcb(fs.readFileSync(pcb, "utf8")));
const window = bd.netBBox(nets, 2.5);
const boxes = [], sites = [];
for (const rp of ports) {
  const [ref, num] = rp.split(".");
  const s = bd.portSite(ref, num);
  if (!s) throw new Error("no port site for " + rp);
  sites.push(s);
  const z1 = Math.min(bd.z[s.plane], bd.z[s.sig]), z2 = Math.max(bd.z[s.plane], bd.z[s.sig]);
  boxes.push([[s.x - PORT_W / 2, s.y - PORT_W / 2, z1], [s.x + PORT_W / 2, s.y + PORT_W / 2, z2]]);
}
let t0 = Date.now();
const lines = buildMesh(bd, nets, window, { res, base, portBoxes: boxes });
const tm = Date.now() - t0; t0 = Date.now();
const vox = assign(bd, lines, {}, sites.map(s => [s.x, s.y]), window);
const tv = Date.now() - t0; t0 = Date.now();
const shorts = mergeCheck(vox, lines, nets);
const [dt, dmin] = cflDt(lines);
console.log(`mesh ${lines.x.length} x ${lines.y.length} x ${lines.z.length} = ${(lines.x.length * lines.y.length * lines.z.length / 1e6).toFixed(2)} Mcells (mesh ${tm} ms, assign ${tv} ms, check ${Date.now() - t0} ms)  smallest ${dmin.map(v => (v * 1e3).toFixed(3)).join("/")} mm  dt ${(dt * 1e15).toFixed(0)} fs  vias ${vox.nVia}  conductors ${vox.names.length}`);
console.log(shorts.length ? `MERGE CHECK: ${shorts.length} shorts, e.g. ${JSON.stringify(shorts.slice(0, 3))}` : "MERGE CHECK: clean");
fs.mkdirSync("out", { recursive: true });
fs.writeFileSync("out/web_mesh_lines.json", JSON.stringify({ x: [...lines.x], y: [...lines.y], z: [...lines.z] }));
