// Energy bookkeeping of the lumped port: closed PEC box (no PML), CPU engine, patch example.
// The energy the port reports delivering (sum V I dt) must equal the field energy in the box
// plus what the port resistor has dissipated is NOT included... so compare against the energy
// balance: W_field(t) = sum V I dt  (nothing else can absorb inside a lossless closed box).
//   node tests/web/check_energy.mjs [res] [base] [gap] [steps]
import fs from "node:fs";
import { performance } from "node:perf_hooks";
globalThis.performance = performance;
import { readKicadPcb } from "../../docs/app/kicad.js";
import { Board, PORT_W, pointInPoly } from "../../docs/app/geom.js";
import { buildMesh } from "../../docs/app/mesh.js";
import { assign } from "../../docs/app/voxel.js";
import { Model, CpuEngine, gaussPulse } from "../../docs/app/fdtd.js";

const [res = 1, base = 2, gap = 3, steps = 6000] = process.argv.slice(2).map(Number);
const bd = new Board(readKicadPcb(fs.readFileSync("docs/examples/patch.kicad_pcb", "utf8")));
const nets = ["/ANT"], win = [-2.5, -bd.h - 2.5, bd.w + 2.5, 2.5];
const s = bd.portSite("P1", "1");
const boxes = [[[s.x - PORT_W / 2, s.y - PORT_W / 2, Math.min(bd.z[s.plane], bd.z[s.sig])], [s.x + PORT_W / 2, s.y + PORT_W / 2, Math.max(bd.z[s.plane], bd.z[s.sig])]]];
const L = buildMesh(bd, nets, win, { res, base, portBoxes: boxes, airGap: gap, airGapCell: 1, airCells: 4 });
const vox = assign(bd, L, {}, [[s.x, s.y]], win);
const erOfZ = z => { for (const [zt, zb, er] of bd.diel) if (zb <= z && z <= zt) return er; return 1.0; };
const M = new Model(L, vox, erOfZ, { npml: 0, insideXY: (x, y) => pointInPoly(bd.outline, x, y) });
M.addPort(boxes[0][0][0], boxes[0][1][0], boxes[0][0][1], boxes[0][1][1], boxes[0][0][2], boxes[0][1][2], 50, 1);
console.log(`mesh ${L.x.length} x ${L.y.length} x ${L.z.length}, dt ${(M.dt * 1e15).toFixed(0)} fs, PEC walls (no PML)`);
const eng = new CpuEngine(M); await eng.init();
const exc = gaussPulse(1.5e9, 1.5e9);
const N = steps, excArr = new Float32Array(N); for (let n = 0; n < N; n++) excArr[n] = exc.f(n * M.dt);
const VI = new Float32Array(2 * N);
let Wport = 0;
const chunk = 500;
console.log("step   t(ns)   Wport=sum(V I dt)   Wfield        ratio");
for (let n0 = 0; n0 < N; n0 += chunk) {
  await eng.runSteps(n0, Math.min(chunk, N - n0), excArr, VI, N);
  for (let n = n0; n < Math.min(n0 + chunk, N); n++) Wport += VI[n] * VI[N + n] * M.dt;
  const W = eng.physEnergy();
  console.log(`${String(n0 + chunk).padStart(5)}  ${((n0 + chunk) * M.dt * 1e9).toFixed(2)}   ${Wport.toExponential(4)}   ${W.W.toExponential(4)}   ${(W.W / Wport).toFixed(3)}   (We ${W.We.toExponential(2)} Wm ${W.Wm.toExponential(2)})`);
}
