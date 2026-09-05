// Time-domain energy balance with the PML on: W_port = W_field + integral of the Poynting flux
// through the far-field box (nothing absorbs between the box and the PML).
//   node tests/web/check_flux.mjs [res] [base] [gap] [steps]
import fs from "node:fs";
import { performance } from "node:perf_hooks";
globalThis.performance = performance;
import { readKicadPcb } from "../../docs/app/kicad.js";
import { Board, PORT_W, pointInPoly } from "../../docs/app/geom.js";
import { buildMesh } from "../../docs/app/mesh.js";
import { assign } from "../../docs/app/voxel.js";
import { Model, CpuEngine, gaussPulse } from "../../docs/app/fdtd.js";
import { nf2ffSurface } from "../../docs/app/nf2ff.js";

const [res = 1, base = 2, gap = 5, steps = 12000] = process.argv.slice(2).map(Number);
const bd = new Board(readKicadPcb(fs.readFileSync("docs/examples/patch.kicad_pcb", "utf8")));
const nets = ["/ANT"], win = [-2.5, -bd.h - 2.5, bd.w + 2.5, 2.5];
const s = bd.portSite("P1", "1");
const boxes = [[[s.x - PORT_W / 2, s.y - PORT_W / 2, Math.min(bd.z[s.plane], bd.z[s.sig])], [s.x + PORT_W / 2, s.y + PORT_W / 2, Math.max(bd.z[s.plane], bd.z[s.sig])]]];
const L = buildMesh(bd, nets, win, { res, base, portBoxes: boxes, airGap: gap, airGapCell: 1 });
const vox = assign(bd, L, {}, [[s.x, s.y]], win);
const erOfZ = z => { for (const [zt, zb, er] of bd.diel) if (zb <= z && z <= zt) return er; return 1.0; };
const M = new Model(L, vox, erOfZ, { insideXY: (x, y) => pointInPoly(bd.outline, x, y) });
M.addPort(boxes[0][0][0], boxes[0][1][0], boxes[0][0][1], boxes[0][1][1], boxes[0][0][2], boxes[0][1][2], 50, 1);
const surf = nf2ffSurface(M, { spacing: 1e-3, inset: L.gapInset });
console.log(`mesh ${L.x.length} x ${L.y.length} x ${L.z.length}, dt ${(M.dt * 1e15).toFixed(0)} fs, box ${surf.npts} points at inset ${L.gapInset}`);
const eng = new CpuEngine(M); await eng.init();
const exc = gaussPulse(1.5e9, 1.5e9);
const N = steps, excArr = new Float32Array(N); for (let n = 0; n < N; n++) excArr[n] = exc.f(n * M.dt);
const VI = new Float32Array(2 * N);
let Wport = 0, Wflux = 0;
console.log("step   t(ns)    Wport         Wfield        Wflux         (Wfield+Wflux)/Wport");
for (let n = 0; n < N; n++) {
  await eng.runSteps(n, 1, excArr, VI, N);
  Wport += VI[n] * VI[N + n] * M.dt;
  Wflux += eng.boxFlux(surf) * M.dt;
  if ((n + 1) % 1000 === 0) { const W = eng.physEnergy(); console.log(`${String(n + 1).padStart(5)}  ${((n + 1) * M.dt * 1e9).toFixed(2)}   ${Wport.toExponential(3)}   ${W.W.toExponential(3)}   ${Wflux.toExponential(3)}   ${((W.W + Wflux) / Wport).toFixed(3)}`); }
}
