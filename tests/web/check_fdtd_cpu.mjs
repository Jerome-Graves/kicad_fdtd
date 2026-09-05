// The browser CPU engine on the microstrip milestone (same synthetic board as tests/test_msl.py):
// expect S21 -0.06 dB, S11 -18.6 dB at 1 GHz, Zin median 42 ohm, 9801 steps.
//   node tests/web/check_fdtd_cpu.mjs
import { Board, PORT_W } from "../../docs/app/geom.js";
import { buildMesh, cflDt } from "../../docs/app/mesh.js";
import { assign, mergeCheck } from "../../docs/app/voxel.js";
import { Model, CpuEngine, run, gaussPulse, sParams } from "../../docs/app/fdtd.js";
import { performance } from "node:perf_hooks";
globalThis.performance = performance;

const L = 12.0, W = 0.2, H = 0.1, x0 = 100.0, y0 = 50.0;
const rect = (xa, ya, xb, yb) => [[xa, ya], [xb, ya], [xb, yb], [xa, yb]];
const g = {
  layers: ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"], bbox: [x0, y0, 20.0, 8.0], outline: rect(x0, y0, x0 + 20, y0 + 8),
  stackup: [{ name: "F.Cu", type: "copper", thickness_mm: 0.035 }, { name: "d1", type: "prepreg", thickness_mm: H, er: 4.5, tand: 0.02 }, { name: "In1.Cu", type: "copper", thickness_mm: 0.035 },
    { name: "d2", type: "core", thickness_mm: 1.24, er: 4.5, tand: 0.02 }, { name: "In2.Cu", type: "copper", thickness_mm: 0.035 }, { name: "d3", type: "prepreg", thickness_mm: 0.1, er: 4.5, tand: 0.02 }, { name: "B.Cu", type: "copper", thickness_mm: 0.035 }],
  copper: { "F.Cu": [{ net: "/T", kind: "track", outline: rect(x0 + 4, y0 + 4 - W / 2, x0 + 4 + L, y0 + 4 + W / 2), holes: [] }],
    "In1.Cu": [{ net: "GND", kind: "zone", outline: rect(x0, y0, x0 + 20, y0 + 8), holes: [] }], "In2.Cu": [], "B.Cu": [] },
  vias: [], pads: [{ ref: "P", num: "1", net: "/T", x: x0 + 4.3, y: y0 + 4, layers: ["F.Cu"] }, { ref: "P", num: "2", net: "/T", x: x0 + 4 + L - 0.3, y: y0 + 4, layers: ["F.Cu"] }], nets: ["/T", "GND"], footprints: [],
};
const bd = new Board(g);
const nets = ["/T"], window = [0.0, -8.0, 20.0, 0.0];
const sites = ["1", "2"].map(n => bd.portSite("P", n));
const boxes = sites.map(s => [[s.x - PORT_W / 2, s.y - PORT_W / 2, bd.z[s.plane]], [s.x + PORT_W / 2, s.y + PORT_W / 2, bd.z[s.sig]]]);
const lines = buildMesh(bd, nets, window, { res: 0.1, base: 0.2, portBoxes: boxes });
const vox = assign(bd, lines, {}, [], window);
const shorts = mergeCheck(vox, lines, nets);
console.log(`mesh ${lines.x.length} x ${lines.y.length} x ${lines.z.length}, conductors ${JSON.stringify(vox.names)}, shorts ${shorts.length}`);
const erOfZ = z => { for (const [zt, zb, er] of bd.diel) if (zb <= z && z <= zt) return er; return 1.0; };
const M = new Model(lines, vox, erOfZ);
console.log("dt %s fs", (M.dt * 1e15).toFixed(1));
boxes.forEach(([s, e], i) => M.addPort(s[0], e[0], s[1], e[1], s[2], e[2], 50.0, i === 0 ? 1.0 : 0.0));
const eng = new CpuEngine(M); await eng.init();
const exc = gaussPulse(1.5e9, 1.5e9);
const nsteps = Math.floor(12e-9 / M.dt);
const r = await run(M, eng, exc.f, nsteps, { end: 1e-3, onProgress: p => { if (p.step % 2000 === 0) console.log(`step ${p.step} energy ${p.energyDb.toFixed(1)} dB ${p.elapsed.toFixed(0)} s`); } });
console.log(`done: ${r.steps} steps in ${r.wall.toFixed(0)} s = ${(r.steps * lines.x.length * lines.y.length * lines.z.length / r.wall / 1e6).toFixed(0)} Mcell-updates/s`);
const f = []; for (let q = 0; q < 100; q++) f.push(200e6 + (3e9 - 200e6) * q / 99);
const S = sParams(M.spectra(f), false);
const i1 = f.reduce((b, v, i) => Math.abs(v - 1e9) < Math.abs(f[b] - 1e9) ? i : b, 0);
const sel = f.map((v, i) => i).filter(i => f[i] > 300e6 && f[i] < 2e9).map(i => S.Z_re[i]).sort((a, b) => a - b);
console.log(`RESULT: S21 ${S.S21_dB[i1].toFixed(2)} dB  S11 ${S.S11_dB[i1].toFixed(1)} dB at 1 GHz | Zin median ${sel[sel.length >> 1].toFixed(0)} ohm | Python: -0.06 / -18.6 / 42, 9801 steps`);
