// Web worker: parsing, board model, stats, mesh, edge assignment, merge check, thermal and DC.
// The main thread keeps the drawing and the FDTD engines (WebGPU lives there).
import { readKicadPcb } from "./kicad.js";
import { Board, PORT_W, formatStats } from "./geom.js";
import { buildMesh, cflDt } from "./mesh.js";
import { assign, mergeCheck } from "./voxel.js";
import { Grid, solveDC, solveThermal, report, layerSlices } from "./thermal.js";
import { Model, CpuEngine, run, gaussPulse, sParams } from "./fdtd.js";

let geometry = null, bd = null, lastMesh = null, stopFlag = false;

function boardView() {
  const layers = bd.layers.map(layer => ({ name: layer, z: bd.z[layer], polys: bd.polys(layer).map(P => ({ net: P.net, kind: P.kind, ext: P.outline, holes: P.holes })) }));
  const s = bd.stats();
  return {
    w: bd.w, h: bd.h, zBot: bd.zBot, diel: bd.diel, refNet: bd.refNet, stackupSource: bd.stackupSource, outline: bd.outline.outline, layers,
    vias: bd.vias(), pads: bd.pads.map(p => { const [x, y] = bd.pt(p.x, p.y); return { ref: p.ref, num: p.num, net: p.net, layers: p.layers, x, y }; }),
    nets: geometry.nets, footprints: geometry.footprints || [], stats: s, statsText: formatStats(s),
  };
}

self.onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === "load") {
      const t0 = performance.now();
      geometry = m.geometry || readKicadPcb(m.text);
      bd = new Board(geometry, m.ref || null);
      const view = boardView();
      post({ id: m.id, ok: true, view, ms: performance.now() - t0 });
    } else if (m.type === "ref") {
      bd = new Board(geometry, m.ref || null);
      post({ id: m.id, ok: true, view: boardView() });
    } else if (m.type === "setup") {
      const t0 = performance.now();
      const tie = {}; for (const n of m.tie || []) tie[n] = bd.refNet;
      const nets = m.nets;
      const window = bd.netBBox(nets, m.margin ?? 2.5);
      const sites = [], boxes = [], log = [];
      for (const rp of m.ports) {
        const [ref, num] = rp.split(".");
        const s = bd.portSite(ref, num, tie);
        if (!s) throw new Error(`no clean port site for ${rp} (pad on ${bd.pad(ref, num)[0].net}, reference ${bd.refNet})`);
        s.name = rp;
        const z1 = Math.min(bd.z[s.plane], bd.z[s.sig]), z2 = Math.max(bd.z[s.plane], bd.z[s.sig]);
        s.z0 = z1; s.z1 = z2;
        sites.push(s); boxes.push([[s.x - PORT_W / 2, s.y - PORT_W / 2, z1], [s.x + PORT_W / 2, s.y + PORT_W / 2, z2]]);
        log.push(`port ${rp} at (${s.x.toFixed(2)}, ${s.y.toFixed(2)}) ${s.sig} over ${s.plane} (${s.ref})${s.shift ? ` shifted ${s.shift.toFixed(1)} mm` : ""}`);
      }
      const lines = buildMesh(bd, nets, window, { res: m.res, base: m.base, tie, portBoxes: boxes });
      const vox = assign(bd, lines, tie, sites.map(s => [s.x, s.y]), window);
      vox.ex = vox.ex.slice(); vox.ey = vox.ey.slice(); vox.ez = vox.ez.slice();      // (copies go to the page)
      const shorts = mergeCheck(vox, lines, nets, tie);
      const [dt, dmin] = cflDt(lines);
      const cells = lines.x.length * lines.y.length * lines.z.length;
      const nx = lines.x.length, ny = lines.y.length, nz = lines.z.length;
      const assignView = bd.layers.map(layer => { let k = 0; for (let q = 1; q < nz; q++) if (Math.abs(lines.z[q] - bd.z[layer]) < Math.abs(lines.z[k] - bd.z[layer])) k = q;
        const ex = new Int8Array((nx - 1) * ny), ey = new Int8Array(nx * (ny - 1));
        for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) ex[i * ny + j] = vox.ex[(i * ny + j) * nz + k];
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) ey[i * (ny - 1) + j] = vox.ey[(i * (ny - 1) + j) * nz + k];
        return { layer, k, ex, ey }; });
      log.push(`mesh ${nx} x ${ny} x ${nz} = ${(cells / 1e6).toFixed(2)} Mcells (${((performance.now() - t0) / 1000).toFixed(1)} s)  smallest cells ${dmin.map(v => (v * 1e3).toFixed(3)).join("/")} mm  dt ${(dt * 1e15).toFixed(0)} fs  vias ${vox.nVia}`);
      log.push(shorts.length ? `MERGE CHECK: ${shorts.length} node(s) join the pair to another conductor` : "MERGE CHECK: clean");
      lastMesh = { lines, vox };
      const setup = { nets, ports: m.ports, tie: m.tie || [], sites, boxes, window, lines: { x: lines.x, y: lines.y, z: lines.z }, cells, dt, dmin, shorts: shorts.slice(0, 50), nShorts: shorts.length,
        nVia: vox.nVia, names: vox.names, assign: assignView, log, diel: bd.diel, fmax: m.fmax, z0: m.z0 };
      // the vox arrays are copied (not transferred) so the worker keeps them for a CPU run
      self.postMessage({ id: m.id, ok: true, setup, vox: { ex: vox.ex, ey: vox.ey, ez: vox.ez, nx, ny, nz } }, assignView.flatMap(a => [a.ex.buffer, a.ey.buffer]));
    } else if (m.type === "stop") {
      stopFlag = true;
    } else if (m.type === "runCpu") {
      // the JS reference engine, in the worker so the page stays responsive
      stopFlag = false;
      const { lines, vox } = lastMesh;
      const erOfZ = z => { for (const [zt, zb, er] of bd.diel) if (zb <= z && z <= zt) return er; return 1.0; };
      const M = new Model(lines, vox, erOfZ);
      const pair = m.nets.length === 2, excOf = pair ? [1, 0, -1, 0] : [1, 0, 0, 0];
      m.boxes.forEach(([s, e], i) => M.addPort(s[0], e[0], s[1], e[1], s[2], e[2], m.z0, i < excOf.length ? excOf[i] : 0));
      const engine = new CpuEngine(M); await engine.init();
      const exc = gaussPulse(m.fmax / 2, m.fmax / 2);
      const nsteps = Math.floor(m.tmax / M.dt);
      let lastPost = 0;
      const r = await run(M, engine, exc.f, nsteps, { end: m.end, snapEvery: m.snapEvery, snapK: m.snapK, batch: 10,
        onSnap: (n, data) => post({ snap: { n, t: n * M.dt, data } }),
        onProgress: p => { const now = performance.now(); if (now - lastPost > 300) { lastPost = now; post({ runProgress: p }); } },
        shouldStop: () => stopFlag });
      const f = []; for (let q = 0; q < 300; q++) f.push(100e6 + (m.fmax - 100e6) * q / 299);
      const S = sParams(M.spectra(f), pair);
      post({ id: m.id, ok: true, result: { f, S11_dB: S.S11_dB, S21_dB: S.S21_dB, Z: S.Z_re, pair, steps: r.steps, wall: r.wall, dt: M.dt } });
    } else if (m.type === "thermal") {
      const t0 = performance.now();
      const g = new Grid(bd, m.cell || 0.25);
      let dc = null;
      const d = m.dc || {};
      if (d.net && d.src && d.sink && d.current) dc = solveDC(g, d.net, d.src.split("."), d.sink.split("."), +d.current, (it, r) => post({ progress: `DC CG iteration ${it}, residual ${r.toExponential(1)}` }));
      const powers = {}; for (const [k, v] of Object.entries(m.powers || {})) if (+v > 0) powers[k] = +v;
      const th = solveThermal(g, powers, { h: m.h ?? 10, tamb: m.tamb ?? 25, extraQ: dc ? dc.joule : null, onIter: (it, r) => post({ progress: `thermal CG iteration ${it}, residual ${r.toExponential(1)}` }) });
      const out = { cell: g.cell, x: g.x, y: g.y, layers: Object.keys(g.kCu), h: th.h, tamb: th.tamb, P_total: th.P_total, Tmax: th.Tmax, report: report(g, th, dc), T: layerSlices(g, th.T), ms: performance.now() - t0 };
      if (dc) out.dc = { net: dc.net, current: dc.current, drop_V: dc.drop_V, R_ohm: dc.R_ohm, joule_W: dc.joule_W, Jmax: dc.Jmax, V: layerSlices(g, dc.V, -1), J: layerSlices(g, dc.J) };
      post({ id: m.id, ok: true, thermal: out });
    }
  } catch (err) {
    post({ id: m.id, ok: false, error: err.message || String(err) });
  }
};
function post(o) { self.postMessage(o); }
