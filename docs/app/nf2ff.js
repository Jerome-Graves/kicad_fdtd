// Near-to-far-field transform on a Huygens box (frequency domain, equivalent currents).
//
// During the run the engines accumulate the DFT of the tangential E and H at sample points
// on the six faces of a box that sits in the air cells between the meshed region and the
// PML (nf2ffSurface). Afterwards farField() forms J = n x H, M = -n x E and the radiation
// vectors N, L for a sphere of directions, giving |E| at distance r for every frequency,
// per unit of the DFT of the source; emissions() scales that by the spectrum of a trapezoid
// clock and compares with CISPR 32 class B (peak detector, 3 m distance).
export const ETA0 = 376.730313668, C0 = 299792458.0;

// sample points on the box faces. Returns Int32Array pts (8 per point: e1 idx, e1 arr, e2 idx,
// e2 arr, h1 idx, h1 arr, h2 idx, h2 arr; arr codes 0..5 = Ex Ey Ez Hx Hy Hz) and geometry.
export function nf2ffSurface(M, { spacing = 0.5e-3, inset = null } = {}) {
  const { nx, ny, nz, x, y, z } = M;
  const npml = M.npml, ins = inset ?? (npml + 2);
  const i0 = ins, i1 = nx - 1 - ins, j0 = ins, j1 = ny - 1 - ins, k0 = ins, k1 = nz - 1 - ins;
  if (i1 - i0 < 4 || j1 - j0 < 4 || k1 - k0 < 4) throw new Error("mesh too small for a far-field box");
  const cx = (x[i0] + x[i1]) / 2, cy = (y[j0] + y[j1]) / 2, cz = (z[k0] + z[k1]) / 2;
  const EX = 0, EY = 1, EZ = 2, HX = 3, HY = 4, HZ = 5;
  const iEx = (i, j, k) => (i * ny + j) * nz + k, iEy = (i, j, k) => (i * (ny - 1) + j) * nz + k, iEz = (i, j, k) => (i * ny + j) * (nz - 1) + k;
  const iHx = (i, j, k) => (i * (ny - 1) + j) * (nz - 1) + k, iHy = (i, j, k) => (i * ny + j) * (nz - 1) + k, iHz = (i, j, k) => (i * (ny - 1) + j) * nz + k;
  const lower = (arr, v) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } return Math.max(0, lo - 1); };
  const pts = [], pos = [], nrm = [], area = [];
  const lattice = (a0, a1) => { const n = Math.max(1, Math.round((a1 - a0) / spacing)); const c = []; for (let q = 0; q < n; q++) c.push(a0 + (a1 - a0) * (q + 0.5) / n); return [c, (a1 - a0) / n]; };
  // x faces: tangential Ey, Ez on the face; Hy, Hz averaged from the cells on both sides of the
  // face (the near field is reactive: a half-cell offset between E and H spoils the real power).
  // 12 ints per point: e1 idx, arr, e2 idx, arr, h1 out idx, arr, h1 in idx, arr, h2 out, arr, h2 in, arr
  for (const [i, n, ho, hn] of [[i0, -1, i0 - 1, i0], [i1, +1, i1, i1 - 1]]) {
    const [ys, dy] = lattice(y[j0], y[j1]), [zs, dz] = lattice(z[k0], z[k1]);
    for (const yc of ys) for (const zc of zs) {
      const j = lower(y, yc), k = lower(z, zc);
      pts.push(iEy(i, j, k), EY, iEz(i, j, k), EZ, iHy(ho, j, k), HY, iHy(hn, j, k), HY, iHz(ho, j, k), HZ, iHz(hn, j, k), HZ);
      pos.push(x[i] - cx, yc - cy, zc - cz); nrm.push(n, 0, 0); area.push(dy * dz);
    }
  }
  for (const [j, n, ho, hn] of [[j0, -1, j0 - 1, j0], [j1, +1, j1, j1 - 1]]) {
    const [xs, dx] = lattice(x[i0], x[i1]), [zs, dz] = lattice(z[k0], z[k1]);
    for (const xc of xs) for (const zc of zs) {
      const i = lower(x, xc), k = lower(z, zc);
      pts.push(iEx(i, j, k), EX, iEz(i, j, k), EZ, iHx(i, ho, k), HX, iHx(i, hn, k), HX, iHz(i, ho, k), HZ, iHz(i, hn, k), HZ);
      pos.push(xc - cx, y[j] - cy, zc - cz); nrm.push(0, n, 0); area.push(dx * dz);
    }
  }
  for (const [k, n, ho, hn] of [[k0, -1, k0 - 1, k0], [k1, +1, k1, k1 - 1]]) {
    const [xs, dx] = lattice(x[i0], x[i1]), [ys, dy] = lattice(y[j0], y[j1]);
    for (const xc of xs) for (const yc of ys) {
      const i = lower(x, xc), j = lower(y, yc);
      pts.push(iEx(i, j, k), EX, iEy(i, j, k), EY, iHx(i, j, ho), HX, iHx(i, j, hn), HX, iHy(i, j, ho), HY, iHy(i, j, hn), HY);
      pos.push(xc - cx, yc - cy, z[k] - cz); nrm.push(0, 0, n); area.push(dx * dy);
    }
  }
  return { pts: Int32Array.from(pts), pos: Float64Array.from(pos), nrm: Int8Array.from(nrm), area: Float64Array.from(area), npts: area.length, box: [i0, i1, j0, j1, k0, k1] };
}

// CPU accumulation for one step (the GPU kernel does the same): acc[(p*nf+q)*8 + c]
export function accumulateCpu(surf, fields, acc, nf, f0, df, t) {
  const { pts, npts } = surf;
  for (let p = 0; p < npts; p++) {
    const b = p * 12;
    const e1 = fields[pts[b + 1]][pts[b]], e2 = fields[pts[b + 3]][pts[b + 2]];
    const h1 = 0.5 * (fields[pts[b + 5]][pts[b + 4]] + fields[pts[b + 7]][pts[b + 6]]), h2 = 0.5 * (fields[pts[b + 9]][pts[b + 8]] + fields[pts[b + 11]][pts[b + 10]]);
    for (let q = 0; q < nf; q++) {
      const w = 2 * Math.PI * (f0 + q * df) * t, c = Math.cos(w), s = -Math.sin(w), o = (p * nf + q) * 8;
      acc[o] += e1 * c; acc[o + 1] += e1 * s; acc[o + 2] += e2 * c; acc[o + 3] += e2 * s;
      acc[o + 4] += h1 * c; acc[o + 5] += h1 * s; acc[o + 6] += h2 * c; acc[o + 7] += h2 * s;
    }
  }
}

// far field from the accumulated surface DFTs. acc already scaled by dt. Returns per frequency:
// Emax (V/m at r, per the source's DFT), the direction of the max, and the radiated power Prad.
export function farField(surf, acc, nf, f0, df, { r = 3.0, dTheta = 15, dPhi = 15, dt = 0 } = {}) {
  const { pos, nrm, area, npts } = surf;
  // H was sampled half a time step before E (leapfrog): rotate its phasor by +omega dt / 2
  if (dt > 0) for (let q = 0; q < nf; q++) {
    const a = 2 * Math.PI * (f0 + q * df) * dt / 2, c = Math.cos(a), s = Math.sin(a);
    for (let p = 0; p < npts; p++) { const o = (p * nf + q) * 8; for (const c0 of [4, 6]) { const re = acc[o + c0], im = acc[o + c0 + 1]; acc[o + c0] = re * c - im * s; acc[o + c0 + 1] = re * s + im * c; } }
  }
  const dirs = [];
  for (let th = dTheta / 2; th < 180; th += dTheta) for (let ph = 0; ph < 360; ph += dPhi) {
    const t = th * Math.PI / 180, p = ph * Math.PI / 180;
    dirs.push({ th, ph, r: [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), Math.cos(t)], t: [Math.cos(t) * Math.cos(p), Math.cos(t) * Math.sin(p), -Math.sin(t)], f: [-Math.sin(p), Math.cos(p), 0], dOmega: Math.sin(t) * (dTheta * Math.PI / 180) * (dPhi * Math.PI / 180) });
  }
  // per point: J = n x H, M = -n x E as complex 3-vectors, times area. Tangential components sit
  // in the two in-plane axes; map (e1, e2, h1, h2) to xyz by the face normal.
  const Emax = new Float64Array(nf), dirMax = new Array(nf), Prad = new Float64Array(nf), Pflux = new Float64Array(nf), pattern = [];
  const J = new Float64Array(npts * 6), Mv = new Float64Array(npts * 6);   // re/im for x,y,z
  for (let q = 0; q < nf; q++) {
    const k = 2 * Math.PI * (f0 + q * df) / C0;
    let flux = 0;
    for (let p = 0; p < npts; p++) {
      const o = (p * nf + q) * 8, n = [nrm[3 * p], nrm[3 * p + 1], nrm[3 * p + 2]];
      // tangential E and H vectors (complex): axes in the face plane, in the order used by nf2ffSurface
      let E = [[0, 0], [0, 0], [0, 0]], H = [[0, 0], [0, 0], [0, 0]];
      const ax = n[0] !== 0 ? [1, 2] : (n[1] !== 0 ? [0, 2] : [0, 1]);
      E[ax[0]] = [acc[o], acc[o + 1]]; E[ax[1]] = [acc[o + 2], acc[o + 3]];
      H[ax[0]] = [acc[o + 4], acc[o + 5]]; H[ax[1]] = [acc[o + 6], acc[o + 7]];
      const cross = (a, B) => [[a[1] * B[2][0] - a[2] * B[1][0], a[1] * B[2][1] - a[2] * B[1][1]], [a[2] * B[0][0] - a[0] * B[2][0], a[2] * B[0][1] - a[0] * B[2][1]], [a[0] * B[1][0] - a[1] * B[0][0], a[0] * B[1][1] - a[1] * B[0][1]]];
      const Jp = cross(n, H), Mp = cross(n, E);
      const dA = area[p];
      // Poynting flux through the surface: 1/2 Re(E x H*) . n dA  (a check independent of the transform)
      const ExH = cross([0, 0, 0], H);   // placeholder to keep shape; compute directly below
      const re = (a, b) => a[0] * b[0] + a[1] * b[1];            // Re(a b*)
      const Sx = re(E[1], H[2]) - re(E[2], H[1]), Sy = re(E[2], H[0]) - re(E[0], H[2]), Sz = re(E[0], H[1]) - re(E[1], H[0]);
      flux += 0.5 * (Sx * n[0] + Sy * n[1] + Sz * n[2]) * dA;
      for (let c = 0; c < 3; c++) { J[p * 6 + 2 * c] = Jp[c][0] * dA; J[p * 6 + 2 * c + 1] = Jp[c][1] * dA; Mv[p * 6 + 2 * c] = -Mp[c][0] * dA; Mv[p * 6 + 2 * c + 1] = -Mp[c][1] * dA; }
    }
    let best = 0, bestDir = null, prad = 0;
    const pat = new Float64Array(dirs.length);
    dirs.forEach((d, di) => {
      const N = [0, 0, 0, 0, 0, 0], L = [0, 0, 0, 0, 0, 0];
      for (let p = 0; p < npts; p++) {
        const ph = k * (d.r[0] * pos[3 * p] + d.r[1] * pos[3 * p + 1] + d.r[2] * pos[3 * p + 2]);
        const c = Math.cos(ph), s = Math.sin(ph);          // e^{+j k r.r'}
        for (let cc = 0; cc < 3; cc++) {
          const jr = J[p * 6 + 2 * cc], ji = J[p * 6 + 2 * cc + 1], mr = Mv[p * 6 + 2 * cc], mi = Mv[p * 6 + 2 * cc + 1];
          N[2 * cc] += jr * c - ji * s; N[2 * cc + 1] += jr * s + ji * c;
          L[2 * cc] += mr * c - mi * s; L[2 * cc + 1] += mr * s + mi * c;
        }
      }
      const dot = (V, u) => [V[0] * u[0] + V[2] * u[1] + V[4] * u[2], V[1] * u[0] + V[3] * u[1] + V[5] * u[2]];
      const Nt = dot(N, d.t), Nf = dot(N, d.f), Lt = dot(L, d.t), Lf = dot(L, d.f);
      const g = k / (4 * Math.PI * r);
      const Eth = [g * (Lf[0] + ETA0 * Nt[0]), g * (Lf[1] + ETA0 * Nt[1])], Eph = [g * (Lt[0] - ETA0 * Nf[0]), g * (Lt[1] - ETA0 * Nf[1])];
      const E2 = Eth[0] ** 2 + Eth[1] ** 2 + Eph[0] ** 2 + Eph[1] ** 2;
      pat[di] = Math.sqrt(E2);
      if (pat[di] > best) { best = pat[di]; bestDir = d; }
      prad += E2 / (2 * ETA0) * r * r * d.dOmega;
    });
    Emax[q] = best; dirMax[q] = bestDir ? [bestDir.th, bestDir.ph] : null; Prad[q] = prad; Pflux[q] = flux; pattern.push(pat);
  }
  return { Emax, dirMax, Prad, Pflux, dirs: dirs.map(d => [d.th, d.ph]), pattern };
}

// trapezoid clock harmonics: amplitude A (V), fundamental f0, duty d, rise = fall = tr
export function trapezoidHarmonics(A, f0, tr, duty = 0.5, fmax = 3e9) {
  const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
  const out = [];
  for (let n = 1; n * f0 <= fmax; n++) out.push({ f: n * f0, V: 2 * A * duty * Math.abs(sinc(n * duty)) * Math.abs(sinc(n * f0 * tr)) });
  return out;
}

// CISPR 32 class B radiated limits at 3 m, dBuV/m: 40 (30-230 MHz), 47 (230 MHz-1 GHz) quasi-peak;
// 1-3 GHz: 70 peak / 50 average. We report against the peak-detector limit above 1 GHz.
export function cispr32B(f) { return f < 230e6 ? 40 : (f < 1e9 ? 47 : (f <= 3e9 ? 70 : 74)); }

// emissions of a trapezoid clock driving the source port: T(f) = Emax(f)/|Vinc(f)| interpolated
export function emissions(freqs, T, waveform, fmax) {
  const harm = trapezoidHarmonics(waveform.A, waveform.f0, waveform.tr, waveform.duty ?? 0.5, fmax);
  const interp = f => { if (f <= freqs[0]) return T[0]; if (f >= freqs[freqs.length - 1]) return T[T.length - 1]; let i = 1; while (freqs[i] < f) i++; const a = (f - freqs[i - 1]) / (freqs[i] - freqs[i - 1]); return T[i - 1] * (1 - a) + T[i] * a; };
  return harm.map(h => { const E = interp(h.f) * h.V; const dB = 20 * Math.log10(E * 1e6 + 1e-30); const lim = cispr32B(h.f); return { f: h.f, V: h.V, E, dBuV: dB, limit: lim, margin: lim - dB }; });
}
