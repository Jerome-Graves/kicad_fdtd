"""Differential pair (or single net) S-parameters from a board on the GPU.

    kicad-fdtd pair <geometry.json> NET_P [NET_N] --ports REF.PAD ... [--res 0.1] [--fmax 3e9]

Ports: P start, P stop [, N start, N stop]; lumped Z0 boxes sited by
geometry.Board.port_site (whole face on the signal, whole face on the reference copper).
Odd mode when a pair is given (P +1, N -1): Zdiff = 2 Zodd, S21dd, S11dd.
Before solving: exact edge assignment + merge check (refuses on any shared
node between the pair and another conductor) and a time-step/steps print.
"""
import json
import os
import sys
import time

import numpy as np

from . import geometry, mesh, voxel, solver


def setup(bd, nets, ports, res=0.1, base=0.2, margin=2.5, tie=(), z_cell=0.1, log=print, use_gpu=True):
    """Mesh + exact edge assignment + merge check. Returns a dict with everything the solve needs. The assignment
    and the check run on the GPU where there is one (voxel.backend)."""
    TIE = {n: bd.ref_net for n in tie}
    window = bd.net_bbox(nets, margin)
    sites, boxes, w = [], [], geometry.PORT_W
    vx = voxel.backend(use_gpu)
    for rp in ports:
        ref, num = rp.split(".")
        s = vx.port_site(bd, ref, num, tie_nets=TIE)
        if not s:
            raise ValueError("no clean port site for %s (pad on %s, reference %s)" % (rp, bd.pad(ref, num)[0]["net"], bd.ref_net))
        x, y, sig, plane, pnet, shift = s
        z1, z2 = sorted((bd.z[plane], bd.z[sig]))
        sites.append({"name": rp, "x": x, "y": y, "sig": sig, "plane": plane, "ref": pnet, "shift": shift, "z0": z1, "z1": z2})
        boxes.append(([x - w / 2, y - w / 2, z1], [x + w / 2, y + w / 2, z2]))
        if log:
            log("   port %-8s at (%.2f, %.2f) %s over %s (%s)%s" % (rp, x, y, sig, plane, pnet, "" if not shift else "  (shifted %.1f mm)" % shift))
    t = time.time()
    lines = mesh.build_mesh(bd, nets, window, res=res, base=base, tie_nets=TIE, port_boxes=boxes, z_cell=z_cell)
    vox = vx.assign(bd, lines, tie_nets=TIE, skip_via_near=[(s["x"], s["y"]) for s in sites], window=window)
    shorts = vx.merge_check(vox, lines, nets, TIE)
    cells = int(np.prod([len(lines[k]) for k in "xyz"]))
    dt, dmin = mesh.cfl_dt(lines)
    if log:
        log("mesh %d x %d x %d = %.2f Mcells (%.0f s)  smallest cells %.3f/%.3f/%.3f mm  dt %.0f fs  vias %d"
            % (len(lines["x"]), len(lines["y"]), len(lines["z"]), cells / 1e6, time.time() - t, dmin[0] * 1e3, dmin[1] * 1e3, dmin[2] * 1e3, dt * 1e15, vox["n_via"]))
        if shorts:
            log("MERGE CHECK: %d node(s) join the pair to another conductor; first few:" % len(shorts))
            for s in shorts[:8]:
                log("   %s ~ %s at (%.2f, %.2f, %.3f)" % s)
        else:
            log("MERGE CHECK: clean")
    return {"nets": nets, "ports": ports, "tie": TIE, "sites": sites, "boxes": boxes, "window": window, "lines": lines,
            "vox": vox, "shorts": shorts, "cells": cells, "dt": dt, "dmin": dmin}


def solve(bd, st, fmax=3e9, z0=50.0, tmax=12e-9, end=1e-3, use_gpu=True, log=print, hook=None, hook_every=200):
    """Odd-mode (pair) or single-ended solve. Returns the result dict (f, S11_dB, S21_dB, Z_re[, Zdiff_re] ...)."""
    def er_of_z(z):
        for zt, zb, er, tand in bd.diel:
            if zb <= z <= zt:
                return er
        return 1.0
    exc, t0, tau = solver.gauss_pulse(fmax / 2, fmax / 2)
    pair = len(st["nets"]) == 2
    sim = solver.FDTD(st["lines"], st["vox"], er_of_z, npml=8, use_gpu=use_gpu)
    exc_of = [1, 0, -1, 0] if pair else [1, 0, 0, 0]
    for i, (s, e) in enumerate(st["boxes"]):
        sim.add_port(s[0], e[0], s[1], e[1], s[2], e[2], z0, excite=exc_of[i] if i < len(exc_of) else 0)
    steps = int(tmax / sim.dt)
    if log:
        log("[%s] %s engine, %d steps for %.0f ns, solving ..." % ("odd" if pair else "single", "GPU" if sim.xp is not np else "CPU", steps, tmax * 1e9))
    sim.run(exc, steps, end_criteria=end, log=log, log_every=5000, hook=hook, hook_every=hook_every)
    if log:
        log("%d steps in %.0f s (%.0f Mcell-updates/s)" % (sim.nsteps_done, sim.wall, sim.nsteps_done * st["cells"] / sim.wall / 1e6))
    f = np.linspace(100e6, fmax, 300)
    S = sim.port_spectra(f)
    s11 = S[0]["ref"] / S[0]["inc"]
    s21 = (S[1]["ref"] / S[0]["inc"]) if len(sim.ports) > 1 else np.zeros_like(s11)
    Z = S[0]["Z"]
    r = {"f": f.tolist(), "S11_dB": (20 * np.log10(abs(s11) + 1e-12)).tolist(), "S21_dB": (20 * np.log10(abs(s21) + 1e-12)).tolist(),
         "Z_re": Z.real.tolist(), "pair": pair, "steps": sim.nsteps_done, "wall": sim.wall, "dt": sim.dt,
         "speed": sim.nsteps_done * st["cells"] / sim.wall / 1e6, "nets": st["nets"], "ports": st["ports"], "cells": st["cells"],
         "port_VI": [{"V": p.V.tolist(), "I": p.I.tolist()} for p in sim.ports],
         "S11": [s11.real.tolist(), s11.imag.tolist()], "S21": [s21.real.tolist(), s21.imag.tolist()], "z0": z0}
    if pair:
        r["Zdiff_re"] = (2 * Z.real).tolist()
    r["summary"] = summary(r)
    return r, sim


def end_ports(bd, p, n):
    """A pair's four ports, P start, P stop, N start, N stop ("REF.PAD"): P's two pads furthest apart, and for each
    the N pad nearest it (the two nets need not end on the same parts: series resistors, one a net). None when
    either net has fewer than two pads, or both ends fall on one N pad."""
    pp = [q for q in bd.pads if q["net"] == p]
    nn = [q for q in bd.pads if q["net"] == n]
    if len(pp) < 2 or len(nn) < 2:
        return None
    d2 = lambda a, b: (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2  # noqa: E731
    a, b = max(((x, y) for i, x in enumerate(pp) for y in pp[i + 1:]), key=lambda xy: d2(*xy))
    na, nb = min(nn, key=lambda q: d2(q, a)), min(nn, key=lambda q: d2(q, b))
    if na is nb:
        return None
    return ["%s.%s" % (q["ref"], q["num"]) for q in (a, b, na, nb)]


def line_impedance(r):
    """The line's own impedance at each frequency, from the two-port S-parameters of the solve (the usual
    extraction for a line between matched ports):

        Zc = Zref sqrt(((1 + S11)^2 - S21^2) / ((1 - S11)^2 - S21^2))

    Zref the ports' (2 z0 for a pair in odd mode: differential). Unlike the input impedance, which swings between
    Zref and Zc^2 / Zref along the band, this is the line's impedance itself; where the line is a whole number of
    half waves long the two brackets both go to zero and the value there is noise, so take a median over a band.
    Returns the real part, an array over r["f"]."""
    s11 = np.array(r["S11"][0]) + 1j * np.array(r["S11"][1])
    s21 = np.array(r["S21"][0]) + 1j * np.array(r["S21"][1])
    zref = r["z0"] * (2.0 if r["pair"] else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = zref * np.sqrt(((1 + s11) ** 2 - s21 ** 2) / ((1 - s11) ** 2 - s21 ** 2))
    return np.abs(zc.real)


def summary(r):
    f = np.array(r["f"]); Z = np.array(r["Zdiff_re"] if r["pair"] else r["Z_re"])
    sel = (f > 200e6) & (f < 2e9)
    lines = ["%s (0.2-2 GHz): %.0f .. %.0f ohm, median %.0f" % ("Zdiff" if r["pair"] else "Zin", Z[sel].min(), Z[sel].max(), np.median(Z[sel]))]
    for fq in (480e6, 1e9, 2e9):
        if fq <= f[-1]:
            i = int(np.argmin(abs(f - fq)))
            lines.append("%4.0f MHz: S21 %.2f dB  S11 %.1f dB  Z %.0f ohm" % (fq / 1e6, r["S21_dB"][i], r["S11_dB"][i], Z[i]))
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kicad-fdtd pair", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geometry")
    ap.add_argument("net_p")
    ap.add_argument("net_n", nargs="?")
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--ref", help="reference net (default: the net with the most zone copper)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--base", type=float, default=0.2)
    ap.add_argument("--fmax", type=float, default=3e9)
    ap.add_argument("--margin", type=float, default=2.5)
    ap.add_argument("--z0", type=float, default=50.0)
    ap.add_argument("--end", type=float, default=1e-3)
    ap.add_argument("--tmax", type=float, default=12e-9)
    ap.add_argument("--tie", nargs="*", default=[], help="nets to treat as the reference (e.g. a decoupled rail)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="run N steps, print the speed, exit")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    bd = geometry.Board(a.geometry, ref_net=a.ref)
    nets = [a.net_p] + ([a.net_n] if a.net_n else [])
    print("reference net:", bd.ref_net, " stackup:", bd.stackup_source)
    st = setup(bd, nets, a.ports, res=a.res, base=a.base, margin=a.margin, tie=a.tie, use_gpu=not a.cpu)
    if st["shorts"]:
        sys.exit("refusing to solve a shorted model")
    if a.setup_only:
        return 0
    if a.bench:
        def er(z):
            for zt, zb, e, td in bd.diel:
                if zb <= z <= zt:
                    return e
            return 1.0
        sim = solver.FDTD(st["lines"], st["vox"], er, npml=8, use_gpu=not a.cpu)
        exc, _, _ = solver.gauss_pulse(a.fmax / 2, a.fmax / 2)
        for i, (s, e) in enumerate(st["boxes"]):
            sim.add_port(s[0], e[0], s[1], e[1], s[2], e[2], a.z0, excite=1.0 if i == 0 else 0.0)
        sim.run(exc, a.bench, end_criteria=0.0, log=None, min_steps=a.bench)
        print("BENCH: %d steps in %.1f s = %.0f Mcell-updates/s (%.1f ms/step)" % (sim.nsteps_done, sim.wall, sim.nsteps_done * st["cells"] / sim.wall / 1e6, 1e3 * sim.wall / sim.nsteps_done))
        return 0
    r, sim = solve(bd, st, fmax=a.fmax, z0=a.z0, tmax=a.tmax, end=a.end, use_gpu=not a.cpu)
    print(r["summary"])
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.geometry).split("_")[0] + "_" + a.net_p.rsplit("/", 1)[-1].strip("/")
    path = os.path.join(a.out, "pair_%s.json" % tag)
    json.dump({k: v for k, v in r.items() if k != "port_VI"}, open(path, "w"))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
