# kicad_fdtd

A GPU field solver for KiCad boards. Point it at any `.kicad_pcb`: it pulls the exact copper
(pads, tracks, zone fills, vias, outline, stackup) through KiCad's own geometry engine, meshes
the nets you care about, checks the model before it spends a second of GPU time, and gives you
S-parameters, impedance and a field view in a browser. Written in Python with NumPy and CuPy;
no openEMS, no black box.

```
kicad-fdtd stats  my_board.kicad_pcb                      # what is on the board
kicad-fdtd pair   my_board.kicad_pcb /USB_DP /USB_DM --ports J1.A6 U5.1 J1.A7 U5.3
kicad-fdtd gui                                            # everything above in a browser
```

## Why

Every field solver hides the same three failure modes: copper assigned to the wrong
conductor on the mesh, absorbing boundaries touching the board, ports sitting on the wrong
copper. In a black box each one costs an hour of simulation before you find out. Here the
physics is a few hundred lines you can read, every mesh edge's owner is visible in the GUI,
and a merge check refuses to solve a model where two conductors touch. On an RTX 4070 a
5 Mcell board mesh runs at 3.2 Gcell-updates per second, so a USB pair at 0.1 mm resolution
takes about a minute.

## Install

```
git clone https://github.com/Jerome-Graves/kicad_fdtd
cd kicad_fdtd
python -m venv venv && venv\Scripts\activate          # or source venv/bin/activate
pip install -e .[gpu]                                 # NumPy, shapely, CuPy for CUDA 12
```

Without a CUDA GPU use `pip install -e .` and add `--cpu` to `pair`; the solver falls back to
NumPy (correct, roughly 100 times slower). KiCad 7 or later must be installed: the exporter
runs inside KiCad's bundled Python (found automatically on Windows and macOS; on Linux the
distro package puts `pcbnew` in the system Python). Set `KICAD_PYTHON` if it is somewhere else.

## Use

**Stats.** `kicad-fdtd stats board.kicad_pcb` prints size, stackup (from the file, or a
1.6 mm FR4 default with a warning), copper area and fill per layer, footprint/pad/via/net
counts, the reference net (the net with the most zone copper, usually GND) and the
differential pairs it can recognise by name. `--json` adds per-net detail.

**Pair or single net.** `kicad-fdtd pair board.kicad_pcb NET_P [NET_N] --ports REF.PAD ...`
Ports are pad names: P start, P stop, then N start, N stop. Each port is a lumped Z0
resistor with a source between the signal copper and the nearest reference copper under
it; the whole port face has to sit on the right copper on both layers or the tool says so.
Two nets are solved in odd mode (Zdiff = 2 Zodd, S11dd, S21dd), one net single-ended.
Options: `--res` fine cell (mm, default 0.1), `--base` coarse cell, `--fmax`, `--z0`,
`--ref NET` to override the reference net, `--tie NET ...` to treat other nets as reference
copper (a decoupled rail), `--setup-only` to stop after the checks, `--bench N` for speed.

**GUI.** `kicad-fdtd gui` then open http://127.0.0.1:8765. Paste a `.kicad_pcb` path and
press Import (or pick an already exported board), read the stats, pick nets (the detected
pairs are one click), click pads for ports, **Set up and check**. The view shows the mesh
window, the mesh, the PEC edge assignment per conductor on any copper layer, the port sites
and any shorts (red rings; Run stays disabled). **Run on GPU** streams progress, keeps Ez
snapshots on the plane under the first port (slider, play) and plots S11/S21 and Zin/Zdiff.

## How it works

```
kicad_fdtd/
  export.py     KiCad's Python (pcbnew) -> geometry.json: exact polygons, vias, pads, stackup
  geometry.py   model frame (y up, z from top copper), stackup defaults, reference net,
                port sites, board stats, differential pair detection
  mesh.py       rectilinear mesh: lines on the copper edges of the nets of interest,
                neighbours' edges within 0.6 mm, base grid, graded air cells; CFL step
  voxel.py      every Yee edge assigned to a conductor by an exact polygon test at the
                edge midpoint (sheets) or cylinder (vias); merge_check finds shared nodes
  solver.py     Yee FDTD on a non-uniform grid, CPML on six faces, PEC edges, lumped ports,
                energy-based end criterion; NumPy fallback
  kernels.py    six fused CuPy kernels (curl, CPML and ports in one launch each),
                port V/I read by one kernel; no host sync inside the step loop
  pair.py       setup (mesh + assignment + checks) and solve (S-parameters, Z)
  gui/          standard-library server + one page
tests/          test_msl.py (microstrip milestone, GPU), test_geometry.py (no GPU)
```

Field quantities are float32; V and I are recorded on the device each step and transformed
at the end. `FDTD_PROFILE=1` prints a per-section time breakdown, `FDTD_UNFUSED=1` runs the
slice-based reference path.

## Validation

| case | this solver | reference |
|---|---|---|
| 0.2 mm microstrip over 0.1 mm FR4, 50 Ω ports (`tests/test_msl.py`) | S21 −0.06 dB, S11 −18.6 dB, Zin 42 Ω | openEMS 43 Ω, 2D FD cross-section 45 Ω |
| USB pair on a 4-layer board, 0.15 mm vs 0.1 mm mesh | Zdiff 58 Ω vs 60 Ω | 2D FD cross-section 66 Ω |
| fused kernels vs slice reference vs pre-fusion code, same mesh | identical step count, S-parameters equal to 1e-6 dB | |

The milestone runs in about a second on the GPU and is the regression gate for every
change to the solver.

## Limits

- Copper is perfect (PEC) and infinitely thin. Conductor and dielectric loss are on the list.
- Only the copper of the selected nets and their neighbours within 0.6 mm is meshed finely;
  the rest of the board is on the base grid. Parts (bodies, connectors) are not modelled.
- Ports need reference copper under the pad; a trace with no plane below gets no port.
- Stackups without thicknesses fall back to 1.6 mm FR4 split evenly.

## Roadmap

- Thermal: steady-state heat conduction on the same copper voxels (component power as
  sources, convection on the faces) and DC current density / IR drop on the power nets.
- Far field on a Huygens box for emissions against CISPR 32.
- Conducting-sheet loss, dielectric loss.
- Whole-board mesh presets and a results archive in the GUI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
