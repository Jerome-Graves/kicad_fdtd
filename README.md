# kicad_fdtd

A GPU field solver for KiCad boards. Exact copper from KiCad (pads, tracks,
zone fills, vias, outline, stackup) → FDTD on the GPU (CuPy) → S-parameters,
impedance, near/far fields → a browser GUI that shows the board, the mesh,
the fields and the results.

Why this exists (2026-09-04): openEMS gave good answers once its rectangular
grid was fed the right geometry, but every failure mode was hidden inside a
black box (edges assigned to the wrong conductor, PML sitting on the board,
ports on the wrong copper) and each run cost 20–60 minutes on the CPU. The
physics is a hundred lines; owning it means every assignment is visible and
checkable, and an RTX 4070 runs 0.05 mm cells at whole-board scale in
minutes. Jerome: "if the solver is this primitive then why not just remake it
using cupy with our own gui, then allow files exported via kicad."

## Design

```
kicad_export.py    KiCad's Python → geometry.json  (exact polygons, mm, y down)
kicad_fdtd/
  geometry.py      load json, model frame (y up, z from top copper), stackup, nets
  mesh.py          rectilinear mesh: lines on conductor edges of the nets of
                   interest, base grid, graded air cells; CFL time step
  voxel.py         assign every Yee E-edge to a conductor by exact polygon
                   test at the edge midpoint (sheets) / cylinder (vias);
                   report merged conductors BEFORE solving
  solver.py        Yee update in CuPy (NumPy fallback), CPML on 6 faces,
                   conducting-sheet loss (surface impedance) or PEC,
                   lumped ports (voltage source + R), Gaussian excitation,
                   energy-based end criterion
  ports.py         port voltages/currents → S-parameters, Zin, mixed-mode
  nf2ff.py         near-to-far-field on a Huygens box
  checks.py        pre-flight: conductor merges, port faces, dt/steps, air cells
gui/               static web page (canvas): board + mesh view, run control,
                   S-param / impedance plots, field slices (served locally)
tests/             microstrip (analytic Z), coupled pair (2D FD reference),
                   PML reflection, port calibration
```

Units: mm in geometry, SI inside the solver. Frame: x right, y up, z = 0 at
the top copper, origin at the board's bounding-box corner (same as
bugbot_sim/tools/openems, whose exporter this reuses).

## Status

Skeleton. First milestone: the microstrip unit test (S21 ≈ 0 dB, Z ≈ 52 Ω
for 0.2 mm over 0.1 mm FR4) on the GPU, faster than openEMS's 5 minutes.
