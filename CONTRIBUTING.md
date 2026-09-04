# Contributing

Issues and pull requests are welcome. A few ground rules that keep the solver trustworthy:

- **Every physics change ships with a check.** `python tests/test_msl.py` (the microstrip milestone)
  must still report S21 within 1 dB of 0, S11 below -12 dB and Zin between 35 and 70 ohm, and
  `python tests/test_geometry.py` must pass. If you touch the kernels, also run a board with
  `FDTD_UNFUSED=1` and compare the S-parameters (they agree to 1e-6 dB today).
- **No silent fallbacks.** If a port cannot be placed, a stackup is missing, or two conductors
  merge on the mesh, the code says so and refuses to give a number. Keep it that way.
- **Speed changes come with a number.** `kicad-fdtd pair ... --bench 300` prints Mcell-updates/s;
  `FDTD_PROFILE=1` prints the per-section breakdown.
- Plain Python, NumPy and shapely; CuPy only behind `solver.HAVE_GPU`. No new dependencies
  without a reason in the pull request.
- Keep KiCad access inside `kicad_fdtd/export.py`; everything else reads the JSON.

Good first contributions: a board that breaks the exporter or the port search (attach the
`.kicad_pcb` or the geometry JSON), a stackup the parser misreads, or a measured impedance
to compare against.
