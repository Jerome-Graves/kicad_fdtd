"""kicad-fdtd command line.

    kicad-fdtd export  board.kicad_pcb [-o out.json]        exact copper geometry -> JSON (runs KiCad's Python)
    kicad-fdtd stats   board.kicad_pcb | geometry.json      board statistics (size, stackup, copper, nets, pairs)
    kicad-fdtd pair    geometry.json NET_P [NET_N] --ports REF.PAD ...   S-parameters / impedance on the GPU
    kicad-fdtd thermal geometry.json --power U1=1.2 ... [--net N --from R.P --to R.P --current A]   steady-state T, IR drop
    kicad-fdtd gui     [--port 8765]                         browser GUI
    kicad-fdtd test                                          microstrip milestone (needs a GPU for speed)
"""
import json
import os
import sys


def _geometry_of(path, out_dir="out"):
    """Accept a .kicad_pcb (exported on the fly) or a geometry JSON."""
    if path.lower().endswith(".kicad_pcb"):
        from .export import export
        os.makedirs(out_dir, exist_ok=True)
        return export(path, os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_geometry.json"))
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "export":
        from .export import export
        import argparse
        ap = argparse.ArgumentParser(prog="kicad-fdtd export")
        ap.add_argument("board")
        ap.add_argument("-o", "--out")
        a = ap.parse_args(rest)
        export(a.board, a.out)
        return 0
    if cmd == "stats":
        import argparse
        from .geometry import Board, format_stats
        ap = argparse.ArgumentParser(prog="kicad-fdtd stats")
        ap.add_argument("board", help=".kicad_pcb or geometry.json")
        ap.add_argument("--ref", help="reference net")
        ap.add_argument("--json", action="store_true", help="print the full JSON (with per-net detail)")
        a = ap.parse_args(rest)
        s = Board(_geometry_of(a.board), ref_net=a.ref).stats()
        print(json.dumps(s, indent=1) if a.json else format_stats(s))
        return 0
    if cmd == "pair":
        from .pair import main as pair_main
        if rest and rest[0].lower().endswith(".kicad_pcb"):
            rest[0] = _geometry_of(rest[0])
        return pair_main(rest)
    if cmd == "thermal":
        from .thermal import main as thermal_main
        if rest and rest[0].lower().endswith(".kicad_pcb"):
            rest[0] = _geometry_of(rest[0])
        return thermal_main(rest)
    if cmd == "gui":
        from .gui.server import main as gui_main
        gui_main(rest)
        return 0
    if cmd == "test":
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import runpy
        sys.argv = ["test_msl.py"] + rest
        runpy.run_path(os.path.join(here, "tests", "test_msl.py"), run_name="__main__")
        return 0
    print("unknown command %r\n%s" % (cmd, __doc__))
    return 2


if __name__ == "__main__":
    sys.exit(main())
