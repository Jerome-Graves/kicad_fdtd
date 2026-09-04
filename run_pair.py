"""Thin wrapper: same as `kicad-fdtd pair ...` (see kicad_fdtd/pair.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kicad_fdtd.pair import main

if __name__ == "__main__":
    sys.exit(main())
