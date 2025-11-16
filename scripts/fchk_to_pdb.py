#!/usr/bin/env python3
"""
Convert a Gaussian .fchk file to a .pdb file using Open Babel (obabel).

Usage
-----
Basic:
    python fchk_to_pdb.py input.fchk

Specify output:
    python fchk_to_pdb.py input.fchk -o output.pdb

Specify custom obabel path:
    python fchk_to_pdb.py input.fchk --obabel /path/to/obabel

Notes
-----
- This script only calls the Open Babel CLI, it does not import any
  Open Babel Python bindings.
- Make sure the environment you run this in has 'obabel' installed
  and on PATH (e.g., via conda-forge or system package manager).
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_obabel_fchk_to_pdb(
    fchk_path: str,
    pdb_path: str | None = None,
    obabel_exec: str = "obabel",
) -> None:
    """Convert a .fchk file to .pdb using Open Babel.

    Parameters
    ----------
    fchk_path : str
        Path to the input .fchk file.
    pdb_path : str or None
        Path to the output .pdb file. If None, uses the same basename
        as the input and changes the extension to .pdb.
    obabel_exec : str
        Name or full path of the obabel executable.
    """
    fchk_path = str(Path(fchk_path).resolve())
    if pdb_path is None:
        pdb_path = str(Path(fchk_path).with_suffix(".pdb"))
    else:
        pdb_path = str(Path(pdb_path).resolve())

    # Check that obabel is available
    if shutil.which(obabel_exec) is None:
        raise RuntimeError(
            f"Could not find '{obabel_exec}' on PATH.\n"
            "Make sure Open Babel is installed and the 'obabel' binary is available."
        )

    # Construct the Open Babel command
    cmd = [
        obabel_exec,
        "-ifchk",
        fchk_path,  # input format and file
        "-opdb",  # output PDB
        "-O",
        pdb_path,  # output file
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("Open Babel failed.\n", file=sys.stderr)
        print("STDOUT:\n", result.stdout, file=sys.stderr)
        print("STDERR:\n", result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    print(f"Conversion successful. Wrote: {pdb_path}")
    # If you want to see obabel's stdout:
    if result.stdout.strip():
        print("Open Babel output:")
        print(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gaussian .fchk to .pdb via Open Babel (obabel)."
    )
    parser.add_argument("fchk", help="Input .fchk file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .pdb file (default: same basename as input with .pdb extension)",
        default=None,
    )
    parser.add_argument(
        "--obabel",
        help="Path to obabel executable (default: 'obabel' on PATH)",
        default="obabel",
    )

    args = parser.parse_args()

    try:
        run_obabel_fchk_to_pdb(
            fchk_path=args.fchk,
            pdb_path=args.output,
            obabel_exec=args.obabel,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
