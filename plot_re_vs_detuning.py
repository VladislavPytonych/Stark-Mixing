#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_rho21_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a 3-column whitespace-delimited file with a 1-line header:
      detuning_MHz  Re_rho21  Im_rho21
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns after the header, got shape {data.shape}."
        )

    detuning_MHz = data[:, 0]
    re_rho21 = data[:, 1]
    im_rho21 = data[:, 2]
    return detuning_MHz, re_rho21, im_rho21


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Re(rho_21) vs detuning (MHz).")
    p.add_argument(
        "datafile",
        nargs="?",
        default="rho21_vs_detuning.txt",
        help="Input data file (default: rho21_vs_detuning.txt)",
    )
    p.add_argument(
        "--save",
        default="re_vs_detuning.png",
        help="Output image filename (default: re_vs_detuning.png)",
    )
    p.add_argument(
        "--noshow",
        action="store_true",
        help="Do not open a GUI window (still saves the figure).",
    )
    args = p.parse_args()

    path = Path(args.datafile)
    detuning_MHz, re_rho21, _ = load_rho21_file(path)

    plt.figure()
    plt.plot(detuning_MHz, re_rho21)
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Re(rho_21)")
    plt.title("Re(rho_21) vs detuning")
    plt.grid(True)

    # Save before show (show can close the figure in some backends)
    plt.savefig(args.save, dpi=200, bbox_inches="tight")

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
