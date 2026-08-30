# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Binary bridge to the compiled Fortran Darcy solver.

Kept separate from ``tesseract_api.py`` so it can be unit-tested without the
Tesseract runtime installed.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SOLVER = os.environ.get("DARCY_SOLVER_BIN", "/tesseract/fortran/darcy")

F8 = np.dtype("<f8")


def _w_f8(fh, arr) -> None:
    fh.write(np.ascontiguousarray(arr, dtype=F8).tobytes(order="F"))


def run_darcy(
    gamma: np.ndarray,
    mu: np.ndarray,
    *,
    dx: float,
    dy: float,
    pin: float,
    kmin: float,
    kmax: float,
    qramp: float,
    tol: float,
    maxit: int,
    cotangents: dict | None = None,
    solver: str | None = None,
) -> dict:
    """Run the Fortran solver.

    ``cotangents`` is either None (forward only) or a dict with keys
    ``p``, ``ux``, ``uy``, ``flux`` -- in which case the hand-derived adjoint
    also runs and ``gamma_bar`` / ``mu_bar`` come back.
    """
    gamma = np.asarray(gamma, dtype=F8)
    mu = np.asarray(mu, dtype=F8)
    if gamma.shape != mu.shape:
        raise ValueError(f"gamma {gamma.shape} and mu {mu.shape} must match")
    nx, ny = gamma.shape
    mode = 1 if cotangents is not None else 0

    tmp = Path(tempfile.mkdtemp(prefix="darcy_"))
    fin, fout = tmp / "in.bin", tmp / "out.bin"
    try:
        with open(fin, "wb") as fh:
            fh.write(struct.pack("<4i", nx, ny, mode, int(maxit)))
            fh.write(struct.pack("<7d", dx, dy, pin, kmin, kmax, qramp, tol))
            _w_f8(fh, gamma)
            _w_f8(fh, mu)
            if mode == 1:
                for key in ("p", "ux", "uy"):
                    _w_f8(fh, np.asarray(cotangents.get(key, np.zeros((nx, ny))), dtype=F8))
                fh.write(struct.pack("<d", float(cotangents.get("flux", 0.0))))

        proc = subprocess.run(
            [solver or SOLVER, str(fin), str(fout)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Darcy solver failed (rc={proc.returncode})\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        raw = fout.read_bytes()
    finally:
        for f in (fin, fout):
            f.unlink(missing_ok=True)
        tmp.rmdir()

    off = 0
    (iters,) = struct.unpack_from("<i", raw, off)
    off += 4
    (resid,) = struct.unpack_from("<d", raw, off)
    off += 8

    n = nx * ny

    def take_field():
        nonlocal off
        a = np.frombuffer(raw, dtype=F8, count=n, offset=off).reshape((nx, ny), order="F")
        off += n * 8
        return np.array(a)

    out = {
        "p": take_field(),
        "ux": take_field(),
        "uy": take_field(),
    }
    (out["flux"],) = struct.unpack_from("<d", raw, off)
    off += 8
    out["iters"] = int(iters)
    out["resid"] = float(resid)
    if mode == 1:
        (adj_iters,) = struct.unpack_from("<i", raw, off)
        off += 4
        (adj_resid,) = struct.unpack_from("<d", raw, off)
        off += 8
        out["adj_iters"] = int(adj_iters)
        out["adj_resid"] = float(adj_resid)
        out["gamma_bar"] = take_field()
        out["mu_bar"] = take_field()
    return out
