"""CSV history of forces and diagnostics.

One header row of column names, then one row per output step. Numbers are written with 17
significant digits (``"%.17g"``), which reproduces every float64 exactly on reading, and the
file is flushed after every row so a run that is killed still leaves a readable history.

The driver writes ``step, t, t_force, fx_body, fy_body, max_target_error_over_dx, cfl,
mean_u, mean_v, kinetic_energy, max_speed`` every ``print_dump`` steps (``pyibm.driver`` and
docs/user-guide.md say what each column holds); nothing here depends on those names.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

_FMT = "%.17g"  # 17 significant digits round-trip float64 exactly


class HistoryWriter:
    """Append rows of numbers to a CSV file, one flush per row.

    Parameters
    ----------
    path : file to create or append to. The parent directory is created if needed. If the
        file already exists and is not empty, its header must equal ``columns`` exactly;
        otherwise we raise ``ValueError`` rather than silently mix two histories.
    columns : the column names, unique, without commas or newlines.

    Use ``write`` for each row (a sequence in column order, a mapping, or keyword arguments),
    then ``close``; the object also works as a context manager. Every value is converted with
    ``float``, so integer columns such as ``step`` come back as float64 from ``read_history``.
    """

    def __init__(self, path, columns) -> None:
        self._path = os.fspath(path)
        columns = tuple(str(c) for c in columns)
        if not columns:
            raise ValueError("HistoryWriter needs at least one column")
        for c in columns:
            if not c or "," in c or "\n" in c or "\r" in c or c != c.strip():
                raise ValueError(f"bad column name {c!r}: no commas, newlines or surrounding whitespace")
        if len(set(columns)) != len(columns):
            raise ValueError(f"column names must be unique, got {columns}")
        self._columns = columns

        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        appending = os.path.exists(self._path) and os.path.getsize(self._path) > 0
        if appending:
            with open(self._path, "r", encoding="utf-8") as fh:
                header = fh.readline().rstrip("\r\n")
            existing = tuple(header.split(","))
            if existing != columns:
                raise ValueError(
                    f"{self._path} already has columns {existing}, not {columns}; "
                    "delete it or use a new path")
        self._fh = open(self._path, "a", encoding="utf-8", newline="\n")
        if not appending:
            self._fh.write(",".join(columns) + "\n")
            self._fh.flush()

    # ------------------------------------------------------------------ properties
    @property
    def path(self) -> str:
        return self._path

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def closed(self) -> bool:
        return self._fh.closed

    # ------------------------------------------------------------------ writing
    def write(self, row=None, /, **values) -> None:
        """Append one row and flush.

        ``row`` is either a sequence of numbers in column order or a mapping from column name
        to number; alternatively pass the values as keyword arguments. Every column must be
        given (missing or extra names raise ``ValueError``).
        """
        if row is not None and values:
            raise ValueError("pass either a row or keyword values, not both")
        if row is None:
            row = values
        if isinstance(row, Mapping):
            missing = [c for c in self._columns if c not in row]
            extra = [k for k in row if k not in self._columns]
            if missing or extra:
                raise ValueError(f"row keys do not match columns: missing {missing}, extra {extra}")
            data = [row[c] for c in self._columns]
        else:
            data = list(row)
            if len(data) != len(self._columns):
                raise ValueError(f"row has {len(data)} values, expected {len(self._columns)}")
        self._fh.write(",".join(_FMT % float(v) for v in data) + "\n")
        self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "HistoryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def read_history(path) -> np.recarray:
    """Read a CSV written by ``HistoryWriter`` into a record array.

    Columns come back in file order as float64 fields, so ``h.t``, ``h["fx_body"]`` and
    ``h.step.astype(int)`` all work. A file with only a header gives a record array of length
    zero with the right fields. Blank lines are ignored; a row with the wrong number of values
    raises ``ValueError`` naming the line.
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f"{path}: empty history file (no header)")
    columns = [c.strip() for c in lines[0].split(",")]
    ncol = len(columns)
    rows = []
    for k, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != ncol:
            raise ValueError(f"{path} line {k}: {len(parts)} values, expected {ncol}")
        rows.append(parts)
    if rows:
        data = np.array(rows, dtype=float)
    else:
        data = np.zeros((0, ncol))
    return np.rec.fromarrays([data[:, j] for j in range(ncol)], names=columns)
