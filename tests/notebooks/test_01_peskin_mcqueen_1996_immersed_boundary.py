"""tutorials/01_peskin_mcqueen_1996_immersed_boundary.ipynb: the file loads and has enough code
cells (always on), and it executes headless without errors, prints no local path, and still
prints what its text says (``_claims.py``) (slow, run with -m slow)."""

from __future__ import annotations

import json
import os

import pytest

# nbformat and nbclient are not pyIBM dependencies: without them these tests are skipped
nbformat = pytest.importorskip("nbformat")

from ._claims import check_claims  # noqa: E402
from ._hygiene import local_path_hits  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_DIR = os.path.join(ROOT, "tutorials")
NB_PATH = os.path.join(NB_DIR, "01_peskin_mcqueen_1996_immersed_boundary.ipynb")


def test_notebook_loads_and_has_code_cells():
    nb = nbformat.read(NB_PATH, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10
    assert nb.metadata["kernelspec"]["name"] == "python3"
    # the first code cell puts the repository on sys.path so the notebook runs from tutorials/
    assert "sys.path" in code_cells[0].source


@pytest.mark.slow
def test_notebook_executes_without_errors():
    NotebookClient = pytest.importorskip("nbclient").NotebookClient

    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": NB_DIR}})
    client.execute()
    # what the cells print must not reveal the machine (paths, user name, temp folders)
    assert local_path_hits(nb) == [], local_path_hits(nb)
    # what the text says about the printed numbers still holds after a fresh run
    check_claims(os.path.basename(NB_PATH), json.loads(nbformat.writes(nb)))
    errors = [(i, out) for i, cell in enumerate(nb.cells) if cell.cell_type == "code"
              for out in cell.outputs if out.output_type == "error"]
    assert not errors, "\n".join(f"cell {i}: {out.ename}: {out.evalue}" for i, out in errors)
    # the last code cell prints the computation time; it must have run
    last = [c for c in nb.cells if c.cell_type == "code"][-1]
    text = "".join(out.get("text", "") for out in last.outputs if out.output_type == "stream")
    assert "notebook computations so far" in text
