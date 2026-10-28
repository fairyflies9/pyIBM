"""tutorials/03_blob_projection_method.ipynb: the file loads and has enough code cells (always),
and it executes from start to end without an error, prints no local path, and still prints what
its text says (``_claims.py``) (slow, run with -m slow)."""

from __future__ import annotations

import json
import os

import pytest

# nbformat and nbclient are not pyIBM dependencies: without them these tests are skipped
nbformat = pytest.importorskip("nbformat")

from ._claims import check_claims  # noqa: E402
from ._hygiene import local_path_hits  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOK = os.path.join(ROOT, "tutorials", "03_blob_projection_method.ipynb")


def test_notebook_loads_and_has_code_cells():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 10
    assert nb.metadata["kernelspec"]["name"] == "python3"


@pytest.mark.slow
def test_notebook_executes_without_errors():
    NotebookClient = pytest.importorskip("nbclient").NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(NOTEBOOK)}})
    client.execute()          # raises CellExecutionError on the first failing cell
    # what the cells print must not reveal the machine (paths, user name, temp folders)
    assert local_path_hits(nb) == [], local_path_hits(nb)
    # what the text says about the printed numbers still holds after a fresh run
    check_claims(os.path.basename(NOTEBOOK), json.loads(nbformat.writes(nb)))
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            assert out.output_type != "error", f"{out.ename}: {out.evalue}"
    assert any(out.output_type == "display_data" and "image/png" in out.get("data", {})
               for cell in nb.cells if cell.cell_type == "code" for out in cell.get("outputs", []))
