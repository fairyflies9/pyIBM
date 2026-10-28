"""The committed tutorials are published with their outputs, so check what they carry.

These tests read the notebooks as plain JSON, so they run without nbformat or Jupyter:

- no source or output holds a local path, a user name or a temporary directory name (the
  first cell used to print the repository root, and notebook 01 its temporary output folder);
- no source or output points readers to the private working files that are not published
  (the design notes, the working notes, the local copies of the papers); papers are cited
  bibliographically and the public docs by file name (docs/user-guide.md, docs/numerics.md,
  docs/examples.md);
- no cell carries execution timestamps, so a re-run changes only what the cells print;
- and, separately, that the notebook tests are skipped rather than fail at collection when
  nbformat and nbclient are not installed (they are not among pyIBM's dependencies).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ._hygiene import PRIVATE, local_path_hits, local_path_pattern, private_hits

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TUTORIALS = os.path.join(ROOT, "tutorials")
NOTEBOOKS = sorted(f for f in os.listdir(TUTORIALS) if f.endswith(".ipynb"))
NOTEBOOK_TESTS = sorted(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
    for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
    if f.startswith("test_0") and f.endswith(".py"))


def _load(name):
    with open(os.path.join(TUTORIALS, name), encoding="utf-8") as fh:
        return json.load(fh)


def test_there_are_three_tutorials():
    assert len(NOTEBOOKS) == 3
    assert len(NOTEBOOK_TESTS) == 3


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_local_paths_or_user_names(name):
    hits = local_path_hits(_load(name))
    assert hits == [], hits


# Home folders other machines have: the check must not flag ordinary words that happen to be
# the home folder's name ("user" is in docs/user-guide.md, "data" in many sentences).
OTHER_MACHINES = [("/home/user", "/tmp"), ("/home/data", "/tmp"), ("/root", "/tmp"),
                  (r"C:\Users\user", r"C:\Users\user\AppData\Local\Temp"),
                  ("/Users/guide", "/var/folders/xy/T")]


@pytest.mark.parametrize("home, tmp", OTHER_MACHINES)
def test_path_check_has_no_false_hits_on_other_machines(home, tmp):
    pattern = local_path_pattern(home, tmp)
    hits = {name: local_path_hits(_load(name), pattern) for name in NOTEBOOKS}
    assert all(h == [] for h in hits.values()), hits
    assert not pattern.search("see docs/user-guide.md and the data/ folder")


@pytest.mark.parametrize("home, tmp", OTHER_MACHINES)
def test_path_check_finds_paths_on_other_machines(home, tmp):
    pattern = local_path_pattern(home, tmp)
    name = home.replace("\\", "/").rstrip("/").split("/")[-1]
    for leaked in (home + "/project/run.py", home.replace("\\", "/") + "/x", tmp + "/tmpab12cd/x",
                   "Users/" + name + "/x", "C:/data/" + name):
        assert pattern.search(leaked), leaked


@pytest.mark.parametrize("name", NOTEBOOKS + ["README.md"])
def test_no_pointers_to_private_files(name):
    if name.endswith(".md"):
        with open(os.path.join(TUTORIALS, name), encoding="utf-8") as fh:
            hits = [("README.md", m.group(0)) for m in PRIVATE.finditer(fh.read())]
    else:
        hits = private_hits(_load(name))
    assert hits == [], hits


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_saved_with_outputs_and_without_timestamps(name):
    nb = _load(name)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # every code cell was run (GitHub shows the outputs) and no cell keeps execution times
    assert all(c.get("execution_count") is not None for c in code)
    assert [i for i, c in enumerate(nb["cells"]) if "execution" in c.get("metadata", {})] == []
    assert not any(out.get("output_type") == "error" for c in code for out in c.get("outputs", []))


def test_notebook_tests_skip_without_nbformat():
    """A plain install (numpy, scipy, matplotlib, pytest) has no nbformat or nbclient: the
    notebook tests must be skipped, not stop the whole run at collection.

    The child pytest loads only its built-in plugins. A notebook plugin installed in the
    environment (nbmake, nbval, pytest-notebook) imports nbformat when it loads, which with
    nbformat blocked stops the child before collection; a plain install has no such plugin."""
    code = ("import sys\n"
            "sys.modules['nbformat'] = None\n"
            "sys.modules['nbclient'] = None\n"
            "import pytest\n"
            "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '-rs'] + sys.argv[1:]))\n")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    proc = subprocess.run([sys.executable, "-c", code] + NOTEBOOK_TESTS, cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    # exit code 5 means "no tests ran": all three modules skipped
    assert proc.returncode in (0, 5), out
    assert "error" not in out.lower(), out
    assert out.count("SKIPPED") >= 3 or "3 skipped" in out, out
