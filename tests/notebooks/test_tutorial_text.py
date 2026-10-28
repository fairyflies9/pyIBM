"""What the tutorials say against what their saved outputs print.

The notebooks are published with their outputs, and a re-execution can change a printed number
while the text that quotes it stays the same. These tests read the saved notebooks as plain JSON
(no nbformat or Jupyter needed) and run the checks of ``_claims.py`` on them; the slow notebook
tests run the same checks on a fresh execution. The last test brings back, in copies of the
saved notebooks, the wrong claims the checks were written for, and samples of the wording the
tone check looks for, and makes sure they are caught.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from ._claims import (CHECKS, NB01, NB02, NB03, area_ratio_credited_to_the_interpolation,
                      bicubic_described_neutrally, cylinder_marker_pattern_backed,
                      eq_28_and_rotational_update_stated_right, gaussian4_cut_keeps_too_much_mass,
                      header_names_where_numbers_come_from, no_criticism_of_sources,
                      pointers_to_tests_exist, ring_runs_show_default_against_delta,
                      section_11_says_what_pyibm_changes)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TUTORIALS = os.path.join(ROOT, "tutorials")


def _load(name):
    with open(os.path.join(TUTORIALS, name), encoding="utf-8") as fh:
        return json.load(fh)


CASES = [(name, check) for name, checks in CHECKS.items() for check in checks]


@pytest.mark.parametrize("name, check", CASES,
                         ids=[f"{name[:2]}-{check.__name__}" for name, check in CASES])
def test_saved_outputs_back_the_text(name, check):
    check(_load(name))


def test_every_tutorial_is_checked():
    assert sorted(CHECKS) == sorted(f for f in os.listdir(TUTORIALS) if f.endswith(".ipynb"))


# ---------------------------------------------------------------------------------------------
# The checks must catch the errors they were written for: each mutation below brings back a
# wrong claim a review found (or a rewording of it) in a copy of the saved notebook. The tone
# checks get a made-up sample of the wording they look for, put at a neutral heading.
# ---------------------------------------------------------------------------------------------
def _mutated(nb, old, new, markdown=True, outputs=True):
    """A copy of ``nb`` with ``old`` replaced by ``new`` in the markdown sources and/or the
    printed (stream) outputs; at least one replacement must happen."""
    nb = copy.deepcopy(nb)
    count = 0
    for cell in nb["cells"]:
        if markdown and cell["cell_type"] == "markdown":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            count += src.count(old)
            cell["source"] = src.replace(old, new)
        for out in cell.get("outputs", []):
            if outputs and out.get("output_type") == "stream":
                text = "".join(out["text"]) if isinstance(out["text"], list) else out["text"]
                count += text.count(old)
                out["text"] = text.replace(old, new)
    assert count, f"{old!r} is not in the notebook any more"
    return nb


MUTATIONS = [
    # the gaussian4 cut said to lose mass, in the text, in the printed line, or both
    (NB03, gaussian4_cut_keeps_too_much_mass, "5.5 % too much", "5.5 % too little", True, True),
    (NB03, gaussian4_cut_keeps_too_much_mass, "keeps 5.5 % too much mass", "drops 5.5 % of the mass",
     False, True),
    (NB03, gaussian4_cut_keeps_too_much_mass, "5.5 % too much inside", "5.5 % too much, and it loses 3.2 % inside",
     True, False),
    # the delta / blob ratio presented as the difference between the couplings
    (NB03, area_ratio_credited_to_the_interpolation, "we compare how much of it leaks through the membrane",
     "how much of it leaks through the membrane is what separates the couplings", True, False),
    (NB03, area_ratio_credited_to_the_interpolation, "comes from the interpolation", "comes from the blob",
     True, False),
    (NB03, area_ratio_credited_to_the_interpolation, "delta, stream fn. 400 steps", "delta, stream fn. skipped",
     False, True),
    (NB03, area_ratio_credited_to_the_interpolation, "about 8 times better", "about 339 times better", True, False),
    # the header's promise
    (NB03, header_names_where_numbers_come_from, "or measured with this code and labelled as such, with its source",
     "or, in the one place where the run is too long for a notebook, measured with its source", True, False),
    # the cylinder README pointer and the marker pattern
    (NB02, cylinder_marker_pattern_backed, "a tether 32 times stiffer still", "a tether 32 times stiffer, springs between neighbours, still",
     True, False),
    (NB02, cylinder_marker_pattern_backed, "0.08 cells out of place", "0.24 cells out of place", True, False),
    (NB02, cylinder_marker_pattern_backed, "a tether 32 times stiffer", "a tether 10 times stiffer", True, False),
    (NB02, cylinder_marker_pattern_backed, "neighbouring markers moving in opposite directions",
     "markers moving", False, True),
    # the bicubic run back on the ring, or the bicubic option tied to the paper
    (NB03, ring_runs_show_default_against_delta, "size of the final area error, delta / blob:",
     "size of the final area error, delta / (blob, bicubic): 1.0;  delta / blob:", False, True),
    (NB03, bicubic_described_neutrally, "The other option, `\"bicubic\"`, interpolates",
     "The other option [tone-check sample: paper], `\"bicubic\"`, interpolates", True, False),
    (NB03, bicubic_described_neutrally, "### 5.1 The bicubic stencil",
     "### 5.1 The bicubic stencil [tone-check sample: Cortez]", True, False),
    # the section 11 heading and the form of eq. 28
    (NB01, section_11_says_what_pyibm_changes, "## 11. What pyIBM changes from the 2008 code",
     "## 11. Notes on the 2008 code", True, False),
    (NB02, eq_28_and_rotational_update_stated_right, "(eq. 28, in the form his code",
     "(eq. 28, in a form his code", True, False),
    # the tone check, with a made-up sample of its wording at a neutral heading
    (NB03, no_criticism_of_sources, "## 1. Why project the singular force",
     "## 1. Why project the singular force [tone-check sample: in our hands]", True, False),
    (NB01, no_criticism_of_sources, "## 1. The periodic lattice: `pyibm.grid.Grid`",
     "## 1. The periodic lattice: `pyibm.grid.Grid` [tone-check sample: misprint]", True, False),
    (NB01, no_criticism_of_sources, "## 9. Target points as tethers: `pyibm.structures`",
     "## 9. Target points as tethers: `pyibm.structures` [tone-check sample: too soft]", True, False),
    (NB02, no_criticism_of_sources, "## 2. The MAC grid: where $u$, $v$ and $p$ live",
     "## 2. The MAC grid: where $u$, $v$ and $p$ live [tone-check sample: printed equation omits]",
     True, False),
    # a quoted test that does not exist
    (NB03, pointers_to_tests_exist, "test_static_leak_tangential_force", "test_static_leak_tangential_force_gone",
     True, False),
]


@pytest.mark.parametrize("name, check, old, new, markdown, outputs", MUTATIONS,
                         ids=[f"{m[0][:2]}-{m[1].__name__}-{i}" for i, m in enumerate(MUTATIONS)])
def test_checks_catch_the_errors_they_target(name, check, old, new, markdown, outputs):
    nb = _load(name)
    check(nb)                                          # the saved notebook passes ...
    with pytest.raises(AssertionError):                # ... and the mutated copy does not
        check(_mutated(nb, old, new, markdown, outputs))
