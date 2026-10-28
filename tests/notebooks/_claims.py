"""Claims the tutorial text makes about what its own cells print, as checks on a notebook.

Each check takes a notebook as a plain dict (json, or nbformat's JSON) and raises
AssertionError when the text and the printed outputs disagree. ``test_tutorial_text.py`` runs
them on the saved notebooks; the slow ``test_0*.py`` tests run them again on a fresh execution,
so a change in the package that changes a printed number fails until the text follows.
The claims are the ones a review found wrong once:

- 03, section 7 quotes the wide-Laplacian far-field error that the section-4 table prints
  (it said 157 % while the table printed 155 %);
- 03, the gaussian4 mass inside a 2 delta cut is too large, not too small (it said "drops"),
  in the text and in the printed line;
- 03, section 6 credits the delta / blob area ratio to the interpolation, with a control run
  that spreads with the delta function and interpolates like the blob (it presented the ratio
  as the difference between the couplings), and the first cell names every place numbers
  measured outside the notebook come from;
- 03, the ellipse's minor and major axes are labelled the right way round;
- 02, the cylinder drag is quoted after the start-up ringing has died out (it was printed at
  t = 1, still ringing and about a fifth below the late value);
- 02, the marker pattern on the cylinder is described from printed numbers, and the pointer to
  the cylinder_wake README claims only what that README reports (it said springs were tried,
  and quoted a target error the README does not have);
- 02, eq. 28 is given in the form Seibold's code computes, and the rotational pressure update
  carries theta and rho;
- 01, the tether-number threshold in the text and output is the driver's rule of thumb, not a
  stability limit taken from Stockie & Wetton 1999;
- every tutorial names existing tests where it quotes them, and ends with the full references of
  the papers it cites.

And the tone: the tutorials say what pyIBM does and which form of each equation it uses, and they
do not comment on the papers, notes and codes they follow (01, section 11, says what pyIBM
changes from the 2008 code). In 03 the bicubic option is described on its own terms, not
attributed to the paper, and the elastic ring of section 6 runs pyIBM's default against the
delta function (with the control), not the bicubic option.
"""

from __future__ import annotations

import os
import re

NB01 = "01_peskin_mcqueen_1996_immersed_boundary.ipynb"
NB02 = "02_projection_method_mac_dirichlet.ipynb"
NB03 = "03_blob_projection_method.ipynb"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 03, section 6: the runs on the elastic ring, as the cell labels them
RING_RUNS = ("delta", "blob", "delta, stream fn.")

# Wording that comments on a source rather than saying what pyIBM does. None of it may appear
# in a tutorial's text, code or printed output.
CRITICISM = re.compile(
    r"misprint|\btypos?\b|\berrat(?:a|um)\b|mistranscri|in our hands|got wrong|\bbugs?\b|"
    r"too soft|\bwrong\b|\brecipe\b|paper-literal|paper's own (?:interpolation|recipe)|"
    r"\bomits?\b|printed equation|our inference|does not say|does not (?:give|reach) (?:its|their)|"
    r"\bunusable\b|\bunexplained\b|\bsloppy\b|\bblame|\bfail(?:s|ed)\b",
    re.IGNORECASE)


def _text(value):
    return value if isinstance(value, str) else "".join(value)


def _markdown(nb):
    return [_text(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]


def _code_cell(nb, needle):
    """(source, printed text) of the one code cell whose source contains ``needle``."""
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and needle in _text(c["source"])]
    assert len(hits) == 1, f"{len(hits)} code cells contain {needle!r}"
    cell = hits[0]
    printed = "".join(_text(o.get("text", "")) for o in cell.get("outputs", [])
                      if o.get("output_type") == "stream")
    return _text(cell["source"]), printed


def _number(pattern, text, what):
    m = re.search(pattern, text)
    assert m, f"{what}: no match for {pattern!r}"
    return [float(g) for g in m.groups()] if len(m.groups()) > 1 else float(m.group(1))


# ---------------------------------------------------------------------------------------------
# 03, the blob projection method
# ---------------------------------------------------------------------------------------------
def far_field_error_quoted_as_printed(nb):
    _, table = _code_cell(nb, "far (wide)")
    # columns: blob, delta/h, patch, far (tight), near (tight), far (wide), far (2 delta)
    far_wide = _number(r"(?m)^compact\s+3\s+\d+\s+\S+\s+\S+\s+(\S+)", table, "table row compact 3h")
    quoted = [float(m.group(1)) for md in _markdown(nb)
              for m in re.finditer(r"checkerboard error of\s+(\d+(?:\.\d+)?)\s*%", md)]
    assert quoted, "section 7 no longer quotes the checkerboard error"
    assert all(abs(q - 100 * far_wide) < 0.5 + 1e-9 for q in quoted), (quoted, far_wide)


def gaussian4_cut_keeps_too_much_mass(nb):
    _, printed = _code_cell(nb, "def phi8")
    disc = _number(r"F\(2 delta\) = ([\d.]+)", printed, "disc mass")
    square = _number(r"square \|x\|, \|y\| <= 2 delta = ([\d.]+)", printed, "square mass")
    assert disc > 1.0 and square > 1.0           # the gaussian4 tail beyond sqrt(2) delta is negative
    excess = {f"{100 * (disc - 1):.1f}", f"{100 * (square - 1):.1f}"}    # "5.5" and "3.2"
    pct = "(?:" + "|".join(re.escape(e) for e in sorted(excess)) + r")\s*%"
    # the printed line says the disc keeps too much, with the percentage it computes
    line = [s for s in printed.splitlines() if "F(2 delta)" in s]
    assert len(line) == 1 and "too much mass" in line[0], line
    assert f"keeps {100 * (disc - 1):.1f} % too much mass" in line[0], line
    md = " ".join(" ".join(_markdown(nb)).split())
    # the text says "too much" at least once, always with the disc's printed excess ...
    too_much = list(re.finditer(r"(\d+\.\d)\s*% too much", md))
    assert too_much, "the text no longer says the 2 delta disc keeps too much mass"
    for m in too_much:
        assert abs(float(m.group(1)) - round(100 * (disc - 1), 1)) < 1e-9, m.group(0)
    # ... and nowhere, in the text or the printed output, that the cut loses mass
    wrong = (r"(?:too little|drops|loses|lost|short|lacks|missing|less)[^.]{0,80}" + pct,
             pct + r"\s*(?:too little|less|short|of the mass is lost|missing)")
    for text in (md, printed):
        for w in wrong:
            m = re.search(w, text, re.IGNORECASE)
            assert m is None, m.group(0)
    for m in re.finditer(r"F\(2\\delta\) = (1\.\d+)", md):
        assert abs(float(m.group(1)) - disc) < 6e-4, m.group(0)


def area_ratio_credited_to_the_interpolation(nb):
    """Section 6 runs a control (delta spreading with the stream-function interpolant), and the
    text says the delta / blob ratio comes from the interpolation, not the blob."""
    _, printed = _code_cell(nb, "def params_for")
    final, largest = {}, {}
    for label in RING_RUNS:
        m = re.search(r"(?m)^" + re.escape(label) + r"\s+\d+ steps .*?relative area error at t = [\d.]+ is "
                      r"([+-][\d.e+-]+), max over the run ([\d.e+-]+)", printed)
        assert m, f"no area line for {label!r}"
        final[label], largest[label] = float(m.group(1)), float(m.group(2))
    ratio = _number(r"\(delta, stream fn\.\) / blob: ([\d.]+)", printed, "control ratio")
    ratio_max = _number(r"largest area error over the run, \(delta, stream fn\.\) / blob: ([\d.]+)",
                        printed, "control ratio, largest")
    assert abs(ratio - abs(final["delta, stream fn."] / final["blob"])) < 0.02 * ratio + 0.01
    assert abs(ratio_max - largest["delta, stream fn."] / largest["blob"]) < 0.02 * ratio_max + 0.01
    # "the same size": the control is within a factor 3 of the blob; the plain delta run is not
    assert 1 / 3 < ratio < 3 and 1 / 3 < ratio_max < 3, (ratio, ratio_max)
    assert abs(final["delta"]) > 30 * abs(final["blob"])
    # "opposite signs"
    assert final["delta, stream fn."] * final["blob"] < 0, final
    md = " ".join(" ".join(_markdown(nb)).split())    # one line, so a re-wrap does not matter
    assert "what separates the couplings" not in md
    assert "comes from the interpolation" in md
    # the text quotes the 128^2 control and the blob's own gain from those numbers
    m = re.search(r"\$-?(\d\.\d)\\times10\^\{-6\}\$, at most \$(\d\.\d)\\times10\^\{-5\}\$ over the run, "
                  r"with delta spreading and the stream-function interpolant", md)
    assert m, "section 6 no longer quotes the 128^2 control run"
    blob_end = re.search(r"\$\+(\d\.\d)\\times10\^\{-6\}\$ with the stream-function interpolant, at most "
                         r"\$(\d\.\d)\\times10\^\{-6\}\$", md)
    assert blob_end, "section 6 no longer quotes the 128^2 blob run"
    end_x = float(m.group(1)) / float(blob_end.group(1))
    max_x = 10 * float(m.group(2)) / float(blob_end.group(2))
    gains = re.search(r"about (\d+) times better than the delta function at \$t = 1\$ and (\d+) times better "
                      r"in the largest error", md)
    assert gains, "section 6 no longer states the blob's own gain"
    assert abs(float(gains.group(1)) - end_x) < 0.6 and abs(float(gains.group(2)) - max_x) < 0.6, \
        (gains.groups(), end_x, max_x)


def ring_runs_show_default_against_delta(nb):
    """Section 6 runs the delta function, the blob with pyIBM's default interpolation and the
    control (delta spreading, stream-function interpolation); the bicubic option is not run on
    the ring, and no code cell outside section 5 uses it."""
    source, printed = _code_cell(nb, "def params_for")
    labels = re.findall(r"(?m)^(\S.*?)\s+\d+ steps in ", printed)
    assert tuple(labels) == RING_RUNS, labels
    assert "bicubic" not in source and "bicubic" not in printed
    section5 = ("def test_field", "def solenoidal_field")
    for c in nb["cells"]:
        src = _text(c["source"])
        if c["cell_type"] == "code" and not any(s in src for s in section5):
            assert "bicubic" not in src, src[:80]
    md = _markdown(nb)
    start = [i for i, s in enumerate(md) if s.startswith("## 6.")]
    end = [i for i, s in enumerate(md) if s.startswith("## 7.")]
    assert len(start) == 1 and len(end) == 1
    assert not any("bicubic" in s for s in md[start[0]:end[0]])


def bicubic_described_neutrally(nb):
    """The bicubic option is described for what it does, not attributed to the paper, and the
    text says why the default is the stream function."""
    md = " ".join(" ".join(_markdown(nb)).split())
    m = re.search(r"(?:paper|Cortez|Minion)[^.]{0,200}bicubic|bicubic[^.]{0,200}(?:paper|Cortez|Minion)", md)
    assert m is None, m.group(0)
    assert "### 5.2 Why pyibm interpolates the stream function" in md
    assert "the curl of an interpolated stream function is exactly divergence-free" in md
    assert re.search(r"`\"bicubic\"`,? interpolates \$u\$ and \$v\$ directly", md)
    assert md.count("divergence-free only to $O(h^2)$") >= 2          # sections 5.2 and 7


def header_names_where_numbers_come_from(nb):
    """The first cell promises where every quoted number comes from; it may not say that one
    section is the only place with numbers measured outside the notebook, since sections 4 and 7
    quote the module docstrings and tests."""
    header = " ".join(_markdown(nb)[0].split())
    assert "the one place" not in header
    assert "sections 4 and 7" in header and "section 6" in header


def ellipse_axes_labelled_right(nb):
    source, printed = _code_cell(nb, "def ellipse_points")
    a, b = _number(r"a, b = ([\d.]+), ([\d.]+)", source, "semi-axes")
    sigma = _number(r"sigma = ([\d.]+)", source, "sigma")
    assert "centre[0] + a * np.cos(th), centre[1] + b * np.sin(th)" in source   # a along x
    semi = {"a": a, "b": b}
    labels = list(re.finditer(r"(minor|major) axis \(([xy]) = 0\.5 -/\+ ([ab])\), "
                              r"kappa = ([ab])/([ab])\^2: ([\d.]+)", printed))
    assert len(labels) == 2, printed
    for m in labels:
        axis, coord, end, num, den, value = m.groups()
        assert (coord, end) in (("x", "a"), ("y", "b"))
        other = "b" if end == "a" else "a"
        assert (num, den) == (end, other)              # curvature at the end of semi-axis s: s/t^2
        assert axis == ("minor" if semi[end] < semi[other] else "major")
        assert abs(float(value) - sigma * semi[end] / semi[other] ** 2) < 6e-4


# ---------------------------------------------------------------------------------------------
# 02, the MAC projection solver
# ---------------------------------------------------------------------------------------------
def cylinder_drag_quoted_after_the_ringing(nb):
    _, printed = _code_cell(nb, "cylinder in a channel")
    swings = {(float(a), float(b)): float(s) for s, a, b in
              re.findall(r"([\d.]+) % for ([\d.]+) <= t <= ([\d.]+)", printed)}
    assert swings[(0.85, 1.0)] > 2.0                   # "still swings by several percent"
    assert swings[(1.5, 2.0)] < 1.0                    # "down to a fraction of a percent"
    last = _number(r"over the last second F_x changes by ([+-][\d.]+) %", printed, "last second")
    assert abs(last) < 1.0                             # "only a fraction of a percent"
    below = _number(r"F_x at t = 1 is (\d+) % below the final value", printed, "t = 1 value")
    assert 15 <= below <= 25                           # "about a fifth below it"
    assert "nothing changes in time any more" not in "\n".join(_markdown(nb))


def cylinder_marker_pattern_backed(nb):
    """The marker pattern on the cylinder is described from what the cell prints, and the
    pointer to examples/cylinder_wake/README.md says only what that README says."""
    _, printed = _code_cell(nb, "cylinder in a channel")
    along, across = _number(r"largest displacement along the ring ([\d.]+) cells, across it ([\d.]+) cells",
                            printed, "displacement")
    assert along > 2 * across                          # "mostly along the ring, not across it"
    pairs = _number(r"opposite directions along the ring: (\d+) % of the pairs", printed, "opposite pairs")
    assert 58 <= pairs <= 75                           # "about two thirds"
    wavelength = _number(r"a wavelength of ([\d.]+) cells", printed, "wavelength")
    assert 0.8 <= wavelength <= 1.7                    # "about a cell long"
    md = " ".join(" ".join(_markdown(nb)).split())
    assert "springs between neighbours" not in md      # the README never tried springs
    # every number the pointer quotes is in the README it points to
    pointer = re.search(r"`examples/cylinder_wake/README\.md` describes.*?\.(?:\s|$)", md)
    assert pointer, "the pointer to the cylinder_wake README is gone"
    with open(os.path.join(ROOT, "examples", "cylinder_wake", "README.md"), encoding="utf-8") as fh:
        readme = " ".join(fh.read().split())
    assert "1 to 1.7 cells" in pointer.group(0) and "1 to 1.7 cells" in readme
    quoted = re.findall(r"(?<![\w.])(\d+\.\d+|\d{4,})(?![\w.])", pointer.group(0))
    assert {"0.64", "0.70", "0.08", "2000"} <= set(quoted), quoted
    for q in quoted:
        assert re.search(r"(?<![\w.])" + re.escape(q) + r"(?![\w])", readme), q
    stiffer = re.search(r"a tether (\d+) times stiffer", pointer.group(0))
    assert stiffer and int(stiffer.group(1)) * 2000 == 64000 and "k = 64000" in readme


def eq_28_and_rotational_update_stated_right(nb):
    md = _markdown(nb)
    eq28 = [s for s in md if r"\gamma = \min" in s]
    assert eq28 and all("in the form his code computes" in " ".join(s.split()) for s in eq28)
    assert all(r"|/h_x" in s and r"|/h_y" in s for s in eq28)
    rot = [line for s in md for line in s.splitlines() if r"p^{\,n+1/2} = p^{\,n-1/2}" in line]
    assert rot and all(r"\theta" in line and r"\rho" in line for line in rot), rot
    assert not any("second order in both velocity and pressure" in s for s in md)


# ---------------------------------------------------------------------------------------------
# 01, Peskin & McQueen 1996
# ---------------------------------------------------------------------------------------------
def tether_threshold_is_the_drivers_rule_of_thumb(nb):
    from pyibm.driver import TETHER_STABLE

    _, printed = _code_cell(nb, "TETHER_STABLE")
    target, tether = _number(r"warns above ([\d.]+) for target points and ([\d.]+) for mass tethers",
                             printed, "tether thresholds")
    assert target == TETHER_STABLE[("target", "pm1996")]
    assert tether == TETHER_STABLE[("mass tether", "pm1996")]
    md = "\n".join(_markdown(nb))
    assert not re.search(r"\\rho\\?,?\s*\\Delta x\)\s*>\s*0\.5", md)
    assert "stability limit; Stockie" not in md
    assert "rule of thumb" in md


def section_11_says_what_pyibm_changes(nb):
    """Section 11 is about what pyIBM changes from the 2008 code, and the forms of eqs. 14.28,
    14.42 and 14.46 are stated as the forms pyIBM uses."""
    md = _markdown(nb)
    s11 = [s for s in md if s.startswith("## 11.")]
    assert len(s11) == 1 and s11[0].startswith("## 11. What pyIBM changes from the 2008 code"), s11
    assert "so the plate follows its path" in s11[0]
    assert "the wrap handles every position" in s11[0]
    text = " ".join(" ".join(md).split())
    assert r"We write the pressure term as $+D^0 p^{n+1}$, the sign it has in eq. 14.40" in text
    assert r"leaves $\Delta t$ on the advection term and $\Delta t/\rho$ on the force" in text
    assert "The sine in the Laplacian symbol is squared" in text


# ---------------------------------------------------------------------------------------------
# all three
# ---------------------------------------------------------------------------------------------
def no_criticism_of_sources(nb):
    """No wording that comments on a source, in the text, the code or what it prints."""
    hits = []
    for i, c in enumerate(nb["cells"]):
        texts = [_text(c["source"])] + [_text(o.get("text", "")) for o in c.get("outputs", [])
                                         if o.get("output_type") == "stream"]
        for t in texts:
            hits += [(i, m.group(0)) for m in CRITICISM.finditer(t)]
    assert hits == [], hits


def pointers_to_tests_exist(nb):
    """Numbers the text quotes from the test suite name their source; that source must exist."""
    for md in _markdown(nb):
        for m in re.finditer(r"(tests/[\w/]+\.py)(?:::(\w+))?", md):
            path = os.path.join(ROOT, *m.group(1).split("/"))
            assert os.path.isfile(path), m.group(0)
            if m.group(2):
                with open(path, encoding="utf-8") as fh:
                    assert re.search(r"(?m)^def " + m.group(2) + r"\(", fh.read()), m.group(0)


def ends_with_references(nb):
    refs = [s for s in _markdown(nb) if s.startswith("## References")]
    assert len(refs) == 1
    entries = [line for line in refs[0].splitlines() if line.startswith("- ")]
    # every entry: authors, title, where it appeared, and the year in brackets
    assert len(entries) >= 3 and all(re.search(r"\((?:[^()]*\s)?(?:19|20)\d\d\)", e) for e in entries)


CHECKS = {
    NB01: [tether_threshold_is_the_drivers_rule_of_thumb, section_11_says_what_pyibm_changes,
           no_criticism_of_sources, pointers_to_tests_exist, ends_with_references],
    NB02: [cylinder_drag_quoted_after_the_ringing, cylinder_marker_pattern_backed,
           eq_28_and_rotational_update_stated_right, no_criticism_of_sources, pointers_to_tests_exist,
           ends_with_references],
    NB03: [far_field_error_quoted_as_printed, gaussian4_cut_keeps_too_much_mass,
           area_ratio_credited_to_the_interpolation, ring_runs_show_default_against_delta,
           bicubic_described_neutrally, header_names_where_numbers_come_from,
           ellipse_axes_labelled_right, no_criticism_of_sources, pointers_to_tests_exist,
           ends_with_references],
}


def check_claims(name, nb):
    """Run every check of tutorial ``name`` on ``nb`` (a dict)."""
    for check in CHECKS[name]:
        check(nb)
