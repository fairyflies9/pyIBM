"""What a published tutorial may not carry, shared by the notebook tests (plain json, no nbformat)."""

from __future__ import annotations

import os
import re
import tempfile

def _both_slashes(path):
    """The path as given, with forward slashes and with backslashes, trailing separators off."""
    path = path.rstrip("\\/")
    return sorted({path, path.replace("\\", "/"), path.replace("/", "\\")})


def local_path_pattern(home, tmp):
    """The pattern of what a published notebook may not carry, for a machine whose home folder
    is ``home`` and whose temporary directory is ``tmp``.

    Generic: a Windows drive path ("C:\\..." or "C:/..."), a POSIX home or temp directory, the
    user folders, and a temporary directory notebook 01 creates (its prefix plus the random part
    that mkdtemp adds). For this machine: the full home and temporary paths with either
    separator, and the home folder's name right after a "Users" or "home" segment. The name on
    its own is not matched: a home folder called "user" or "data" would then match ordinary text
    such as "docs/user-guide.md".
    """
    name = re.split(r"[\\/]", home.rstrip("\\/"))[-1]
    machine = [re.escape(p) for p in _both_slashes(home) + _both_slashes(tmp) if len(p) > 1]
    if name:
        machine.append(r"(?:Users|home)[\\/]+" + re.escape(name) + r"(?![\w-])")
    return re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?=[\w.~-])|/home/|/Users/|\\Users\\|AppData|"
                      r"/tmp/|pyibm_tutorial\d+_[A-Za-z0-9_]{6,}|" + "|".join(machine),
                      re.IGNORECASE)


LOCAL_PATH = local_path_pattern(os.path.expanduser("~"), tempfile.gettempdir())

# The private design notes and working files, which are not published (the public docs are
# lower case: docs/user-guide.md, docs/numerics.md, docs/examples.md).
PRIVATE = re.compile(r"DESIGN\.md|STATUS\.md|EXAMPLES\.md|DRIVER_API\.md|\bnotes/|papers/|"
                     r"reproduction-notes|thesis-check|private/")


def texts(nb):
    """(where, text) for every cell source and every text output of a notebook given as a dict
    (json or nbformat); images are left out, since base64 can spell anything."""
    for i, cell in enumerate(nb["cells"]):
        source = cell.get("source", "")
        yield f"cell {i} source", source if isinstance(source, str) else "".join(source)
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                yield f"cell {i} output", text if isinstance(text, str) else "".join(text)
            elif out.get("output_type") == "error":
                yield f"cell {i} error", out.get("evalue", "")
            for mime, data in out.get("data", {}).items():
                if mime.startswith("text/"):
                    yield f"cell {i} {mime}", data if isinstance(data, str) else "".join(data)


def local_path_hits(nb, pattern=None):
    """Every match of ``pattern`` (default ``LOCAL_PATH``) in the sources and text outputs, as
    (where, match)."""
    pattern = LOCAL_PATH if pattern is None else pattern
    return [(where, m.group(0)) for where, text in texts(nb) for m in pattern.finditer(text)]


def private_hits(nb):
    """Every match of ``PRIVATE`` in the sources and text outputs, as (where, match)."""
    return [(where, m.group(0)) for where, text in texts(nb) for m in PRIVATE.finditer(text)]
