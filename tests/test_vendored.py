"""The report-design files vendored from ctdcast stay byte-identical to their recorded hashes.

``scripts/check_vendored.py`` hashes the six vendored files and compares them to ``VENDORED.txt``;
this test fails the suite if a vendored file is edited in place without re-vendoring from ctdcast.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_vendored.py"


def _load_check_vendored():
    """Import ``scripts/check_vendored.py`` as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("check_vendored", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_files_match_manifest():
    """`verify_local` reports zero drift — no vendored file was edited without re-vendoring."""
    assert _load_check_vendored().verify_local() == 0
