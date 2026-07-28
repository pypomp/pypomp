"""Rewrite the package version everywhere it is hardcoded.

Invoked via ``make bump VERSION=x.y.z``. Three files hold the version literally:

``pyproject.toml``
    The source of truth (``[project] version``).
``CITATION.bib`` and ``README.md``
    BibTeX citation metadata (``version = {...}``).

``docs/source/conf.py`` reads the version out of ``pyproject.toml`` at build
time and needs no change.

Pre-releases (``0.4.9rc1``) only update ``pyproject.toml``: the citation files
should keep pointing at the last real release, since citing a release candidate
would be wrong. ``tools/check_version.py`` skips those files for pre-releases to
match.
"""

from __future__ import annotations

import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_version import (  # noqa: E402  (path set up above)
    BIBTEX_VERSION_RE,
    CITATION_FILES,
    PYPROJECT_VERSION_RE,
    read_pyproject_version,
)


def bump_pyproject(new_version: str) -> bool:
    path = REPO_ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    new_text, count = PYPROJECT_VERSION_RE.subn(
        f'version = "{new_version}"', text, count=1
    )
    if count != 1:
        raise SystemExit(
            "error: could not locate the [project] version in pyproject.toml"
        )
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def bump_bibtex(relative_path: str, new_version: str) -> bool:
    path = REPO_ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    new_text, count = BIBTEX_VERSION_RE.subn(rf"\g<1>{new_version}\g<3>", text)
    if count == 0:
        raise SystemExit(f"error: could not locate a version field in {relative_path}")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python tools/bump_version.py VERSION", file=sys.stderr)
        return 2

    new_version = argv[1]
    try:
        parsed = Version(new_version)
    except InvalidVersion:
        print(f"error: {new_version!r} is not a valid PEP 440 version", file=sys.stderr)
        return 1

    old_version = read_pyproject_version()
    if Version(old_version) == parsed and old_version != new_version:
        print(
            f"note: {new_version} and the current {old_version} are equivalent under "
            f"PEP 440, but the literal strings differ; rewriting anyway."
        )

    changed: list[str] = []
    if bump_pyproject(new_version):
        changed.append("pyproject.toml")

    if parsed.is_prerelease:
        print(
            f"{new_version} is a pre-release; leaving "
            f"{' and '.join(CITATION_FILES)} at the last released version."
        )
    else:
        for relative_path in CITATION_FILES:
            if bump_bibtex(relative_path, new_version):
                changed.append(relative_path)

    if not changed:
        print(f"Nothing to do: already at {new_version}.")
        return 0

    print(f"Bumped {old_version} -> {new_version} in:")
    for name in changed:
        print(f"  {name}")
    print("\nReview with `git diff`, then commit before releasing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
