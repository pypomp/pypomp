"""Verify that the version is consistent everywhere before a release.

Run by the ``guard`` job in ``.github/workflows/release.yml``, which is the first
thing a release does. It is deliberately fast (no dependency install beyond
``packaging``) so a mistake costs ~15 seconds rather than a full matrix run.

Two trigger paths are supported:

``workflow_dispatch`` (preferred)
    The version is typed into the Run-workflow form as a confirmation. The tag
    does not exist yet -- the release creates it after everything passes -- so
    this path additionally requires that ``v<version>`` is *not* already taken
    and that HEAD is on ``main``.

``push`` of a ``v*`` tag (legacy)
    The tag already exists, so it is compared against ``pyproject.toml``.
    Comparison uses :class:`packaging.version.Version` rather than string
    equality, so the 4-component tags used through ``v0.4.7.1`` still match a
    3-component ``pyproject.toml`` version (``Version("0.4.8.0") ==
    Version("0.4.8")``).

Release candidates and other PEP 440 pre-releases are supported. For those,
``CITATION.bib`` and ``README.md`` are intentionally *not* required to match:
those files carry citation metadata, and citing ``0.4.9rc1`` would be wrong.

Usage (also runnable locally for testing):

    python tools/check_version.py                 # reads env, as in CI
    python tools/check_version.py 0.4.9           # simulate a dispatch
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent

# Both CITATION.bib and README.md carry the version in a BibTeX field:
#   version = {0.4.8},
BIBTEX_VERSION_RE = re.compile(r"^(\s*version\s*=\s*\{)([^}]*)(\}.*)$", re.MULTILINE)

# pyproject.toml's [project] version. Anchored to the start of a line so it
# cannot match `required-version` or a dependency specifier.
PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

CITATION_FILES = ("CITATION.bib", "README.md")


class CheckError(Exception):
    """A version inconsistency that should fail the release."""


def read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = PYPROJECT_VERSION_RE.search(text)
    if match is None:
        raise CheckError('could not find `version = "..."` in pyproject.toml')
    return match.group(1)


def read_bibtex_version(relative_path: str) -> str:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    match = BIBTEX_VERSION_RE.search(text)
    if match is None:
        raise CheckError(
            f"could not find a `version = {{...}}` field in {relative_path}"
        )
    return match.group(2).strip()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise CheckError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def check_citation_files(project_version: str) -> list[str]:
    """Return a list of human-readable problems (empty if all agree)."""
    problems: list[str] = []
    for relative_path in CITATION_FILES:
        found = read_bibtex_version(relative_path)
        if found != project_version:
            problems.append(
                f"{relative_path} says {found!r}, pyproject.toml says {project_version!r}"
            )
    return problems


def check_dispatch(project_version: str, requested: str) -> None:
    if Version(requested) != Version(project_version):
        raise CheckError(
            f"the version you entered ({requested!r}) does not match "
            f"pyproject.toml ({project_version!r}).\n"
            f"Run `make bump VERSION={requested}` and commit before releasing."
        )

    tag = f"v{project_version}"
    if tag_exists(tag):
        raise CheckError(
            f"tag {tag} already exists. Version numbers on PyPI can never be "
            f"reused, so bump to a new version rather than re-releasing this one."
        )

    # The release publishes whatever is checked out; make sure that is main.
    head = git("rev-parse", "HEAD")
    try:
        git("merge-base", "--is-ancestor", head, "origin/main")
    except CheckError:
        raise CheckError(
            f"HEAD ({head[:8]}) is not an ancestor of origin/main. "
            f"Releases must be cut from main."
        ) from None


def check_tag_push(project_version: str, ref_name: str) -> None:
    tag_version = ref_name.lstrip("v")
    try:
        parsed = Version(tag_version)
    except InvalidVersion:
        raise CheckError(f"tag {ref_name!r} is not a valid PEP 440 version") from None
    if parsed != Version(project_version):
        raise CheckError(
            f"tag {ref_name!r} does not match pyproject.toml ({project_version!r})"
        )


def main(argv: list[str]) -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    requested = argv[1] if len(argv) > 1 else os.environ.get("INPUT_VERSION", "")

    try:
        project_version = read_pyproject_version()
        try:
            parsed_project = Version(project_version)
        except InvalidVersion:
            raise CheckError(
                f"pyproject.toml version {project_version!r} is not valid PEP 440"
            ) from None

        if event_name == "push":
            ref_name = os.environ.get("GITHUB_REF_NAME", "")
            check_tag_push(project_version, ref_name)
            tag = ref_name
        else:
            if not requested:
                raise CheckError(
                    "no version supplied. Pass one as an argument, or set INPUT_VERSION."
                )
            try:
                Version(requested)
            except InvalidVersion:
                raise CheckError(
                    f"{requested!r} is not a valid PEP 440 version"
                ) from None
            check_dispatch(project_version, requested)
            tag = f"v{project_version}"

        # Citation metadata is only expected to track final releases.
        if parsed_project.is_prerelease:
            print(
                f"{project_version} is a pre-release; "
                f"skipping {' and '.join(CITATION_FILES)} checks."
            )
        else:
            problems = check_citation_files(project_version)
            if problems:
                raise CheckError(
                    "version files disagree:\n  "
                    + "\n  ".join(problems)
                    + f"\nRun `make bump VERSION={project_version}` and commit."
                )

    except CheckError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    print(f"Version {project_version} is consistent. Tag: {tag}")
    if parsed_project.is_prerelease:
        print("This is a pre-release; the GitHub Release will be marked as such.")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"tag={tag}\n")
            handle.write(f"version={project_version}\n")
            handle.write(f"prerelease={str(parsed_project.is_prerelease).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
