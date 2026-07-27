#!/usr/bin/env python3
"""Bump the hardcoded pypomp version string in pyproject.toml, CITATION.bib, and README.md.

Usage: bump_version.py <new-version>   (e.g. bump_version.py 1.5.0)
"""

import re
import sys

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def read_current_version(path="pyproject.toml"):
    with open(path) as f:
        text = f.read()
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if not match:
        sys.exit(f"Could not find a version line in {path}")
    return match.group(1)


def bump(path, pattern, old, new):
    with open(path) as f:
        text = f.read()
    new_text, count = pattern.subn(lambda m: m.group(0).replace(old, new), text)
    if count == 0:
        sys.exit(f"Version string {old!r} not found in {path}")
    with open(path, "w") as f:
        f.write(new_text)
    print(f"{path}: {old} -> {new} ({count} occurrence{'s' if count != 1 else ''})")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: bump_version.py <new-version>")
    new_version = sys.argv[1]
    if not VERSION_RE.match(new_version):
        sys.exit(f"Version must look like X.Y.Z, got {new_version!r}")

    old_version = read_current_version()
    if old_version == new_version:
        sys.exit(f"New version {new_version} matches current version, nothing to do")

    bump(
        "pyproject.toml",
        re.compile(r'^version = "' + re.escape(old_version) + r'"$', re.MULTILINE),
        old_version,
        new_version,
    )
    bump(
        "CITATION.bib",
        re.compile(r"version = \{" + re.escape(old_version) + r"\},"),
        old_version,
        new_version,
    )
    bump(
        "README.md",
        re.compile(r"version = \{" + re.escape(old_version) + r"\},"),
        old_version,
        new_version,
    )


if __name__ == "__main__":
    main()
