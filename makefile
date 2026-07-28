.PHONY: test-light test-heavy test-all bump check-version

test-light:
	.venv/bin/pytest -m "not heavy"

test-heavy:
	.venv/bin/pytest -m "heavy"

test-all:
	.venv/bin/pytest

# Rewrite the version in pyproject.toml, CITATION.bib and README.md.
# Pre-releases (e.g. 0.4.9rc1) only touch pyproject.toml, since citing a
# release candidate would be wrong.
bump:
	@test -n "$(VERSION)" || { echo "usage: make bump VERSION=0.4.9"; exit 1; }
	.venv/bin/python tools/bump_version.py $(VERSION)

# The same check the release workflow runs first. Useful before releasing.
check-version:
	@test -n "$(VERSION)" || { echo "usage: make check-version VERSION=0.4.9"; exit 1; }
	.venv/bin/python tools/check_version.py $(VERSION)
