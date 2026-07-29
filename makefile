.PHONY: test-light test-heavy test-all test-cpu-scaling bump check-version

test-light:
	.venv/bin/pytest -m "not heavy"

test-heavy:
	.venv/bin/pytest -m "heavy"

test-all:
	.venv/bin/pytest

# Wall-clock CPU parallel-scaling checks. They time particle filters, so they
# need the machine to themselves: `-n 0` keeps xdist from filling every core
# with other tests. They are not collected at all without PYPOMP_CPU_SCALING=1
# (see tests/test_cpu_parallel/conftest.py), which is why `make test-all` does
# not pick them up.
test-cpu-scaling:
	PYPOMP_CPU_SCALING=1 .venv/bin/pytest tests/test_cpu_parallel -n 0 -s

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
