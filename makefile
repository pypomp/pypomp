.PHONY: install_requirements

install_requirements: .venv
	pip install -e ".[tests,benchmarks,viz]"

.venv:
	python3.12 -m venv .venv

.PHONY: test-light test-heavy test-all

test-light:
	.venv/bin/pytest -m "not heavy"

test-heavy:
	.venv/bin/pytest -m "heavy"

test-all:
	.venv/bin/pytest
