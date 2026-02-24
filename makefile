.PHONY: install_requirements

install_requirements: .venv
	pip install -r requirements.txt

.venv:
	python3.12 -m venv .venv
