# GEMINI.md

## Environment Setup
Always activate the virtual environment before running tests or commands:
```bash
source .venv/bin/activate
```

## Running Tests
To run all tests:
```bash
pytest test
```
You may need to run `source .venv/bin/activate && pip install -e .` first to update the package in the environment.

## Build and Package
This project uses `pypomp` as the main package.
