# Developer Guide

This document provides instructions for developers contributing to the **pypomp** repository.

---

## 1. Local Environment Setup

We recommend setting up a virtual environment using Python 3.14 (the primary target version configured for type checking). However, Python versions `[3.10, 3.11, 3.12, 3.13]` are also fully supported.

To create and configure your local development environment:

```bash
# Clone the repository and navigate into the root directory
git clone https://github.com/pypomp/pypomp.git
cd pypomp

# Create the virtual environment
python3.14 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install main requirements
pip install -r requirements.txt

# Install the package in editable mode with all development, testing, and benchmark extras
pip install -e .[tests,benchmarks,viz]
```

---

## 2. Formatting, Linting, and Type Checking

We enforce strict formatting, linting, and type checking rules to maintain code quality. All checks must pass before pushing code to `main`.

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
# Install ruff if not already installed
pip install ruff

# Check for lint errors
ruff check .

# Automatically fix lint issues and format imports/code
ruff format .
```

### Static Type Checking

We use [Pyright](https://github.com/microsoft/pyright) for type checking. It is run in strict mode as configured in `pyproject.toml`:

```bash
# Install pyright if not already installed
pip install pyright

# Run type checking on the repository
pyright
```

---

## 3. Running Tests

We use `pytest` for unit testing. Our test suite is configured with `pytest-xdist` to run tests in parallel across CPU cores.

```bash
# Run the entire test suite
pytest

# Measure code coverage
pytest --cov
```

> [!NOTE]
> - A JAX persistent compilation cache is configured under `.pytest_cache/jax_cache` to speed up subsequent test runs.
> - Ensure you have installed the package in editable mode (`pip install -e .[tests,benchmarks,viz]`) so that code coverage is measured against the active source files.

---

## 4. Building Documentation

The documentation is written using Sphinx and is automatically built and hosted at [pypomp.readthedocs.io](https://pypomp.readthedocs.io) after each push to the `main` branch.

To build the HTML documentation locally:

```bash
# Navigate to the docs directory
cd docs

# Build HTML pages using Make
make html
```

---

## 5. Releases & Publishing to PyPI

We use an automated release workflow powered by GitHub Actions (`.github/workflows/publish.yml`) and **PyPI Trusted Publishing**.

When you are ready to publish a new release to PyPI:

1. **Update the Version**: Bump the package version in the following files:
   - `pyproject.toml` (under the `[project]` table)
   - `docs/source/conf.py` (the `release` variable)
   - `CITATION.bib` (in the citation metadata block)
   - `README.md` (in the BibTeX citation block)
2. **Commit and Tag**: Commit the changes and tag the release commit using a version tag matching `v[0-9]*`:
   ```bash
   git add pyproject.toml docs/source/conf.py CITATION.bib README.md
   git commit -m "vX.Y.Z Increment version number"
   git tag vX.Y.Z
   ```
3. **Push to GitHub**: Push your commit and the tag to the GitHub repository:
   ```bash
   git push origin main
   git push origin vX.Y.Z
   ```

After pushing the tag, a reviewer will receive a notification to review the tag push. Once approved, the GitHub Actions workflow will run. The workflow will:
- Run the full test suite.
- Build the source distribution and binary wheel.
- Safely upload the artifacts to PyPI via Trusted Publishing.
