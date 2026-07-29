# Developer Guide

This document provides instructions for developers contributing to the Pypomp repository.

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

# Install the package in editable mode with all development, testing, and benchmark extras
pip install -e ".[tests,benchmarks,viz]"
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

We use [Pyright](https://github.com/microsoft/pyright) for type checking. The configuration lives in the `[tool.pyright]` table of `pyproject.toml`, which sets `typeCheckingMode = "standard"`:

```bash
# Install pyright if not already installed
pip install pyright

# Run type checking on the repository
pyright
```

### Pre-commit Hooks

We use `pre-commit` to automatically run code formatting and linting checks on every git commit.

To set up pre-commit in your local repository:
```bash
# Install the git hook scripts
pre-commit install

# (Optional) Run the hooks manually against all files
pre-commit run --all-files
```

Once installed, the hooks configured in `.pre-commit-config.yaml` will run automatically whenever you commit changes.

> [!IMPORTANT]
> The pre-commit hooks include a local **Pyright** type-checking step. Because this hook runs in your system shell, **you must execute `git commit` from an active virtual environment** (`source .venv/bin/activate`) so that `pyright` can resolve third-party imports (like `jax`, `numpy`, etc.) and find the `pyright` executable. If the virtual environment is not active, the commit check will fail.

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

## 5. Continuous Integration

Two workflows cover everything:

**`.github/workflows/ci.yml`** runs on every push to `main` and on pull requests:

| Job | What it does |
| --- | --- |
| `checks` | `pre-commit run --all-files` (ruff lint, ruff format, pyright) |
| `test` | `pytest -m "not heavy"` on Python 3.12 and 3.14, Ubuntu and macOS |
| `build` | Builds the sdist and wheel, `twine check --strict`, installs the wheel into a clean venv and imports it |
| `docs` | Builds the Sphinx documentation |

Because `checks` runs pre-commit, CI executes the exact tool versions pinned in
`.pre-commit-config.yaml`. To upgrade a linter, run `pre-commit autoupdate` — CI follows
automatically, with no workflow edit.

Consecutive pushes cancel superseded runs, so only the latest commit is tested.

To run the full Python matrix (3.10–3.14) or the heavy tests without cutting a release, use
**Actions → CI → Run workflow** and tick `full-matrix` and/or `run-heavy`. Doing this before a
release is worthwhile, since heavy tests do not run on ordinary pushes.

---

## 6. Releases & Publishing to PyPI

Releases are driven by `.github/workflows/release.yml` using **PyPI Trusted Publishing**. The
release workflow *creates* the tag once everything has passed, rather than being triggered by one.
If a release fails at any stage, no tag is created — fix the problem, commit normally, and run it
again. There is no tag to delete.

### Cutting a release

1. **Bump the version.** `make bump` rewrites `pyproject.toml`, `CITATION.bib` and `README.md`
   (`docs/source/conf.py` reads the version out of `pyproject.toml` and needs no edit):
   ```bash
   make bump VERSION=0.4.9
   git diff                      # confirm the three files
   git commit -am "chore(release): v0.4.9"
   git push origin main          # ordinary CI run; wait for green
   ```
2. **Start the release.** Go to **Actions → Release → Run workflow**, and type the version
   (`0.4.9`) to confirm. This is checked against `pyproject.toml`; it is a confirmation, not a
   source of truth.
3. **Approve.** Once the guard and the full test suite pass, the run pauses for manual approval.
   Click **Approve** to publish, or **Cancel** to abort with nothing created.

The tag is derived as `v` + the `pyproject.toml` version, so version `0.4.9` produces tag `v0.4.9`.
Keep the two in sync by putting whatever form you want in `pyproject.toml` — a 4-component version
(`0.4.9.0`) produces a 4-component tag.

### What runs, in order

```
guard          version consistency; tag must not already exist; HEAD must be on main
  ↓
ci.yml         full 3.10–3.14 matrix on Ubuntu and macOS, heavy tests included,
               plus checks / build / docs
  ↓
publish        ⏸ pauses for manual approval
  ↓            downloads the distributions ci.yml already built and smoke-tested
               attests build provenance
               creates and pushes the tag        ← first side effect
               uploads to PyPI                   ← irreversible
               creates the GitHub Release
```

Every side effect is at the end, ordered by how hard it is to undo.

### Release candidates

Pre-release versions work without any special handling:

```bash
make bump VERSION=0.4.9rc1
```

`pip install pypomp` will **not** pick these up — pip ignores pre-releases unless you pass `--pre`
or pin the exact version — so they are safe to publish to real PyPI. Two differences from a final
release: `make bump` leaves `CITATION.bib` and `README.md` alone (citing a release candidate would
be wrong, and the guard skips those checks to match), and the GitHub Release is marked as a
pre-release so it is not shown as "Latest".

### If something fails

Nothing is tagged or published until you click Approve, so any earlier failure costs only time.
Fix it, commit, and run the workflow again.

The one exception is the PyPI upload itself. **A PyPI version number can never be reused**, even
after deleting the release. If the upload fails partway, re-running is safe (`skip-existing` fills
in whatever is missing). If it succeeds with the wrong contents, there is no fix: bump to `0.4.9.1`
and release again, optionally yanking the bad version on PyPI.

You can verify the version files at any time without starting a release:

```bash
make check-version VERSION=0.4.9
```
