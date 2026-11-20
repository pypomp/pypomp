"""Run tests on a rented Modal GPU and exit with pytest's status.

CI's opt-in `gpu` job uses this, since GitHub's hosted runners have no GPU new
enough for the CuTe backend (compute capability 8.0+). Needs a Modal token in
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, or from `modal token new`.

Usage: modal run tools/modal_gpu_tests.py [--pytest-args "tests/test_cute -x"]

Writes coverage-gpu.xml to the working directory.
"""

import re
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/pypomp"

# Dependencies get their own layer, so it is rebuilt only when pyproject.toml
# changes rather than on every commit. Containers re-import this file from
# /root, away from the repository, and only need the already-built image.
DEPS: list[str] = []
if modal.is_local():
    with open(ROOT / "pyproject.toml", "rb") as f:
        _project = tomllib.load(f)["project"]
    DEPS = [
        *_project["dependencies"],
        *_project["optional-dependencies"]["tests"],
        *_project["optional-dependencies"]["cute"],
        "jax[cuda12]",
    ]

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_pip_install(*DEPS)
    .add_local_dir(
        ROOT,
        REMOTE,
        copy=True,
        ignore=[
            ".git",
            "**/.venv",
            ".claude",
            "scratch",
            "dist",
            "docs",
            "**/__pycache__",
            "**/.pytest_cache",
        ],
    )
    .run_commands(f"uv pip install --system --no-deps -e {REMOTE}")
    .workdir(REMOTE)
)

app = modal.App("pypomp-gpu-tests", image=image)

# JAX's compilation cache outlives the container here, since compiling is over
# half of a run with an empty cache. The volume holds one tarball per GPU model
# (cache keys include the device, so models share no entries) rather than the
# cache directory itself: with a max size set (tests/conftest.py), every JAX
# cache write scans the whole directory, and per-file operations on a volume
# are slow enough to make that dominate the run.
cache = modal.Volume.from_name("pypomp-jax-cache", create_if_missing=True)
CACHE = "/jax-cache"
LOCAL_CACHE = Path(".pytest_cache/jax_cache")
# Leaves time to save the cache before Modal's own timeout, so a slow run still
# warms the next one.
PYTEST_TIMEOUT = 80 * 60


def _gpu_name() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    name = out.splitlines()[0].strip() if out.strip() else "unknown"
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def _restore_cache(tarball: Path) -> None:
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    if tarball.exists():
        with tarfile.open(tarball) as tar:
            tar.extractall(LOCAL_CACHE, filter="data")
        print(f"Restored JAX cache from {tarball}")


def _save_cache(tarball: Path, since_ns: int) -> None:
    # Keeps only the entries this run used, so the cache stays at one run's
    # working set.
    subprocess.run(
        [sys.executable, "tools/prune_jax_cache.py", str(LOCAL_CACHE), str(since_ns)],
        check=False,
    )
    tmp = tarball.with_suffix(".tmp")
    with tarfile.open(tmp, "w") as tar:
        tar.add(LOCAL_CACHE, arcname=".")
    tmp.replace(tarball)
    cache.commit()


# L4 (Ada) is the cheapest GPU Modal offers that CuTe supports; A10 (Ampere) is
# the fallback when L4s are scarce.
@app.function(
    gpu=["L4", "A10"], cpu=4, memory=8192, timeout=90 * 60, volumes={CACHE: cache}
)
def run_pytest(args: list[str]) -> tuple[int, bytes]:
    subprocess.run(["nvidia-smi", "-L"], check=False)
    tarball = Path(CACHE) / f"{_gpu_name()}.tar"
    _restore_cache(tarball)
    since_ns = time.time_ns()
    # The CuTe tests skip themselves when the backend is unavailable, which
    # would pass the job without testing anything. Checked in a subprocess so
    # this process doesn't hold JAX's preallocated GPU memory during the tests.
    check = "from pypomp.cute import available; ok, why = available(); assert ok, why"
    if subprocess.run([sys.executable, "-c", check]).returncode:
        return 1, b""
    # One process: xdist workers would contend for the single GPU.
    cmd = [sys.executable, "-m", "pytest", "-n", "0", "--cov", "--cov-report=xml"]
    try:
        code = subprocess.run([*cmd, *args], timeout=PYTEST_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        print(f"pytest exceeded {PYTEST_TIMEOUT // 60} minutes")
        code = 1
    _save_cache(tarball, since_ns)
    cov = Path("coverage.xml")
    return code, cov.read_bytes() if cov.exists() else b""


@app.local_entrypoint()
def main(pytest_args: str = "tests/test_cute") -> None:
    code, cov = run_pytest.remote(shlex.split(pytest_args))
    if cov:
        Path("coverage-gpu.xml").write_bytes(cov)
    if code:
        sys.exit(code)
