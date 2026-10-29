"""Run tests on a rented Modal GPU and exit with pytest's status.

CI's opt-in `gpu` job uses this, since GitHub's hosted runners have no GPU new
enough for the CuTe backend (compute capability 8.0+). Needs a Modal token in
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, or from `modal token new`.

Usage: modal run tools/modal_gpu_tests.py [--pytest-args "tests/test_cute -x"]

Writes coverage-gpu.xml to the working directory.
"""

import shlex
import subprocess
import sys
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
# half of a run with an empty cache. tests/conftest.py caps it at 5 GiB.
cache = modal.Volume.from_name("pypomp-jax-cache", create_if_missing=True)
CACHE = "/jax-cache"


# L4 (Ada) is the cheapest GPU Modal offers that CuTe supports; A10 (Ampere) is
# the fallback when L4s are scarce.
@app.function(
    gpu=["L4", "A10"], cpu=4, memory=8192, timeout=90 * 60, volumes={CACHE: cache}
)
def run_pytest(args: list[str]) -> tuple[int, bytes]:
    subprocess.run(["nvidia-smi", "-L"], check=False)
    Path(".pytest_cache").mkdir(exist_ok=True)
    Path(".pytest_cache/jax_cache").symlink_to(CACHE)
    # The CuTe tests skip themselves when the backend is unavailable, which
    # would pass the job without testing anything. Checked in a subprocess so
    # this process doesn't hold JAX's preallocated GPU memory during the tests.
    check = "from pypomp.cute import available; ok, why = available(); assert ok, why"
    if subprocess.run([sys.executable, "-c", check]).returncode:
        return 1, b""
    # One process: xdist workers would contend for the single GPU.
    code = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", "0", "--cov", "--cov-report=xml", *args]
    ).returncode
    cache.commit()
    cov = Path("coverage.xml")
    return code, cov.read_bytes() if cov.exists() else b""


@app.local_entrypoint()
def main(pytest_args: str = "tests/test_cute") -> None:
    code, cov = run_pytest.remote(shlex.split(pytest_args))
    if cov:
        Path("coverage-gpu.xml").write_bytes(cov)
    if code:
        sys.exit(code)
