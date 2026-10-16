"""Print the runtime dependencies in pyproject.toml pinned to their floors.

The output is a requirements file, one pin per line.

CI's oldest-deps job installs these pins together, so every floor is tested in
one environment. A dependency without a plain ``>=`` floor is an error, since
the job could not test it.

Usage: python tools/dependency_floors.py > floors.txt
"""

import re
import sys
import tomllib


def main() -> None:
    with open("pyproject.toml", "rb") as f:
        deps: list[str] = tomllib.load(f)["project"]["dependencies"]
    pins: list[str] = []
    for dep in deps:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*>=\s*([^\s,;]+)", dep)
        if match is None:
            sys.exit(f"{dep!r} needs a plain '>=' floor for the oldest-deps job")
        pins.append(f"{match[1]}=={match[2]}")
    print("\n".join(pins))


if __name__ == "__main__":
    main()
