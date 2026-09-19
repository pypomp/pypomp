from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def coerce_seed(seed):

    if seed is None:
        return None
    values = np.asarray(seed).ravel()
    if not len(values):
        return None
    try:
        value = float(values[0])
        if not math.isfinite(value) or not -(2**31) < value < 2**31:
            raise ValueError
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "seed must coerce to a non-NA signed 32-bit integer"
        ) from error


class RUniform:
    def __init__(self):
        self._rng = None

    def seed(self, seed=None):
        value = coerce_seed(seed)
        if value is None:
            value = int(np.random.SeedSequence().generate_state(1)[0])
        words = []

        for index in range(675):
            value = (69069 * value + 1) & 0xFFFFFFFF
            if index > 50:
                words.append(value)
        self._rng = np.random.RandomState()
        self._rng.set_state(
            ("MT19937", np.asarray(words, dtype=np.uint32), 624, 0, 0.0)
        )

    def get_state(self):
        return None if self._rng is None else self._rng.get_state()

    def set_state(self, state):
        if state is None:
            self._rng = None
        else:
            if self._rng is None:
                self._rng = np.random.RandomState()
            self._rng.set_state(state)

    def uniform(self, low=0.0, high=1.0, size=None):
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError("uniform bounds must be finite with low <= high")
        if self._rng is None:
            self.seed()
        assert self._rng is not None
        if low == high:
            return np.full(size, low) if size is not None else float(low)
        words = self._rng.randint(0, 2**32, size=size, dtype=np.uint32)
        values = np.asarray(words, dtype=float) / 2**32

        values = np.where(values == 0, 0.5 * 2.328306437080797e-10, values)
        result = low + (high - low) * values
        return float(result) if size is None else result


r_uniform = RUniform()
_directory = ContextVar("pypomp_archive_directory", default=None)


@contextmanager
def archive_directory(path):

    token = _directory.set(path)
    try:
        yield
    finally:
        _directory.reset(token)


@dataclass
class ArchiveValue:
    value: Any
    ingredients: dict | None = None
    system_time: dict | None = None
