from typing import Generic

import numpy as np
from dataclasses import dataclass

from ..types import FloatLike, F


@dataclass(eq=True, frozen=True)
class Mode:
    pulsation: float  # omega (pulsation)
    decay_rate: float

    @property
    def frequency(self):
        return 2 * np.pi * self.pulsation

    def __call__(self, t: FloatLike) -> FloatLike:
        raise NotImplementedError()


@dataclass
class Result(Generic[F]):
    offset: F

    times: np.ndarray[float]
    signal: np.ndarray[F]

    def plot(self):
        from ..plot import plot

        plot(self)
