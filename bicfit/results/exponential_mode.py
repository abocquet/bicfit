import numpy as np
from dataclasses import dataclass
from typing import List, Generic, TypeVar

from .common import Result
from ..models import _exponential_model
from ..types import FloatLike, F


@dataclass(eq=True, frozen=True)
class ExponentialMode(Generic[F]):
    amplitude: F
    kappa: float
    offset: F

    def __call__(self, t: FloatLike) -> FloatLike:
        return _exponential_model(t, 0.0, np.array([self.amplitude]), np.array([self.kappa]))


@dataclass
class ExponentialResult(Result, Generic[F]):
    amplitudes: np.ndarray[F]
    kappas: np.ndarray[float]

    def __post_init__(self):
        order = np.argsort(self.kappas)
        self.amplitudes = self.amplitudes[order]
        self.kappas = self.kappas[order]

    @property
    def modes(self) -> List[ExponentialMode]:
        return [
            ExponentialMode(amplitude=amplitude, kappa=kappa, offset=self.offset)
            for amplitude, kappa in zip(self.amplitudes, self.kappas)
        ]

    def __call__(self, t: FloatLike) -> np.ndarray[FloatLike]:
        return _exponential_model(t, self.offset, self.amplitudes, self.kappas)

    def __repr__(self):
        return f"ExponentialResult(offset={self.offset}, modes={self.modes})"
