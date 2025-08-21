import numpy as np
from dataclasses import dataclass
from typing import List, Generic

from .common import Result
from ..models import _exponential_model
from ..types import FloatLike, F


@dataclass(eq=True, frozen=True)
class ExponentialDecayMode(Generic[F]):
    amplitude: F
    kappa: float
    offset: F

    def __call__(self, t: FloatLike) -> FloatLike:
        return _exponential_model(
            t, 0.0, np.array([self.amplitude]), np.array([self.kappa])
        )


@dataclass
class ExponentialDecayResult(Result, Generic[F]):
    amplitudes: np.ndarray[F]
    kappas: np.ndarray[float]

    def __post_init__(self):
        order = np.argsort(self.kappas)
        self.amplitudes = self.amplitudes[order]
        self.kappas = self.kappas[order]

    @property
    def modes(self) -> List[ExponentialDecayMode]:
        return [
            ExponentialDecayMode(amplitude=amplitude, kappa=kappa, offset=self.offset)
            for amplitude, kappa in zip(self.amplitudes, self.kappas)
        ]

    def __call__(self, t: FloatLike) -> np.ndarray[FloatLike]:
        return _exponential_model(t, self.offset, self.amplitudes, self.kappas)

    def __repr__(self):
        return f"ExponentialDecayResult(offset={self.offset}, modes={self.modes})"

    def pretty_repr(self):
        if np.iscomplex(self.offset):
            offset_str = f"offset = {self.offset.real:0.2f} + {self.offset.imag:0.2f}j"
        else:
            offset_str = f"offset = {self.offset:0.2f}"


        if len(self.amplitudes) == 1:
            if np.iscomplex(self.amplitudes[0]):
                amplitude_str = f"{self.amplitudes[0].real:0.2e} + {self.amplitudes[0].imag:0.2e}j"
            else:
                amplitude_str = f"{self.amplitudes[0]:0.2e}"

            return f"offset = {offset_str}, amplitude = {amplitude_str}, kappa = {self.kappas[0]:0.2e}"
        else:
            return f"offset = {offset_str}, {len(self.amplitudes)} modes"
