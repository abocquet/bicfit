import numpy as np
from dataclasses import dataclass
from typing import List, Generic

from .common import Result
from ..models import _exponential_model
from ..types import FloatLike, F


@dataclass(eq=True, frozen=True)
class ExponentialDecayMode(Generic[F]):
    amplitude: F
    decay_rate: float
    offset: F

    def __call__(self, t: FloatLike) -> FloatLike:
        return _exponential_model(
            t, 0.0, np.array([self.amplitude]), np.array([self.decay_rate])
        )


@dataclass
class ExponentialDecayResult(Result, Generic[F]):
    amplitudes: np.ndarray[F]
    decay_rates: np.ndarray[float]

    def __post_init__(self):
        order = np.argsort(self.decay_rates)
        self.amplitudes = self.amplitudes[order]
        self.decay_rates = self.decay_rates[order]

    @property
    def modes(self) -> List[ExponentialDecayMode]:
        return [
            ExponentialDecayMode(amplitude=amplitude, decay_rate=decay_rate, offset=self.offset)
            for amplitude, decay_rate in zip(self.amplitudes, self.decay_rates)
        ]

    def __call__(self, t: FloatLike) -> np.ndarray[FloatLike]:
        return _exponential_model(t, self.offset, self.amplitudes, self.decay_rates)

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

            return f"offset = {offset_str}, amplitude = {amplitude_str}, decay_rate / 2pi = {self.decay_rates[0] / (2*np.pi):0.2e}"
        else:
            return f"offset = {offset_str}, {len(self.amplitudes)} modes"
