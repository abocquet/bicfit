import numpy as np
from dataclasses import dataclass
from typing import List

from .common import Mode, Result
from ..models import _complex_exponential_model
from ..types import FloatLike


@dataclass(eq=True, frozen=True)
class ComplexMode(Mode):
    complex_amplitude: complex

    @property
    def amplitude(self):
        return np.abs(self.complex_amplitude)

    @property
    def phase(self):
        return np.angle(self.complex_amplitude)

    def __call__(self, t: FloatLike) -> FloatLike:
        return _complex_exponential_model(
            t,
            0,
            np.array([self.complex_amplitude]),
            np.array([self.pulsation]),
            np.array([self.decay_rate]),
        )


@dataclass
class ComplexResult(Result):
    amplitudes: np.ndarray[complex]
    pulsations: np.ndarray[float]
    decay_rates: np.ndarray[float]

    def __post_init__(self):
        order = np.argsort(self.decay_rates)
        self.amplitudes = self.amplitudes[order]
        self.pulsations = self.pulsations[order]
        self.decay_rates = self.decay_rates[order]

    @property
    def modes(self) -> List[ComplexMode]:
        return [
            ComplexMode(complex_amplitude=amplitude, pulsation=pulsation, decay_rate=decay_rate)
            for amplitude, pulsation, decay_rate in zip(self.amplitudes, self.pulsations, self.decay_rates)
        ]

    @property
    def frequencies(self) -> np.ndarray[float]:
        return self.pulsations / (2 * np.pi)

    def __call__(self, t: FloatLike) -> np.ndarray[complex]:
        return _complex_exponential_model(
            t, self.offset, self.amplitudes, self.pulsations, self.decay_rates
        )

    def __repr__(self):
        return f"ComplexResult(offset={self.offset}, modes={self.modes})"

    def pretty_repr(self):
        if len(self.amplitudes) == 1:
            return f"offset = {self.offset.real:0.2f} + {self.offset.imag:0.2f}j, amplitude = {self.amplitudes[0].real:0.2e}+{self.amplitudes[0].imag:0.2e}j,  pulsation = {self.pulsations[0]:0.2e}, decay_rate / 2pi = {self.decay_rates[0] / (2*np.pi):0.2e}"
        else:
            return f"offset = {self.offset.real:0.2f} + {self.offset.imag:0.2f}j, {len(self.amplitudes)} modes"

