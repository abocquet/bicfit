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
            np.array([self.w]),
            np.array([self.kappa]),
        )


@dataclass
class ComplexResult(Result):
    amplitudes: np.ndarray[complex]
    ws: np.ndarray[float]
    kappas: np.ndarray[float]

    def __post_init__(self):
        order = np.argsort(self.kappas)
        self.amplitudes = self.amplitudes[order]
        self.ws = self.ws[order]
        self.kappas = self.kappas[order]

    @property
    def modes(self) -> List[ComplexMode]:
        return [
            ComplexMode(complex_amplitude=amplitude, w=w, kappa=kappa)
            for amplitude, w, kappa in zip(self.amplitudes, self.ws, self.kappas)
        ]

    def __call__(self, t: FloatLike) -> np.ndarray[complex]:
        return _complex_exponential_model(
            t, self.offset, self.amplitudes, self.ws, self.kappas
        )

    def __repr__(self):
        return f"ComplexResult(offset={self.offset}, modes={self.modes})"
