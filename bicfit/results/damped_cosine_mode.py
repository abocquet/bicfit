import numpy as np
from dataclasses import dataclass

from .common import Mode, Result
from ..models import _damped_cosine_model
from ..types import FloatLike


@dataclass(eq=True, frozen=True)
class DampedCosineMode(Mode):
    amplitude: float
    phase: float

    def __call__(self, t: FloatLike) -> FloatLike:
        return _damped_cosine_model(
            t,
            0.0,
            np.array([self.amplitude]),
            np.array([self.phase]),
            np.array([self.w]),
            np.array([self.kappa]),
        )


@dataclass
class DampedCosineResult(Result):
    offset: float
    amplitudes: np.ndarray[float]
    phases: np.ndarray[float]
    ws: np.ndarray[float]
    kappas: np.ndarray[float]

    @property
    def modes(self) -> list[DampedCosineMode]:
        return [
            DampedCosineMode(amplitude=amplitude, phase=phase, w=w, kappa=kappa)
            for amplitude, phase, w, kappa in zip(
                self.amplitudes, self.phases, self.ws, self.kappas
            )
        ]

    def __call__(self, t: FloatLike) -> FloatLike:
        return _damped_cosine_model(
            t, self.offset, self.amplitudes, self.phases, self.ws, self.kappas
        )

    def __repr__(self):
        return f"DampedCosineResult(offset={self.offset}, modes={self.modes})"

    def pretty_repr(self):
        if len(self.amplitudes) == 1:
            return f"offset = {self.offset:0.2f}, amplitude = {self.amplitudes[0]:0.2e}, phase = {self.phases[0]:0.2f}, w = {self.ws[0]:0.2e}, kappa = {self.kappas[0]:0.2e}"
        else:
            return f"offset = {self.offset:0.2f}, {len(self.amplitudes)} modes"
