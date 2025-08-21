import numpy as np
from dataclasses import dataclass

from .common import Mode, Result
from ..models import _damped_cosine_model
from ..types import FloatLike


@dataclass(eq=True, frozen=True)
class RealMode(Mode):
    amplitude: float
    phase: float

    def __call__(self, t: FloatLike) -> FloatLike:
        return _damped_cosine_model(
            t, 0.0, np.array([self.amplitude]), np.array([self.phase]), np.array([self.w]), np.array([self.kappa])
        )


@dataclass
class RealResult(Result):
    offset: float
    amplitudes: np.ndarray[float]
    phases: np.ndarray[float]
    ws: np.ndarray[float]
    kappas: np.ndarray[float]

    @property
    def modes(self) -> list[RealMode]:
        return [
            RealMode(amplitude=amplitude, phase=phase, w=w, kappa=kappa)
            for amplitude, phase, w, kappa in zip(
                self.amplitudes, self.phases, self.ws, self.kappas
            )
        ]

    def __call__(self, t: FloatLike) -> FloatLike:
        return _damped_cosine_model(
            t, self.offset, self.amplitudes, self.phases, self.ws, self.kappas
        )

    def __repr__(self):
        return f"RealResult(offset={self.offset}, modes={self.modes})"
