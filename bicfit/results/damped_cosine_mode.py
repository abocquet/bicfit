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
            np.array([self.pulsation]),
            np.array([self.decay_rate]),
        )


@dataclass
class DampedCosineResult(Result):
    offset: float
    amplitudes: np.ndarray[float]
    phases: np.ndarray[float]
    pulsations: np.ndarray[float]
    decay_rates: np.ndarray[float]

    @property
    def modes(self) -> list[DampedCosineMode]:
        return [
            DampedCosineMode(amplitude=amplitude, phase=phase, pulsation=pulsation, decay_rate=decay_rate)
            for amplitude, phase, pulsation, decay_rate in zip(
                self.amplitudes, self.phases, self.pulsations, self.decay_rates
            )
        ]

    @property
    def frequencies(self) -> np.ndarray[float]:
        return self.pulsations / (2 * np.pi)

    def __call__(self, t: FloatLike) -> FloatLike:
        return _damped_cosine_model(
            t, self.offset, self.amplitudes, self.phases, self.pulsations, self.decay_rates
        )

    def __repr__(self):
        return f"DampedCosineResult(offset={self.offset}, modes={self.modes})"

    def pretty_repr(self):
        if len(self.amplitudes) == 1:
            return f"offset = {self.offset:0.2f}, amplitude = {self.amplitudes[0]:0.2e}, phase = {self.phases[0]:0.2f}, pulsation = {self.pulsations[0]:0.2e}, decay_rate / 2pi = {self.decay_rates[0] / (2*np.pi):0.2e}"
        else:
            return f"offset = {self.offset:0.2f}, {len(self.amplitudes)} modes"
