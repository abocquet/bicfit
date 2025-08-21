import numpy as np
from dataclasses import dataclass
from typing import List, Union, Generic, TypeVar

FloatLike = float | np.ndarray


@dataclass(eq=True, frozen=True)
class Mode:
    w: float  # omega (pulsation)
    kappa: float

    @property
    def frequency(self):
        return 2 * np.pi * self.w

    def __call__(self, t: FloatLike) -> FloatLike:
        raise NotImplementedError()


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
        return self.complex_amplitude * np.exp((1j * self.w - self.kappa) * t)


@dataclass(eq=True, frozen=True)
class RealMode(Mode):
    amplitude: float
    phase: float

    def __call__(self, t: FloatLike) -> FloatLike:
        return (
            self.amplitude * np.cos(self.phase + self.w * t) * np.exp(-self.kappa * t)
        )


@dataclass(eq=True, frozen=True)
class Exponential:
    amplitude: float | complex
    kappa: float

    def __call__(self, t: FloatLike) -> FloatLike:
        return self.amplitude * np.exp(-self.kappa * t)


M = TypeVar("M", bound=Union[ComplexMode, RealMode, Exponential])


@dataclass
class Result(Generic[M]):
    offset: complex
    modes: List[M]

    times: np.ndarray
    signal: np.ndarray

    def __post_init__(self):
        self.modes = sorted(self.modes, key=lambda mode: mode.w)

    def __call__(self, t):
        return sum([mode(t) for mode in self.modes]) + self.offset

    def __repr__(self):
        return f"Result(offset={self.offset}, modes={self.modes})"

    def plot(self):
        from .plot import plot

        plot(self)
