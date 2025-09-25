import numpy as np

from .types import FloatLike

"""
    Models are functions that describe the behavior of a system over time.
    They are used both in data fitting and in the generation of synthetic data for the plots.
    In post-processing, we use "adapter" functions to map the array of parameters to the functions arguments.
"""


def _complex_exponential_model(
    t: float | np.ndarray[float],
    offset: complex,
    amplitudes: np.ndarray[complex],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
) -> np.ndarray[complex]:
    t = np.array(t)
    return offset + np.sum(
        amplitudes[:, None] * np.exp((1j * pulsations[:, None] - decay_rates[:, None]) * t),
        axis=0,
    ).reshape(t.shape)


def _exponential_model(
    t: float | np.ndarray[float],
    offset: complex,
    amplitudes: np.ndarray[FloatLike],
    decay_rates: np.ndarray[float],
) -> np.ndarray[FloatLike]:
    t = np.array(t)
    return offset + np.sum(
        amplitudes[:, None] * np.exp(-decay_rates[:, None] * t), axis=0
    ).reshape(t.shape)


def _damped_cosine_model(
    t: float | np.ndarray[float],
    offset: float,
    amplitudes: np.ndarray[float],
    phases: np.ndarray[float],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
) -> np.ndarray[float]:
    t = np.array(t)
    return offset + np.sum(
        amplitudes[:, None]
        * np.cos(phases[:, None] + pulsations[:, None] * t)
        * np.exp(-decay_rates[:, None] * t),
        axis=0,
    ).reshape(t.shape)
