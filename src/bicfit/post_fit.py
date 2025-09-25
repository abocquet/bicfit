import functools as ft
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize

from .models import _damped_cosine_model, _complex_exponential_model, _exponential_model
from .results import DampedCosineResult
from .types import FloatLike

NO_BOUND = (None, None)
POSITIVE_BOUND = (0, None)


def _cost(
    x: np.ndarray,
    times: np.ndarray,
    signal: np.ndarray,
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    model_signal = model(times, x)
    return np.sum(np.abs(model_signal - signal) ** 2)


# =====================================================================================================================
# Complex exponential model
# =====================================================================================================================


def _complex_exponential_adapter(t: np.ndarray[float], x: np.ndarray[FloatLike]):
    offset_re, offset_im, modes = x[0], x[1], x[2:].reshape(-1, 4)
    offset = offset_re + 1j * offset_im

    amplitudes_re, amplitudes_im, pulsations, decay_rates = (
        modes[:, 0],
        modes[:, 1],
        modes[:, 2],
        modes[:, 3],
    )
    amplitudes = amplitudes_re + 1j * amplitudes_im
    return _complex_exponential_model(t, offset, amplitudes, pulsations, decay_rates)


def _post_fit_complex_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
) -> Tuple[complex, np.ndarray[complex], np.ndarray[float], np.ndarray[float]]:
    assert len(amplitudes) == len(pulsations) == len(decay_rates)

    cost = ft.partial(
        _cost, times=times, signal=signal, model=_complex_exponential_adapter
    )

    x0 = [offset.real, offset.imag]
    x1 = np.stack((amplitudes.real, amplitudes.imag, pulsations, decay_rates)).T.flatten()
    x0 = np.concatenate((x0, x1))

    bounds = [NO_BOUND, NO_BOUND]  # No bounds for offset
    bounds += [NO_BOUND, NO_BOUND, NO_BOUND, POSITIVE_BOUND] * len(amplitudes)

    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0] + 1j * xopt[1]
    amplitudes_re, amplitudes_im, pulsations, decay_rates = xopt[2:].reshape(-1, 4).T
    amplitudes = amplitudes_re + 1j * amplitudes_im

    return offset, amplitudes, pulsations, decay_rates


# =====================================================================================================================
# Real exponential model
# =====================================================================================================================


def _exponential_adapter(
    t: np.ndarray[float], x: np.ndarray[FloatLike], is_complex: bool
) -> np.ndarray[FloatLike]:
    if is_complex:
        offset = complex(x[0] + 1j * x[1])
        modes = x[2:].reshape(-1, 3)
        amplitudes_re, amplitudes_im, decay_rates = modes[:, 0], modes[:, 1], modes[:, 2]
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        offset = complex(x[0])
        modes = x[1:].reshape(-1, 2)
        amplitudes, decay_rates = modes[:, 0], modes[:, 1]

    return _exponential_model(t, offset, amplitudes, decay_rates)


def _post_fit_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    decay_rates: np.ndarray[complex],
    is_complex: bool,
) -> Tuple[FloatLike, np.ndarray[FloatLike], np.ndarray[FloatLike]]:
    assert len(amplitudes) == len(decay_rates)

    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(_exponential_adapter, is_complex=is_complex),
    )

    if is_complex:
        x0 = [offset.real, offset.imag]
        x1 = np.stack((amplitudes.real, amplitudes.imag, decay_rates)).T.flatten()
        x0 = np.concatenate((x0, x1))

        bounds = [NO_BOUND, NO_BOUND] + [NO_BOUND, NO_BOUND, POSITIVE_BOUND] * len(
            amplitudes
        )
    else:
        x0 = [offset.real]
        x1 = np.stack((amplitudes.real, decay_rates)).T.flatten()
        x0 = np.concatenate((x0, x1))
        bounds = [NO_BOUND] + [NO_BOUND, POSITIVE_BOUND] * len(amplitudes)

    x0 = np.array(x0)

    xopt = minimize(cost, x0, bounds=bounds).x
    if is_complex:
        offset = xopt[0] + 1j * xopt[1]
        amplitudes_re, amplitudes_im, decay_rates = xopt[2:].reshape(-1, 3).T
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        offset = xopt[0]
        amplitudes, decay_rates = xopt[1:].reshape(-1, 2).T

    return offset, amplitudes, decay_rates


# =====================================================================================================================
# Damped cosine model
# =====================================================================================================================


def _damped_cosine_adapter(
    t: np.ndarray[float], x: np.ndarray[float]
) -> np.ndarray[float]:
    offset, modes = float(x[0]), x[1:].reshape(-1, 4)
    amplitudes, phases, pulsations, decay_rates = modes[:, 0], modes[:, 1], modes[:, 2], modes[:, 3]
    return _damped_cosine_model(t, offset, amplitudes, phases, pulsations, decay_rates)


def _post_fit_damped_cosine(
    times: np.ndarray[float],
    signal: np.ndarray[float],
    offset: complex,
    amplitudes: np.ndarray[float],
    phases: np.ndarray[float],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
) -> DampedCosineResult:
    cost = ft.partial(_cost, times=times, signal=signal, model=_damped_cosine_adapter)

    x0 = [offset]
    x1 = np.stack((amplitudes, phases, pulsations, decay_rates)).T.flatten()
    x0 = np.concatenate((x0, x1))

    bounds = [NO_BOUND]
    bounds += [NO_BOUND, NO_BOUND, POSITIVE_BOUND, POSITIVE_BOUND] * len(amplitudes)
    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0]

    amplitudes, phases, pulsations, decay_rates = xopt[1:].reshape(-1, 4).T

    new_result = DampedCosineResult(
        times=times,
        signal=signal,
        offset=offset,
        amplitudes=amplitudes,
        pulsations=pulsations,
        decay_rates=decay_rates,
        phases=phases,
    )

    return new_result
