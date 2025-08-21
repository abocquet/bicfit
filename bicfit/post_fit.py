from typing import Callable

import numpy as np
import functools as ft

from scipy.optimize import minimize

from .models import _damped_cosine_model, _complex_exponential_model, _exponential_model
from .results import ComplexResult, RealResult, ExponentialResult
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

    amplitudes_re, amplitudes_im, ws, kappas = (
        modes[:, 0],
        modes[:, 1],
        modes[:, 2],
        modes[:, 3],
    )
    amplitudes = amplitudes_re + 1j * amplitudes_im
    return _complex_exponential_model(t, offset, amplitudes, ws, kappas)


def _post_fit_complex_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    ws: np.ndarray[complex],
    kappas: np.ndarray[complex],
) -> ComplexResult:
    assert len(amplitudes) == len(ws) == len(kappas)

    cost = ft.partial(
        _cost, times=times, signal=signal, model=_complex_exponential_adapter
    )

    x0 = [offset.real, offset.imag]
    x1 = np.stack((amplitudes.real, amplitudes.imag, ws, kappas)).T.flatten()
    x0 = np.concatenate((x0, x1))

    bounds = [NO_BOUND, NO_BOUND]  # No bounds for offset
    bounds += [NO_BOUND, NO_BOUND, NO_BOUND, POSITIVE_BOUND] * len(amplitudes)

    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0] + 1j * xopt[1]
    amplitudes_re, amplitudes_im, ws, kappas = xopt[2:].reshape(-1, 4).T
    amplitudes = amplitudes_re + 1j * amplitudes_im

    return offset, amplitudes, ws, kappas


# =====================================================================================================================
# Real exponential model
# =====================================================================================================================


def _exponential_adapter(
    t: np.ndarray[float], x: np.ndarray[FloatLike], is_complex: bool
) -> np.ndarray[FloatLike]:
    if is_complex:
        offset = x[0] + 1j * x[1]
        modes = x[2:].reshape(-1, 3)
        amplitudes_re, amplitudes_im, kappas = modes[:, 0], modes[:, 1], modes[:, 2]
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        offset = x[0]
        modes = x[1:].reshape(-1, 2)
        amplitudes, kappas = modes[:, 0], modes[:, 1]

    return _exponential_model(t, offset, amplitudes, kappas)


def _post_fit_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    kappas: np.ndarray[complex],
    is_complex: bool,
) -> ExponentialResult:
    assert len(amplitudes) == len(kappas)

    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(_exponential_adapter, is_complex=is_complex),
    )

    if is_complex:
        x0 = [offset.real, offset.imag]
        x1 = np.stack((amplitudes.real, amplitudes.imag, kappas)).T.flatten()
        x0 = np.concatenate((x0, x1))

        bounds = [NO_BOUND, NO_BOUND] + [NO_BOUND, NO_BOUND, POSITIVE_BOUND] * len(
            amplitudes
        )
    else:
        x0 = [offset.real]
        x1 = np.stack((amplitudes.real, kappas)).T.flatten()
        x0 = np.concatenate((x0, x1))
        bounds = [NO_BOUND] + [NO_BOUND, POSITIVE_BOUND] * len(amplitudes)

    x0 = np.array(x0)

    xopt = minimize(cost, x0, bounds=bounds).x
    if is_complex:
        offset = xopt[0] + 1j * xopt[1]
        amplitudes_re, amplitudes_im, kappas = xopt[2:].reshape(-1, 3).T
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        offset = xopt[0]
        amplitudes, kappas = xopt[1:].reshape(-1, 2).T

    new_result = ExponentialResult(
        offset=offset,
        times=times,
        signal=signal,
        amplitudes=amplitudes,
        kappas=kappas,
    )

    return new_result


# =====================================================================================================================
# Damped cosine model
# =====================================================================================================================


def _damped_cosine_adapter(
    t: np.ndarray[float], x: np.ndarray[float]
) -> np.ndarray[float]:
    offset, modes = x[0], x[1:].reshape(-1, 4)
    amplitudes, phases, ws, kappas = modes[:, 0], modes[:, 1], modes[:, 2], modes[:, 3]
    return _damped_cosine_model(t, offset, amplitudes, phases, ws, kappas)


def _post_fit_damped_cosine(
    times: np.ndarray[float],
    signal: np.ndarray[float],
    offset: complex,
    amplitudes: np.ndarray[float],
    phases: np.ndarray[float],
    ws: np.ndarray[float],
    kappas: np.ndarray[float],
) -> RealResult:
    cost = ft.partial(_cost, times=times, signal=signal, model=_damped_cosine_adapter)

    x0 = [offset]
    x1 = np.stack((amplitudes, phases, ws, kappas)).T.flatten()
    x0 = np.concatenate((x0, x1))

    bounds = [NO_BOUND]
    bounds += [NO_BOUND, NO_BOUND, POSITIVE_BOUND, POSITIVE_BOUND] * len(amplitudes)
    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0]

    amplitudes, phases, ws, kappas = xopt[1:].reshape(-1, 4).T

    new_result = RealResult(
        times=times,
        signal=signal,
        offset=offset,
        amplitudes=amplitudes,
        ws=ws,
        kappas=kappas,
        phases=phases,
    )

    return new_result
