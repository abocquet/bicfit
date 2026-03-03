import functools as ft
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize

from .models import _damped_cosine_model, _complex_exponential_model, _exponential_model
from .results import DampedCosineResult
from .types import FloatLike

# =====================================================================================================================
# Post fit options
# =====================================================================================================================


@dataclass
class NoOffset:
    pass


_PostFitOptions = None | NoOffset

# =====================================================================================================================
# Common functions
# =====================================================================================================================


def _cost(
    parameters: np.ndarray,
    times: np.ndarray,
    signal: np.ndarray,
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    model_signal = model(times, parameters)
    return np.sum(np.abs(model_signal - signal) ** 2)


# =====================================================================================================================
# Complex exponential model
# =====================================================================================================================


def _complex_exponential_adapter(
    t: np.ndarray[float], x: np.ndarray[FloatLike], is_there_offset: bool = True
):
    if is_there_offset:
        offset = x[0] + 1j * x[1]
        x = x[2:]
    else:
        offset = complex(0.0)
    modes = x.reshape(-1, 4)

    amplitudes_re, amplitudes_im, pulsations, decay_rates = (
        modes[:, 0],
        modes[:, 1],
        modes[:, 2],
        modes[:, 3],
    )
    decay_rates = np.abs(decay_rates)  # enforce the positivity of the decay rate
    amplitudes = amplitudes_re + 1j * amplitudes_im
    return _complex_exponential_model(t, offset, amplitudes, pulsations, decay_rates)


def _post_fit_complex_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
    options: _PostFitOptions,
) -> Tuple[complex, np.ndarray[complex], np.ndarray[float], np.ndarray[float]]:
    assert len(amplitudes) == len(pulsations) == len(decay_rates)
    is_there_offset = not isinstance(options, NoOffset)

    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(_complex_exponential_adapter, is_there_offset=is_there_offset),
    )

    x0_offset = []
    if is_there_offset:
        x0_offset = [offset.real, offset.imag]
    x0_modes = np.stack(
        (amplitudes.real, amplitudes.imag, pulsations, decay_rates)
    ).T.flatten()
    x0 = np.concatenate((x0_offset, x0_modes))

    xopt = minimize(cost, x0).x

    if is_there_offset:
        offset = xopt[0] + 1j * xopt[1]
        xopt = xopt[2:]
    else:
        offset = complex(0.0)
    amplitudes_re, amplitudes_im, pulsations, decay_rates = xopt.reshape(-1, 4).T
    amplitudes = amplitudes_re + 1j * amplitudes_im
    decay_rates = np.abs(decay_rates)  # Enforce the positivity of the decay rates

    return offset, amplitudes, pulsations, decay_rates


# =====================================================================================================================
# Real exponential model
# =====================================================================================================================


def _exponential_adapter(
    t: np.ndarray[float],
    x: np.ndarray[FloatLike],
    is_complex: bool,
    is_there_offset: bool = True,
) -> np.ndarray[FloatLike]:
    if is_complex:
        if is_there_offset:
            offset = complex(x[0] + 1j * x[1])
            x = x[2:]
        else:
            offset = complex(0.0)
        modes = x.reshape(-1, 3)
        amplitudes_re, amplitudes_im, decay_rates = (
            modes[:, 0],
            modes[:, 1],
            modes[:, 2],
        )
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        if is_there_offset:
            offset = complex(x[0])
            x = x[1:]
        else:
            offset = complex(0.0)
        modes = x.reshape(-1, 2)
        amplitudes, decay_rates = modes[:, 0], modes[:, 1]

    decay_rates = np.abs(decay_rates)  # Enforce the positivity of the decay rate
    return _exponential_model(t, offset, amplitudes, decay_rates)


def _post_fit_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    offset: complex,
    amplitudes: np.ndarray[complex],
    decay_rates: np.ndarray[complex],
    is_complex: bool,
    options: _PostFitOptions,
) -> Tuple[FloatLike, np.ndarray[FloatLike], np.ndarray[FloatLike]]:
    assert len(amplitudes) == len(decay_rates)
    is_there_offset = not isinstance(options, NoOffset)

    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(
            _exponential_adapter, is_complex=is_complex, is_there_offset=is_there_offset
        ),
    )

    x0_offset = []
    if is_there_offset:
        if is_complex:
            x0_offset = [offset.real, offset.imag]
        else:
            x0_offset = [offset.real]
    if is_complex:
        x0_modes = np.stack((amplitudes.real, amplitudes.imag, decay_rates)).T.flatten()
    else:
        x0_modes = np.stack((amplitudes.real, decay_rates)).T.flatten()
    x0 = np.concatenate((x0_offset, x0_modes))

    xopt = minimize(cost, x0).x

    offset = complex(0.0)
    if is_there_offset:
        if is_complex:
            offset = xopt[0] + 1j * xopt[1]
            xopt = xopt[2:]
        else:
            offset = xopt[0]
            xopt = xopt[1:]
    if is_complex:
        amplitudes_re, amplitudes_im, decay_rates = xopt.reshape(-1, 3).T
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        amplitudes, decay_rates = xopt.reshape(-1, 2).T
    decay_rates = np.abs(decay_rates)  # Enforce the positivity of the decay rates

    return offset, amplitudes, decay_rates


# =====================================================================================================================
# Damped cosine model
# =====================================================================================================================


def _damped_cosine_adapter(
    t: np.ndarray[float], x: np.ndarray[float], is_there_offset: bool = True
) -> np.ndarray[float]:
    if is_there_offset:
        offset = x[0]
        x = x[1:]
    else:
        offset = 0.0
    modes = x.reshape(-1, 4)
    amplitudes, phases, pulsations, decay_rates = (
        modes[:, 0],
        modes[:, 1],
        modes[:, 2],
        modes[:, 3],
    )

    decay_rates = np.abs(decay_rates)  # Enforce the positivity of the decay rate
    return _damped_cosine_model(t, offset, amplitudes, phases, pulsations, decay_rates)


def _post_fit_damped_cosine(
    times: np.ndarray[float],
    signal: np.ndarray[float],
    offset: complex,
    amplitudes: np.ndarray[float],
    phases: np.ndarray[float],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
    options: _PostFitOptions,
) -> DampedCosineResult:
    is_there_offset = not isinstance(options, NoOffset)

    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(_damped_cosine_adapter, is_there_offset=is_there_offset),
    )

    x0_offset = []
    if is_there_offset:
        x0_offset = [offset]
    x0_modes = np.stack((amplitudes, phases, pulsations, decay_rates)).T.flatten()
    x0 = np.concatenate((x0_offset, x0_modes))

    xopt = minimize(cost, x0).x

    offset = 0.0
    if is_there_offset:
        offset = xopt[0]
        xopt = xopt[1:]
    amplitudes, phases, pulsations, decay_rates = xopt.reshape(-1, 4).T
    decay_rates = np.abs(decay_rates)  # Enforce the positivity of the decay rate

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
