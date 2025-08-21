from typing import Callable

import numpy as np
import functools as ft

from scipy.optimize import minimize

from .types import Result, ComplexMode, RealMode, Exponential

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


def _complex_exponential_model(t: np.ndarray, x: np.ndarray):
    offset_re, offset_im, modes = x[0], x[1], x[2:].reshape(-1, 4)
    offset = offset_re + 1j * offset_im

    amplitudes_re, amplitudes_im, ws, kappas = (
        modes[:, 0],
        modes[:, 1],
        modes[:, 2],
        modes[:, 3],
    )

    return offset + np.sum(
        (amplitudes_re + 1j * amplitudes_im) * np.exp((1j * ws - kappas) * t[:, None]),
        axis=1,
    )


def _post_fit_complex_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    result: Result[ComplexMode],
) -> Result:
    cost = ft.partial(
        _cost, times=times, signal=signal, model=_complex_exponential_model
    )

    x0 = [result.offset.real, result.offset.imag]
    bounds = [NO_BOUND, NO_BOUND]  # No bounds for offset
    for mode in result.modes:
        x0.extend(
            [
                mode.complex_amplitude.real,
                mode.complex_amplitude.imag,
                mode.w,
                mode.kappa,
            ]
        )
        bounds.extend([NO_BOUND, NO_BOUND, NO_BOUND, POSITIVE_BOUND])
    x0 = np.array(x0)

    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0] + 1j * xopt[1]

    modes = []
    for amplitude_re, amplitude_im, w, kappa in xopt[2:].reshape(-1, 4):
        mode = ComplexMode(
            complex_amplitude=amplitude_re + 1j * amplitude_im, w=w, kappa=kappa
        )
        modes.append(mode)

    new_result = Result(
        offset=offset,
        modes=modes,
        times=times,
        signal=signal,
    )

    return new_result


# =====================================================================================================================
# Real exponential model
# =====================================================================================================================


def _exponential_model(t: np.ndarray, x: np.ndarray, is_complex: bool):
    if is_complex:
        offset = x[0] + 1j * x[1]
        modes = x[2:].reshape(-1, 3)
        amplitudes_re, amplitudes_im, kappas = modes[:, 0], modes[:, 1], modes[:, 2]
        amplitudes = amplitudes_re + 1j * amplitudes_im
    else:
        offset = x[0]
        modes = x[1:].reshape(-1, 2)
        amplitudes, kappas = modes[:, 0], modes[:, 1]

    return offset + np.sum(amplitudes * np.exp(-kappas * t[:, None]), axis=1)


def _post_fit_exponential(
    times: np.ndarray, signal: np.ndarray, result: Result[ComplexMode], is_complex: bool
) -> Result:
    cost = ft.partial(
        _cost,
        times=times,
        signal=signal,
        model=ft.partial(_exponential_model, is_complex=is_complex),
    )

    if is_complex:
        x0 = [result.offset.real, result.offset.imag]
        bounds = [NO_BOUND, NO_BOUND]  # No bounds for offset
        for mode in result.modes:
            x0.extend([mode.amplitude.real, mode.amplitude.imag, mode.kappa])
            bounds.extend([NO_BOUND, NO_BOUND, POSITIVE_BOUND])
    else:
        x0 = [result.offset]
        bounds = [NO_BOUND]  # No bounds for offset
        for mode in result.modes:
            x0.extend([mode.amplitude, mode.kappa])
            bounds.extend([NO_BOUND, POSITIVE_BOUND])

    x0 = np.array(x0)

    xopt = minimize(cost, x0, bounds=bounds).x
    modes = []
    if is_complex:
        offset = xopt[0] + 1j * xopt[1]
        for amplitude_re, amplitude_im, kappa in xopt[2:].reshape(-1, 3):
            mode = Exponential(amplitude=amplitude_re + 1j * amplitude_im, kappa=kappa)
            modes.append(mode)
    else:
        offset = xopt[0]
        for amplitude, kappa in xopt[1:].reshape(-1, 2):
            mode = Exponential(amplitude=amplitude.real, kappa=kappa)
            modes.append(mode)

    new_result = Result(
        offset=offset,
        modes=modes,
        times=times,
        signal=signal,
    )

    return new_result


# =====================================================================================================================
# Damped cosine model
# =====================================================================================================================


def _damped_cosine_model(t: np.ndarray, x: np.ndarray):
    offset, modes = x[0], x[1:].reshape(-1, 4)

    amplitudes, ws, phases, kappas = modes[:, 0], modes[:, 1], modes[:, 2], modes[:, 3]
    return offset + np.sum(
        amplitudes * np.cos(phases + ws * t[:, None]) * np.exp(-kappas * t[:, None]), axis=1
    )


def _post_fit_damped_cosine(
    times: np.ndarray,
    signal: np.ndarray,
    result: Result[RealMode],
) -> Result:
    cost = ft.partial(_cost, times=times, signal=signal, model=_damped_cosine_model)

    x0 = [result.offset]
    bounds = [NO_BOUND]
    for mode in result.modes:
        x0.extend([mode.amplitude, mode.w, mode.phase, mode.kappa])
        bounds.extend([NO_BOUND, POSITIVE_BOUND, NO_BOUND, POSITIVE_BOUND])
    x0 = np.array(x0)
    xopt = minimize(cost, x0, bounds=bounds).x
    offset = xopt[0]

    modes = []
    for amplitude, w, phase, kappa in xopt[1:].reshape(-1, 4):
        mode = RealMode(amplitude=amplitude, w=w, phase=phase, kappa=kappa)
        modes.append(mode)

    new_result = Result(
        offset=offset,
        modes=modes,
        times=times,
        signal=signal,
    )

    return new_result
