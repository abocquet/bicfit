from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .post_fit import (
    _post_fit_complex_exponential,
    _post_fit_damped_cosine,
    _post_fit_exponential,
)
from .results import ComplexResult, ExponentialDecayResult, DampedCosineResult


def fit_complex_exponential(
    times: np.ndarray[float],
    signal: np.ndarray[complex],
    n_modes: int = 1,
    with_post_fit: bool = True,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> ComplexResult:
    r"""
    Fits a complex exponential signal of the form
    $f(t) = x_0 + \sum_k A_k \exp((j\omega_k - \kappa_k) t)$

    :param times: np.ndarray[float]
        The time points at which the signal is sampled.
    :param signal: np.ndarray[complex]
        The complex signal to fit.
    :param n_modes: int
        The number of modes to fit. Each mode corresponds to a complex exponential term.
    :param with_post_fit: bool
        Whether to perform a post-fit refinement of the parameters, using least-square error minimization.
        This is recommended to improve the accuracy of the fit.
    :param tol: float
        The tolerance for the fitting process. It is used to determine the convergence of the fit.
    :param L_fraction: float
        The fraction of the signal length to use for the pencil method. Usually a value between 1/3 and 1/2 is
        recommended
    :return:
    """


    offset, amplitudes, pulsations, decay_rates = bicfit(
        times, signal, n_modes=n_modes, tol=tol, L_fraction=L_fraction
    )
    if with_post_fit:
        offset, amplitudes, pulsations, decay_rates = _post_fit_complex_exponential(
            times, signal, offset, amplitudes, pulsations, decay_rates
        )

    return ComplexResult(
        times=times,
        signal=signal,
        offset=offset,
        amplitudes=amplitudes,
        pulsations=pulsations,
        decay_rates=decay_rates,
    )


def fit_exponential_decay(
    times: np.ndarray[float],
    signal: np.ndarray[float | complex],
    n_modes: int = 1,
    with_post_fit: bool = True,
    is_complex: bool = False,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> ExponentialDecayResult:
    r""" Fits an exponential decay signal of the form
    $f(t) = x_0 + \sum_k A_k \exp(-\kappa_k t)$

    :param times: np.ndarray[float]
        The time points at which the signal is sampled.
    :param signal: np.ndarray[float | complex]
        The signal to fit. It can be real or complex.
    :param n_modes: int
        The number of modes to fit. Each mode corresponds to an exponential decay term.
    :param with_post_fit: bool
        Whether to perform a post-fit refinement of the parameters, using least-square error minimization.
        This is recommended to improve the accuracy of the fit.
    :param is_complex: bool
        Whether the signal is complex. If True, the with will take the $(A_k)$ complex, and real otherwise.
    :param tol: float
        The tolerance for the fitting process. It is used to determine the convergence of the fit.
    :param L_fraction: float
        The fraction of the signal length to use for the pencil method. Usually a value between 1/3 and 1/2 is
        recommended.
    """
    offset, amplitudes, pulsations, decay_rates = bicfit(
        times,
        signal,
        n_modes=n_modes,
        tol=tol,
        L_fraction=L_fraction,
    )
    if with_post_fit:
        offset, amplitudes, decay_rates = _post_fit_exponential(
            times, signal, offset, amplitudes, decay_rates, is_complex
        )
    else:
        if not is_complex:
            amplitudes = amplitudes.real
            offset = offset.real

    result = ExponentialDecayResult(
        times=times,
        signal=signal,
        offset=offset,
        amplitudes=amplitudes,
        decay_rates=decay_rates,
    )

    return result


def fit_damped_cosine(
    times: np.ndarray[float],
    signal: np.ndarray[float],
    n_modes: int = 1,
    with_post_fit: bool = True,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> DampedCosineResult:
    r"""
    Fits a damped cosine signal of the form
    $f(t) = x_0 + \sum_k A_k \exp(-\kappa_k t) \cos(\omega_k t + \phi_k)$

    :param times: np.ndarray[float]
        The time points at which the signal is sampled.
    :param signal: np.ndarray[float]
        The real signal to fit.
    :param n_modes: int
        The number of modes to fit. Each mode corresponds to a damped cosine term.
    :param with_post_fit: bool
        Whether to perform a post-fit refinement of the parameters, using least-square error minimization.
        This is recommended to improve the accuracy of the fit.
    :param tol: float
        The tolerance for the fitting process. It is used to determine the convergence of the fit.
    :param L_fraction: float
        The fraction of the signal length to use for the pencil method. Usually a value between 1/3 and 1/2 is
        recommended.
    :return:
    """
    # since the signal is real, there are two exponential per term
    # since 2cos(x) = exp(ix) + exp(-ix)

    offset, amplitudes, pulsations, decay_rates = bicfit(
        times,
        signal,
        n_modes=2 * n_modes,
        tol=tol,
        L_fraction=L_fraction,
    )

    amplitudes, phases, pulsations, decay_rates = _match_real_modes(amplitudes, pulsations, decay_rates, tol=tol)
    if abs(offset.imag) > tol:
        raise RuntimeError(
            f"Expected the offset to be real, but got {offset.imag} imaginary part, above fixed tolerance {tol}"
        )
    offset = offset.real

    if with_post_fit:
        result = _post_fit_damped_cosine(
            times, signal, offset, amplitudes, phases, pulsations, decay_rates
        )
    else:
        result = DampedCosineResult(
            times=times,
            signal=signal,
            offset=offset,
            amplitudes=amplitudes,
            phases=phases,
            pulsations=pulsations,
            decay_rates=decay_rates,
        )

    return result


def bicfit(
    times: np.ndarray[float],
    signal: np.ndarray[complex],
    n_modes: int = 1,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> Tuple[complex, np.ndarray[complex], np.ndarray[float], np.ndarray[float]]:
    """
    Fits a signal of the form s(t) = sum_k a_k exp(x_k t)
    using a pencil method. This method is exposed for advanced users
    who want to use it directly, but it is recommended to use the
    `fit_complex_exponential`, `fit_exponential_decay` or
    `fit_damped_cosine` functions instead.

    The algorithm is taken from
    **Generalized Pencil-of-Function Method for Extracting Poles
    of an EM System from Its Transient Response**
    from Hua and Sarkar (IEEE TRANSACTIONS ON ANTENNAS
    AND PROPAGATION, VOL. 37, NO. 2, FEBRUARY 1989)
    """

    if n_modes < 1:
        raise ValueError(f"Expected at least one mode to find, got {n_modes}")

    if times.shape != signal.shape or times.ndim != 1:
        raise ValueError(
            f"Expected times and signal of shape (n,) but got them of shape {times.shape} and {signal.shape}"
        )

    # preprocess by adding an artificial offset to make
    # the offset fit more stable
    offset = -signal.mean() + (1 + 1j) * 1e3 * signal.std()
    signal = signal + offset

    L = int(L_fraction * len(signal))
    N = len(signal)
    Y = np.zeros((N - L, L), dtype=np.complex128)

    times_diff = np.diff(times)
    if np.max(times_diff - times_diff[0]) > tol:
        raise ValueError("Non uniform sampling times are not supported.")

    # denoise the data using a SVD
    for i in range(L):
        Y[:, i] = signal[i : i + N - L]
    U, S, Vh = np.linalg.svd(Y, False)

    cutoff_idx = n_modes + 1  # set one mode for the constant term

    # filter all eigenvalues lower than the cutoff
    cutoff = np.sort(S)[-cutoff_idx]
    S[S < cutoff] = 0
    Y_filtered = U @ np.diag(S) @ Vh

    # retrieve the filtered signal
    Y1 = Y_filtered[:, :-1]
    Y2 = Y_filtered[:, 1:]

    # compute the eigenvalues of the pencil to find the modes
    Y1_inv = np.linalg.pinv(Y1)
    eigenvalues = np.linalg.eigvals(Y1_inv @ Y2)

    amplitudes, pulsations, decay_rates = _fit_amplitudes(eigenvalues, times, signal, cutoff_idx)
    constant_mode_idx = np.argmin(abs(np.exp(1j * pulsations - decay_rates) - 1))

    offset = amplitudes[constant_mode_idx] - offset
    amplitudes = np.delete(amplitudes, constant_mode_idx)
    pulsations = np.delete(pulsations, constant_mode_idx)
    decay_rates = np.delete(decay_rates, constant_mode_idx)

    return offset, amplitudes, pulsations, decay_rates


def _fit_amplitudes(
    eigenvalues: np.ndarray[complex],
    times: np.ndarray[float],
    signal: np.ndarray[complex],
    n_modes: int,
) -> Tuple[np.ndarray[complex], np.ndarray[complex], np.ndarray[complex]]:
    N = len(times)

    # Vandermonde Matrix
    V = np.zeros((N, n_modes), dtype=eigenvalues.dtype)
    eigenvalues = eigenvalues[np.argsort(abs(eigenvalues))][
        ::-1
    ]  # sort in reversed order
    eigenvalues = eigenvalues[:n_modes]
    acc = np.ones_like(eigenvalues)
    for i in range(N):
        V[i] = acc
        acc = acc * eigenvalues

    coefficients = np.linalg.lstsq(V, signal, rcond=None)[0]

    dt = np.diff(times)[0]
    pulsations = np.angle(eigenvalues) / dt
    decay_rates = -np.log(np.abs(eigenvalues)) / dt

    return coefficients, pulsations, decay_rates


def _match_real_modes(
    amplitudes: np.ndarray[complex],
    pulsations: np.ndarray[float],
    decay_rates: np.ndarray[float],
    tol: float,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    n = len(amplitudes)
    assert len(amplitudes) == len(pulsations) == len(decay_rates)
    assert n % 2 == 0, "Expected an even number of modes to match real modes"

    normalized_frequency = 1j * abs(pulsations) + decay_rates

    # step 1: separate frequencies that are in the upper and lower complex plane
    positive_w_indices, negative_w_indices = np.where(pulsations > 0)[0], np.where(pulsations < 0)[0]

    # step 2: get their normalized frequencies, i.e. mapped to the upper plane
    positive_w_normalized_frequencies = normalized_frequency[positive_w_indices]
    negative_w_normalized_frequencies = normalized_frequency[negative_w_indices]

    # step 3: create the cost matrix, that is the distance between the normalized frequencies
    positive_w_normalized_frequencies_mg, negative_w_normalized_frequencies_mg = (
        np.meshgrid(
            positive_w_normalized_frequencies, negative_w_normalized_frequencies
        )
    )
    cost = np.abs(
        positive_w_normalized_frequencies_mg - negative_w_normalized_frequencies_mg
    )

    # step 4: use the linear sum assignment to find the best matching pairs
    # and store the initial indices (ie of amplitudes, pulsations, decay_rates) in idx_1 and idx_2
    row_indices, col_indices = linear_sum_assignment(cost)
    idx_1, idx_2 = positive_w_indices[row_indices], negative_w_indices[col_indices]

    # step 5: check if the pairs are valid, i.e. they have close frequencies, decay rates and amplitudes
    if np.any(abs(pulsations[idx_1] - (-pulsations[idx_2])) > tol):
        raise RuntimeError(
            f"All real modes frequencies are expected to have close frequencies "
            f"(within {tol}) but got two paired modes with pulsations {pulsations[idx_1]} and {pulsations[idx_2]}"
        )

    if np.any(abs(decay_rates[idx_1] - decay_rates[idx_2]) > tol):
        raise RuntimeError(
            f"All real modes frequencies are expected to have close decay rates "
            f"(within {tol}) but got two paired modes with decay rates {decay_rates[idx_1]} and {decay_rates[idx_2]}"
        )

    if np.any(abs(amplitudes[idx_1] - np.conj(amplitudes[idx_2])) > tol):
        raise RuntimeError(
            f"All real modes frequencies are expected to have conjugate complex amplitudes "
            f"(within {tol}) but got two paired modes "
            f"with amplitudes {amplitudes[idx_1]} and {amplitudes[idx_2]}"
        )

    # step 6: create the real modes from the matched pairs
    real_pulsations = np.mean([pulsations[idx_1], -pulsations[idx_2]], axis=0).real
    real_decay_rates = np.mean([decay_rates[idx_1], decay_rates[idx_2]], axis=0).real
    real_amplitudes = np.abs([amplitudes[idx_1], amplitudes[idx_2]]).sum(axis=0).real
    real_phases = np.mean(
        [np.angle(amplitudes[idx_1]), -np.angle(amplitudes[idx_2])], axis=0
    ).real

    return real_amplitudes, real_phases, real_pulsations, real_decay_rates
