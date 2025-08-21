from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .post_fit import (
    _post_fit_complex_exponential,
    _post_fit_damped_cosine,
    _post_fit_exponential,
)
from .types import Result, ComplexMode, RealMode, Exponential


def fit_complex_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    n_modes: int = 1,
    with_post_fit: bool = True,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> Result[ComplexMode]:
    result = bicfit(
        times, signal, n_modes=n_modes, is_complex=True, tol=tol, L_fraction=L_fraction
    )
    if with_post_fit:
        result = _post_fit_complex_exponential(times, signal, result)

    return result


def fit_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    n_modes: int = 1,
    with_post_fit: bool = True,
    is_complex: bool = False,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> Result[Exponential]:
    result = bicfit(
        times,
        signal,
        n_modes=n_modes,
        is_complex=True,
        tol=tol,
        L_fraction=L_fraction,
    )
    if with_post_fit:
        result = _post_fit_exponential(times, signal, result, is_complex)
    else:
        result.modes = [Exponential(mode.amplitude, mode.kappa) for mode in result.modes]

    return result


def fit_damped_cosine(
    times: np.ndarray,
    signal: np.ndarray,
    n_modes: int = 1,
    with_post_fit: bool = True,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
) -> Result[RealMode]:
    result = bicfit(
        times, signal, n_modes=n_modes, is_complex=False, tol=tol, L_fraction=L_fraction
    )
    if with_post_fit:
        result = _post_fit_damped_cosine(times, signal, result)


    return result


def bicfit(
    times: np.ndarray,
    signal: np.ndarray,
    n_modes: int = 1,
    is_complex: bool | None = None,
    tol: float = 1e-3,
    L_fraction: float = 0.3,
):
    """
    Fits a signal of the form s(t) = sum_k a_k exp(x_k t)
    using a pencil method.

    The algorithm is taken from

    **Generalized Pencil-of-Function Method for Extracting Poles
    of an EM System from Its Transient Response**

    from Hua and Sarkar (IEEE TRANSACTIONS ON ANTENNAS
    AND PROPAGATION, VOL. 37, NO. 2, FEBRUARY 1989)
    """

    if n_modes < 1:
        raise ValueError(f"Expected at least one mode to find, got {n_modes}")

    if times.shape != signal.shape or len(times.shape) != 1:
        raise ValueError(
            f"Expected times and signal of shape (n,) but got them of shape {times.shape} and {signal.shape}"
        )

    if is_complex is None:
        is_complex = np.iscomplexobj(signal)

    # preprocess by adding an artificial offset to make
    # the offset fit more stable
    original_signal = np.copy(signal)
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

    cutoff_idx = 1  # set one mode for the constant term
    if is_complex:
        cutoff_idx += n_modes
    else:
        # is the signal is real, there are
        # two exponential per term
        # since 2cos(x) = exp(ix) + exp(-ix)
        cutoff_idx += 2 * n_modes

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

    modes = _fit_amplitudes(eigenvalues, times, signal, cutoff_idx)
    constant_mode_idx = np.argmin(
        [abs(np.exp(mode.kappa + 1j * mode.w) - 1) for mode in modes]
    )
    constant_mode = modes[constant_mode_idx]
    offset = constant_mode.complex_amplitude - offset
    modes = modes[:constant_mode_idx] + modes[constant_mode_idx + 1 :]

    if not is_complex:
        modes = _match_real_modes(modes, tol=tol)
        if abs(offset.imag) > tol:
            raise RuntimeError(
                f"Expected the offset to be real, but got {offset.imag} imaginary part, above fixed tolerance {tol}"
            )
        offset = offset.real

    return Result(offset=offset, modes=modes, times=times, signal=original_signal)


def _fit_amplitudes(
    eigenvalues: np.ndarray, times: np.ndarray, signal: np.ndarray, n_modes: int
) -> List[ComplexMode]:
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

    modes = []
    dt = np.diff(times)[0]
    for eigenvalue, coefficient in zip(eigenvalues, coefficients):
        w = np.angle(eigenvalue) / dt
        kappa = -np.log(np.abs(eigenvalue)) / dt
        modes.append(
            ComplexMode(
                complex_amplitude=coefficient,
                w=w,
                kappa=kappa,
            )
        )

    return modes


def _match_real_modes(modes: List[ComplexMode], tol: float) -> List[RealMode]:  # noqa: F821
    # todo: the matching should be made only between the modes with negative frequencies to
    # the modes with positive frequencies to avoid love triangles
    n = len(modes)
    assert n % 2 == 0, "Expected an even number of modes to match real modes"
    normalized_frequency = [1j * abs(mode.w) + mode.kappa for mode in modes]
    cost = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                cost[i, j] = np.inf
            else:
                cost[i, j] = abs(normalized_frequency[i] - normalized_frequency[j])

    print(cost)
    row_indices, col_indices = linear_sum_assignment(cost)
    print(row_indices, col_indices)

    assignment = dict()
    for idx_1, idx_2 in zip(row_indices, col_indices):
        mode_1, mode_2 = modes[idx_1], modes[idx_2]
        if mode_1.w < 0:
            mode_2, mode_1 = mode_1, mode_2
        assignment[mode_1] = mode_2

    real_modes = []
    for mode_1, mode_2 in assignment.items():
        if abs(mode_1.w - (-mode_2.w)) > tol:
            raise RuntimeError(
                f"All real modes frequencies are expected to have close frequencies "
                f"(within {tol}) but got two paired modes with pulsations {mode_1.w} and {mode_2.w}"
            )

        if abs(mode_1.kappa - mode_2.kappa) > tol:
            raise RuntimeError(
                f"All real modes frequencies are expected to have close decay rates "
                f"(within {tol}) but got two paired modes with decay rates {mode_1.kappa} and {mode_2.kappa}"
            )

        if abs(mode_1.complex_amplitude - np.conj(mode_2.complex_amplitude)) > tol:
            raise RuntimeError(
                f"All real modes frequencies are expected to have conjugate complex amplitudes "
                f"(within {tol}) but got two paired modes "
                f"with amplitudes {mode_1.complex_amplitude} and {mode_2.complex_amplitude}"
            )

        w = np.mean([mode_1.w, -mode_2.w]).real
        kappa = np.mean([mode_1.kappa, mode_2.kappa]).real
        amplitude = (
            np.abs([mode_1.complex_amplitude, mode_2.complex_amplitude]).sum().real
        )
        phase = np.mean(
            [np.angle(mode_1.complex_amplitude), -np.angle(mode_2.complex_amplitude)]
        ).real

        real_modes.append(RealMode(w=w, kappa=kappa, amplitude=amplitude, phase=phase))

    return real_modes
