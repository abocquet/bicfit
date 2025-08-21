
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from bicfit import fit_damped_cosine

testdata = [
    (1.0, 0.2, 0.05, np.pi/2, 10.0, 150, 110),
    (2.3, 0.5, 0.1, 1.0, 3.3, 120, 90),
]


@pytest.mark.parametrize("amplitude, w, kappa, phase, offset, horizon, n_points", testdata)
@pytest.mark.parametrize("with_post_fit", [False, True])
@pytest.mark.parametrize("noise,tol", [(0, 0.08), (0.05, 0.2)])
def test_single_damped_cosine(amplitude, w, kappa, phase, offset, horizon, n_points, with_post_fit, noise, tol):
    rng = Generator(PCG64(42))
    times = np.linspace(0, horizon, n_points)
    noise_vec = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)
    signal = offset + amplitude * np.cos(phase + w * times) * np.exp(-kappa * times) + noise_vec.real

    result = fit_damped_cosine(times, signal, n_modes=1, with_post_fit=with_post_fit)
    try:
        assert abs(result.modes[0].amplitude - amplitude) / abs(amplitude) < tol, "amplitude"
        assert abs(result.modes[0].w - w) / w < tol, "w"
        assert abs(result.modes[0].kappa - kappa) / kappa < tol, "kappa"
        dphi = (result.modes[0].phase - phase + np.pi) % (2 * np.pi) - np.pi
        assert abs(dphi) < tol * np.pi, "phase"
        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude = {amplitude} (got {result.modes[0].amplitude})\n"
            f"- w        = {w} (got {result.modes[0].w})\n"
            f"- kappa    = {kappa} (got {result.modes[0].kappa})\n"
            f"- phase    = {phase} (got {result.modes[0].phase})\n"
            f"- offset   = {offset} (got {result.offset})\n"
            f"- horizon  = {horizon}\n"
            f"- n_points = {n_points}"
        )
        raise e

@pytest.mark.parametrize("with_post_fit", [False, True])
@pytest.mark.parametrize("noise,tol", [(0, 0.08), (0.05, 0.2)])
def test_two_damped_cosines(with_post_fit, noise, tol):
    rng = Generator(PCG64(42))
    n_points = 100
    times = np.linspace(0, 150, n_points)
    noise_vec = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)

    offset = 5.2
    a1, a2 = 1.0, 0.6
    w1, w2 = 0.2, 0.4
    kappa1, kappa2 = 0.05, 0.01
    phase1, phase2 = 1.0, -0.2

    signal = offset
    signal += a1 * np.cos(phase1 + w1 * times) * np.exp(-kappa1 * times)
    signal += a2 * np.cos(phase2 + w2 * times) * np.exp(-kappa2 * times)
    signal += noise_vec.real

    result = fit_damped_cosine(times, signal, n_modes=2, with_post_fit=with_post_fit)
    try:
        m1, m2 = result.modes
        if m1.w > m2.w:
            m1, m2 = m2, m1
        assert abs(m1.amplitude - a1) / abs(a1) < tol, "a1"
        assert abs(m1.w - w1) / w1 < tol, "w1"
        assert abs(m1.kappa - kappa1) / kappa1 < tol, "kappa1"
        dphi1 = (m1.phase - phase1 + np.pi) % (2 * np.pi) - np.pi
        assert abs(dphi1) < tol * np.pi, "phase1"

        assert abs(m2.amplitude - a2) / abs(a2) < tol, "a2"
        assert abs(m2.w - w2) / w2 < tol, "w2"
        assert abs(m2.kappa - kappa2) / kappa2 < tol, "kappa2"
        dphi2 = (m2.phase - phase2 + np.pi) % (2 * np.pi) - np.pi
        assert abs(dphi2) < tol * np.pi, "phase2"

        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- m1 = {m1}\n"
            f"- m2 = {m2}\n"
            f"- offset = {result.offset}"
        )
        raise e

