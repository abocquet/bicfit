from numpy.random import Generator, PCG64
import numpy as np
import pytest

from bicfit import fit_complex_exponential, ComplexResult, NoOffset

testdata = [
    (1.0, 13.7, 0.92, 61 + 62j, 3.6, 110),
    (21, 2, 0.01, 1, 50, 100),
]


@pytest.mark.parametrize("amplitude, pulsation, decay_rate, offset, horizon, n_points", testdata)
@pytest.mark.parametrize("post_fit", [False, True])
@pytest.mark.parametrize("noise,tol", [(0, 0.01), (0.1, 0.15)])
def test_single_exponential(
    amplitude, decay_rate, pulsation, offset, horizon, n_points, post_fit, noise, tol
):
    rng = Generator(PCG64(42))
    times = np.linspace(0, horizon, n_points)
    noise = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)
    signal = offset + amplitude * np.exp((1j * pulsation - decay_rate) * times) + noise

    # fit the signal
    result = fit_complex_exponential(
        times, signal, n_modes=1, post_fit=post_fit
    )
    try:
        assert abs(result.modes[0].amplitude - amplitude) / abs(amplitude) < tol, (
            "amplitude"
        )
        assert abs(result.modes[0].pulsation - pulsation) / pulsation < tol, "pulsation"
        assert abs(result.modes[0].decay_rate - decay_rate) / decay_rate < tol, "decay_rate"
        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude  = {amplitude} (got {result.modes[0].amplitude})\n"
            f"- pulsation  = {pulsation} (got {result.modes[0].pulsation})\n"
            f"- decay_rate = {decay_rate} (got {result.modes[0].decay_rate})\n"
            f"- offset     = {offset} (got {result.offset})\n"
            f"- horizon    = {horizon}\n"
            f"- n_points   = {n_points}"
        )
        raise e


@pytest.mark.parametrize(
    "post_fit,noise,tol",
    [(False, 0, 0.01), (False, 0.1, 0.25), (True, 0, 0.01), (True, 0.1, 0.2)],
)
def test_two_exponential(post_fit, noise, tol):
    rng = Generator(PCG64(42))
    n_points = 100
    times = np.linspace(0, 150, n_points)
    noise = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)

    offset = 10.2 + 20.3j
    a1, a2 = 0.5, 1.0
    pulsation1, pulsation2 = 0.4, 0.2
    decay_rate1, decay_rate2 = 0.02, 0.05

    signal = offset
    signal += a1 * np.exp((1j * pulsation1 - decay_rate1) * times)
    signal += a2 * np.exp((1j * pulsation2 - decay_rate2) * times)
    signal += noise

    # fit the signal
    result = fit_complex_exponential(
        times, signal, n_modes=2, post_fit=post_fit
    )

    try:
        assert abs(result.modes[0].amplitude - a1) / a1 < tol, "a1"
        assert abs(result.modes[0].pulsation - pulsation1) / pulsation1 < tol, "pulsation1"
        assert abs(result.modes[0].decay_rate - decay_rate1) / decay_rate1 < tol, "decay_rate1"

        assert abs(result.modes[1].amplitude - a2) / a2 < tol, "a2"
        assert abs(result.modes[1].pulsation - pulsation2) / pulsation2 < tol, "pulsation2"
        assert abs(result.modes[1].decay_rate - decay_rate2) / decay_rate2 < tol, "decay_rate2"

        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude 1   = {result.modes[0].amplitude}\n"
            f"- pulsation 1   = {result.modes[0].pulsation}\n"
            f"- decay_rate 1  = {result.modes[0].decay_rate})\n"
            f"- amplitude 2   = {result.modes[1].amplitude}\n"
            f"- pulsation 2   = {result.modes[1].pulsation}\n"
            f"- decay_rate 2  = {result.modes[1].decay_rate})\n"
            f"- offset        = {result.offset})\n"
        )
        raise e


def test_individual_mode():
    result = ComplexResult(
        0.0,
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
        np.array([10.0, 11.0, 12.0]),
    )

    t = np.linspace(0, 10, 11)
    assert result(t).shape == (11,)
    assert np.allclose(
        result(t), result.modes[0](t) + result.modes[1](t) + result.modes[2](t)
    )
    assert result(0.0).shape == tuple()
    assert np.isclose(
        result(0.0), result.modes[0](0.0) + result.modes[1](0.0) + result.modes[2](0.0)
    )

@pytest.mark.parametrize("amplitude, pulsation, decay_rate, _offset, horizon, n_points", testdata)
def test_no_offset(
        amplitude, decay_rate, pulsation, _offset, horizon, n_points
):
    rng = Generator(PCG64(42))
    noise,tol = 0.1, 0.15
    times = np.linspace(0, horizon, n_points)
    noise = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)
    signal = amplitude * np.exp((1j * pulsation - decay_rate) * times) + noise

    # fit the signal
    result = fit_complex_exponential(
        times, signal, n_modes=1, post_fit=NoOffset()
    )
    try:
        assert abs(result.modes[0].amplitude - amplitude) / abs(amplitude) < tol, (
            "amplitude"
        )
        assert abs(result.modes[0].pulsation - pulsation) / pulsation < tol, "pulsation"
        assert abs(result.modes[0].decay_rate - decay_rate) / decay_rate < tol, "decay_rate"
        assert abs(result.offset) < 1e-5, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude  = {amplitude} (got {result.modes[0].amplitude})\n"
            f"- pulsation  = {pulsation} (got {result.modes[0].pulsation})\n"
            f"- decay_rate = {decay_rate} (got {result.modes[0].decay_rate})\n"
            f"- offset     = 0.0 (got {result.offset})\n"
            f"- horizon    = {horizon}\n"
            f"- n_points   = {n_points}"
        )
        raise e

