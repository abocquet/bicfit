import numpy as np
import pytest
from numpy.random import Generator, PCG64

from bicfit import fit_exponential_decay, ExponentialDecayResult, NoOffset

real_testdata = [
    (1.0, 0.05, 10.0, 150, 110),
    (-2.3, 0.1, 3.3, 120, 90),
]

complex_testdata = [
    (1.0 + 0.2j, 0.05, 10.0 + 5j, 150, 110),
    (2.3 - 1.4j, 0.1, 3.3 - 2j, 120, 90),
    (-4.3 - 1.0j, 0.15, 3.3 - 2j, 100, 90),
]


@pytest.mark.parametrize("amplitude, decay_rate, offset, horizon, n_points", real_testdata)
@pytest.mark.parametrize(
    "post_fit,noise,tol",
    [(False, 0, 0.02), (False, 0.05, 0.2), (True, 0, 0.02), (True, 0.05, 0.1)],
)
def test_single_real_exponential(
    amplitude, decay_rate, offset, horizon, n_points, post_fit, noise, tol
):
    rng = Generator(PCG64(42))
    times = np.linspace(0, horizon, n_points)
    noise_vec = rng.normal(0, noise, n_points)
    signal = offset + amplitude * np.exp(-decay_rate * times) + noise_vec

    result = fit_exponential_decay(times, signal, n_modes=1, post_fit=post_fit, is_complex=False)
    try:
        assert abs(result.modes[0].amplitude - amplitude) / abs(amplitude) < tol, (
            "amplitude"
        )
        assert abs(result.modes[0].decay_rate - decay_rate) / decay_rate < tol, "decay_rate"
        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed = e.args[0]
        print(
            f"Failed estimating '{failed}' for real exponential:\n"
            f"- amplitude true  = {amplitude} got={result.modes[0].amplitude}\n"
            f"- decay_rate true = {decay_rate} got={result.modes[0].decay_rate}\n"
            f"- offset true     = {offset} got={result.offset}\n"
            f"- horizon         = {horizon}\n"
            f"- n_points        = {n_points}"
        )
        raise e


@pytest.mark.parametrize(
    "amplitude, decay_rate, offset, horizon, n_points", complex_testdata
)
@pytest.mark.parametrize(
    "post_fit, noise,tol",
    [(False, 0, 0.03), (False, 0.05, 0.2), (True, 0, 0.03), (True, 0.05, 0.1)],
)
def test_single_complex_exponential(
    amplitude, decay_rate, offset, horizon, n_points, post_fit, noise, tol
):
    rng = Generator(PCG64(42))
    times = np.linspace(0, horizon, n_points)
    noise_vec = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)
    signal = offset + amplitude * np.exp(-decay_rate * times) + noise_vec

    result = fit_exponential_decay(times, signal, n_modes=1, post_fit=post_fit, is_complex=True)
    amp_est = result.modes[0].amplitude

    try:
        assert abs(abs(amp_est) - abs(amplitude)) / abs(amplitude) < tol, "amplitude"
        assert abs(result.modes[0].decay_rate - decay_rate) / decay_rate < tol, "decay_rate"
        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed = e.args[0]
        print(
            f"Failed estimating '{failed}' for complex exponential:\n"
            f"- amplitude true={amplitude} got={amp_est}\n"
            f"- decay_rate     true={decay_rate} got={result.modes[0].decay_rate}\n"
            f"- offset    true={offset} got={result.offset}\n"
            f"- horizon   = {horizon}\n"
            f"- n_points  = {n_points}"
        )
        raise e


@pytest.mark.parametrize(
    "post_fit,noise,tol",
    [(False, 0, 0.05), (False, 0.01, 0.6), (True, 0, 0.01), (True, 0.01, 0.1)],
)
def test_two_real_exponentials(post_fit, noise, tol):
    rng = Generator(PCG64(42))
    n_points = 120
    times = np.linspace(0, 150, n_points)
    noise_vec = rng.normal(0, noise, n_points)

    offset = 4.5
    a1, a2 = 0.6, 1.0
    decay_rate1, decay_rate2 = 0.01, 0.1

    signal = (
        offset + a1 * np.exp(-decay_rate1 * times) + a2 * np.exp(-decay_rate2 * times) + noise_vec
    )

    result = fit_exponential_decay(times, signal, n_modes=2, post_fit=post_fit, is_complex=False)

    try:
        assert abs(result.modes[0].amplitude - a1) / abs(a1) < tol, "a1"
        assert abs(result.modes[0].decay_rate - decay_rate1) / decay_rate1 < tol, "decay_rate1"

        assert abs(result.modes[1].amplitude - a2) / abs(a2) < tol, "a2"
        assert abs(result.modes[1].decay_rate - decay_rate2) / decay_rate2 < tol, "decay_rate2"

        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude 1  = {result.modes[0].amplitude}\n"
            f"- decay_rate 1 = {result.modes[0].decay_rate})\n"
            f"- amplitude 2  = {result.modes[1].amplitude}\n"
            f"- decay_rate 2 = {result.modes[1].decay_rate})\n"
            f"- offset       = {result.offset})\n"
        )
        raise e


@pytest.mark.parametrize(
    "post_fit,noise,tol",
    [(False, 0, 0.05), (False, 0.01, 0.3), (True, 0, 0.01), (True, 0.01, 0.05)],
)
def test_two_complex_exponentials(post_fit, noise, tol):
    rng = Generator(PCG64(42))
    n_points = 120
    times = np.linspace(0, 150, n_points)
    noise_vec = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)

    offset = 4.5 + 2.2j
    a1, a2 = 1.0 + 0.2j, 0.6 - 0.1j
    decay_rate1, decay_rate2 = 0.02, 0.1

    signal = (
        offset + a1 * np.exp(-decay_rate1 * times) + a2 * np.exp(-decay_rate2 * times) + noise_vec
    )

    result = fit_exponential_decay(times, signal, n_modes=2, post_fit=post_fit, is_complex=True)
    try:
        assert abs(result.modes[0].amplitude - a1) / abs(a1) < tol, "a1"
        assert abs(result.modes[0].decay_rate - decay_rate1) / decay_rate1 < tol, "decay_rate1"

        assert abs(result.modes[1].amplitude - a2) / abs(a2) < tol, "a2"
        assert abs(result.modes[1].decay_rate - decay_rate2) / decay_rate2 < tol, "decay_rate2"

        assert abs(result.offset - offset) / abs(offset) < tol, "offset"
    except AssertionError as e:
        failed_test = e.args[0].split("\n")[0]
        print(
            f"Failed estimating '{failed_test}' for \n"
            f"- amplitude 1  = {result.modes[0].amplitude}\n"
            f"- decay_rate 1 = {result.modes[0].decay_rate})\n"
            f"- amplitude 2  = {result.modes[1].amplitude}\n"
            f"- decay_rate 2 = {result.modes[1].decay_rate})\n"
            f"- offset       = {result.offset})\n"
        )
        raise e


def test_individual_mode():
    result = ExponentialDecayResult(
        0.0,
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0, 5.0]),
        np.array([6.0, 7.0, 8.0]),
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


@pytest.mark.parametrize(
    "amplitude, decay_rate, offset, horizon, n_points", complex_testdata
)
@pytest.mark.parametrize(
    "noise,tol",
    [(0, 0.03), (0.05, 0.1)]
)
def test_no_offset(
        amplitude, decay_rate, offset, horizon, n_points, noise, tol
):
    rng = Generator(PCG64(42))
    times = np.linspace(0, horizon, n_points)
    noise_vec = rng.normal(0, noise, n_points) + 1j * rng.normal(0, noise, n_points)
    signal = amplitude * np.exp(-decay_rate * times) + noise_vec

    result = fit_exponential_decay(times, signal, n_modes=1, post_fit=NoOffset(), is_complex=True)
    amp_est = result.modes[0].amplitude

    try:
        assert abs(abs(amp_est) - abs(amplitude)) / abs(amplitude) < tol, "amplitude"
        assert abs(result.modes[0].decay_rate - decay_rate) / decay_rate < tol, "decay_rate"
        assert abs(result.offset) < 1e-5, "offset"
    except AssertionError as e:
        failed = e.args[0]
        print(
            f"Failed estimating '{failed}' for complex exponential:\n"
            f"- amplitude true={amplitude} got={amp_est}\n"
            f"- decay_rate     true={decay_rate} got={result.modes[0].decay_rate}\n"
            f"- offset    true={offset} got={result.offset}\n"
            f"- horizon   = {horizon}\n"
            f"- n_points  = {n_points}"
        )
        raise e

