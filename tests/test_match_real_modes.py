import numpy as np
import pytest

from bicfit import _match_real_modes


def test_match_real_modes_happy_path():
    amplitudes = np.array([1 + 2j, 1 - 2j, 3 + 4j, 3 - 4j, 5 + 6j, 5 - 6j])
    ws = np.array([10.0, -10.0, 20.0, -20.0, 30.0, -30.0])
    kappas = np.array([5.0, 5.0, 7.0, 7.0, 9.0, 9.0])
    tol = 1e-9

    real_amplitudes, real_phases, real_ws, real_kappas = _match_real_modes(
        amplitudes, ws, kappas, tol
    )

    assert real_ws.shape == (3,)
    assert real_kappas.shape == (3,)
    assert real_phases.shape == (3,)

    assert np.allclose(real_ws, [10.0, 20.0, 30.0])
    assert np.allclose(real_kappas, [5.0, 7.0, 9.0])

    expected_phases = [np.angle(1 + 2j), np.angle(3 + 4j), np.angle(5 + 6j)]
    assert np.allclose(real_phases, expected_phases)

    expected_real_amplitude = np.array(
        [2 * abs(1 + 2j), 2 * abs(3 + 4j), 2 * abs(5 + 6j)]
    )
    assert np.allclose(real_amplitudes, expected_real_amplitude)


def test_match_real_modes_frequency_mismatch():
    amplitudes = np.array([1 + 2j, 3 + 4j, 1 - 2j, 3 - 4j])

    ws = np.array([10.0, 20.0, -10.01, -20.0])
    kappas = np.array([5.0, 7.0, 5.0, 7.0])
    tol = 1e-3
    with pytest.raises(RuntimeError):
        _match_real_modes(amplitudes, ws, kappas, tol)


def test_match_real_modes_kappa_mismatch():
    amplitudes = np.array([1 + 2j, 3 + 4j, 1 - 2j, 3 - 4j])
    ws = np.array([10.0, 20.0, -10.0, -20.0])
    kappas = np.array([5.0, 7.0, 5.002, 7.0])
    tol = 1e-3
    with pytest.raises(RuntimeError):
        _match_real_modes(amplitudes, ws, kappas, tol)


def test_match_real_modes_amplitude_mismatch():
    amplitudes = np.array([1 + 2j, 3 + 4j, 1 - 2.01j, 3 - 4j])
    ws = np.array([10.0, 20.0, -10.0, -20.0])
    kappas = np.array([5.0, 7.0, 5.0, 7.0])
    tol = 5e-3
    with pytest.raises(RuntimeError):
        _match_real_modes(amplitudes, ws, kappas, tol)
