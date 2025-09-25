from typing import List

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes._axes import Axes

from .results import ComplexResult, ExponentialDecayResult, DampedCosineResult
from .types import FloatLike


def plot(
    result: ComplexResult | DampedCosineResult | ExponentialDecayResult[FloatLike],
    axs: Axes | List[Axes] | None = None,
):
    if np.iscomplexobj(result.signal):
        if axs is not None:
            if not isinstance(axs, (list, np.ndarray)):
                raise TypeError(
                    "Expected axs to be a list or numpy array of matplotlib.axes.Axes or None"
                )
            elif len(axs) != 3:
                raise ValueError(
                    "Expected axs to contain exactly 3 Axes for complex plot"
                )

        _plot_complex(result, axs)  # ty: ignore[invalid-argument-type]
    else:
        if axs is not None and not isinstance(axs, Axes):
            raise TypeError("Expected axs to be a matplotlib.axes.Axes or None")
        _plot_real(result, axs)  # ty: ignore[invalid-argument-type]


def _plot_complex(
    result: ComplexResult | ExponentialDecayResult[complex], axs: List[Axes] | None
):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    if fig is not None:
        fig.suptitle("Fit result: " + result.pretty_repr())

    dense_times = np.linspace(result.times[0], result.times[-1], 1000)
    axs[0].plot(result.times, result.signal.real, label="Real Part")
    axs[0].plot(
        dense_times, result(dense_times).real, label="Fitted Real Part", linestyle="--"
    )
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Real Amplitude")
    axs[0].legend()

    axs[1].plot(result.times, result.signal.imag, label="Imaginary Part")
    axs[1].plot(
        dense_times,
        result(dense_times).imag,
        label="Fitted Imaginary Part",
        linestyle="--",
    )
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Imaginary Amplitude")
    axs[1].legend()

    axs[2].plot(result.signal.real, result.signal.imag, label="Signal")
    axs[2].plot(
        result(dense_times).real,
        result(dense_times).imag,
        label="Fitted Signal",
        linestyle="--",
    )
    axs[2].set_xlabel("Real Amplitude")
    axs[2].set_ylabel("Imaginary Amplitude")
    axs[2].legend()


def _plot_real(result: DampedCosineResult | ExponentialDecayResult[float], ax: Axes | None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    if fig is not None:
        fig.suptitle("Fit result: " + result.pretty_repr())

    dense_times = np.linspace(result.times[0], result.times[-1], 1000)
    ax.plot(result.times, result.signal, label="Signal")
    ax.plot(dense_times, result(dense_times), label="Fitted Signal", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal Amplitude")
    ax.legend()
