from typing import TypeVar

import numpy as np

FloatLike = float | np.ndarray
F = TypeVar("F", bound=FloatLike)
