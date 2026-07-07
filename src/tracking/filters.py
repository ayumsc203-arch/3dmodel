import numpy as np
import time
from typing import Optional, Union


class LowPassFilter:
    """Standard low-pass filter."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y: Optional[Union[float, np.ndarray]] = None

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


class OneEuroFilter:
    """
    Adaptive low-pass filter (One Euro Filter) designed to filter jittery signals.
    Adapts cutoff frequency based on input signal speed.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.004,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter(0.0)
        self.dx_filt = LowPassFilter(0.0)
        self.last_time: Optional[float] = None

    def _get_alpha(self, cutoff: float, dt: float) -> float:
        """Helper to calculate alpha from cutoff frequency and delta time."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(
        self,
        x: Union[float, np.ndarray],
        timestamp: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """Filters the input signal x. Supports both scalars and numpy arrays."""
        # Calculate time step
        current_time = timestamp if timestamp is not None else time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.x_filt.y = x
            self.dx_filt.y = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            return x

        dt = current_time - self.last_time
        # Prevent division by zero or negative time
        if dt <= 0.0:
            return self.x_filt.y

        self.last_time = current_time

        # Calculate rate of change (derivative/velocity)
        prev_x = self.x_filt.y
        dx = (x - prev_x) / dt

        # Filter rate of change
        alpha_d = self._get_alpha(self.d_cutoff, dt)
        self.dx_filt.alpha = alpha_d
        edx = self.dx_filt(dx)

        # Calculate cutoff frequency based on velocity magnitude
        speed = np.abs(edx)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter signal
        alpha = self._get_alpha(cutoff, dt)
        self.x_filt.alpha = alpha
        return self.x_filt(x)

    def reset(self) -> None:
        """Resets the filter state."""
        self.x_filt.y = None
        self.dx_filt.y = None
        self.last_time = None
