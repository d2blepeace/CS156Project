"""
windowing.py

Real-time sliding window management for phyphox sensor data.

This module provides a SlidingWindow class to:
- Buffer incoming sensor samples
- Generate overlapping windows for continuous predictions
- Support multiple windowing strategies (fixed stride, overlap, padding)
- Maintain proper data ordering and window boundaries
"""

from collections import deque
from typing import List, Optional, Tuple
import numpy as np


class SlidingWindow:
    """
    Manages sliding windows over a continuous stream of sensor samples.
    
    Parameters:
        window_size : int
            Number of samples per window (e.g., 50)
        step_size : int, optional
            Number of samples to advance between windows.
            If None, defaults to window_size (non-overlapping).
            If < window_size, creates overlapping windows.
            Default: None
        padding_strategy : str, optional
            - 'none': Skip until we have a full window
            - 'zero': Pad with zeros at the beginning
            - 'repeat': Repeat the first sample to fill the window
            Default: 'none'
        max_buffer_size : int, optional
            Maximum internal buffer size to prevent unbounded memory growth.
            Default: window_size * 10
    
    Attributes:
        buffer : deque
            Sliding buffer of [x, y, z] samples
        is_ready : bool
            Whether the current window is ready for prediction
    """
    
    def __init__(
        self,
        window_size: int,
        step_size: Optional[int] = None,
        padding_strategy: str = 'none',
        max_buffer_size: Optional[int] = None,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        self.padding_strategy = padding_strategy
        
        if self.step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.step_size > self.window_size:
            raise ValueError(
                f"step_size ({self.step_size}) cannot exceed "
                f"window_size ({self.window_size})"
            )
        
        # Max buffer size: prevent runaway memory usage
        if max_buffer_size is None:
            max_buffer_size = max(window_size * 10, 1000)
        self.max_buffer_size = max_buffer_size
        
        # Sliding buffer of [x, y, z] samples
        self.buffer: deque = deque(maxlen=max_buffer_size)
        
        # Track how many samples we've advanced past
        self.samples_processed = 0
        
        # Current window start position (relative to buffer[0])
        self.window_start_idx = 0
    
    def add_sample(self, sample: List[float]) -> None:
        """
        Add a new [x, y, z] sample to the buffer.
        
        Parameters:
            sample : List[float]
                A single [x, y, z] acceleration sample (3 values)
        
        Raises:
            ValueError
                If sample does not have exactly 3 values.
        """
        if len(sample) != 3:
            raise ValueError(
                f"Sample must have exactly 3 values [x, y, z], "
                f"got {len(sample)}: {sample}"
            )
        
        self.buffer.append(list(sample))
    
    @property
    def is_ready(self) -> bool:
        """
        Check if there is enough samples to make a prediction
        """
        available = len(self.buffer) - self.window_start_idx
        
        if self.padding_strategy == 'none':
            # Need at least window_size samples from the current window start
            return available >= self.window_size
        else:
            # With padding, we're ready as soon as we have at least 1 sample
            # (padding will fill the rest)
            return available >= 1
    
    def get_current_window(self) -> np.ndarray:
        """
        Extract the current window as a numpy array.
        
        Returns the next complete window of sensor data, applying
        padding if configured.
        
        Returns shape (window_size, 3) containing [x, y, z] samples.
            None if there is insufficient data and padding_strategy is 'none',
            
        np.ndarray
            Shape (window_size, 3) containing [x, y, z] samples.
            If insufficient data and padding='none', returns None.
            If padding='zero' or 'repeat', pads incomplete window.
        
        Raises:
            RuntimeError
                If is_ready is False and padding_strategy is 'none'.
        """
        if not self.is_ready and self.padding_strategy == 'none':
            raise RuntimeError(
                f"Window not ready. Buffer has {len(self.buffer)} samples, "
                f"need {self.window_size} from index {self.window_start_idx}."
            )
        
        start = self.window_start_idx
        end = start + self.window_size
        
        if len(self.buffer) >= end:
            # We have all the data we need
            window_data = list(self.buffer)[start:end]
            return np.array(window_data, dtype=float)
        else:
            # Incomplete window - apply padding strategy
            available_data = list(self.buffer)[start:]
            n_available = len(available_data)
            n_needed = self.window_size - n_available
            
            if self.padding_strategy == 'zero':
                # Pad with zeros at the beginning
                padding = [[0.0, 0.0, 0.0]] * n_needed
                padded_data = padding + available_data
            elif self.padding_strategy == 'repeat':
                # Repeat the first available sample
                first_sample = available_data[0] if available_data else [0.0, 0.0, 0.0]
                padding = [first_sample] * n_needed
                padded_data = padding + available_data
            else:
                raise ValueError(f"Unknown padding_strategy: {self.padding_strategy}")
            
            return np.array(padded_data, dtype=float)
    
    def advance(self) -> None:
        """
        Advance the window by step_size samples.
        
        Call this after making a prediction to slide the window forward
        for the next prediction.
        """
        self.window_start_idx += self.step_size
        
        # Don't let window_start_idx exceed buffer size
        if self.window_start_idx >= len(self.buffer):
            self.window_start_idx = len(self.buffer)
    
    def reset(self) -> None:
        """
        Clear the buffer and reset state.
        
        Use this to start fresh (e.g., when starting a new activity sequence).
        """
        self.buffer.clear()
        self.samples_processed = 0
        self.window_start_idx = 0
    
    @property
    def buffer_size(self) -> int:
        """Current number of samples in the buffer."""
        return len(self.buffer)
    
    @property
    def samples_available_for_window(self) -> int:
        """
        Number of samples available for the current window.
        
        Returns the count of samples from window_start_idx to the end of buffer.
        """
        return max(0, len(self.buffer) - self.window_start_idx)
    
    def __repr__(self) -> str:
        return (
            f"SlidingWindow(window_size={self.window_size}, "
            f"step_size={self.step_size}, "
            f"padding={self.padding_strategy}, "
            f"buffer={self.buffer_size}/{self.max_buffer_size} "
            f"samples, ready={self.is_ready})"
        )

def create_window_from_static_data(
    data: List[List[float]],
    window_size: int,
) -> np.ndarray:
    """
    Utility to create a single window from static sensor data.
    
    Useful for batch/offline processing or testing.
    
    Parameters:
        data : List[List[float]]
            List of [x, y, z] samples
        window_size : int
            Number of samples per window
    
    Returns:
        np.ndarray
            Shape (window_size, 3) window, or None if insufficient data.
    """
    if len(data) < window_size:
        return None
    
    return np.array(data[:window_size], dtype=float)
