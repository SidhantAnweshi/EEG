"""
Segmentation Module.

Segments continuous EEG recordings into fixed-length windows and assigns
labels (seizure=1, non-seizure=0) based on overlap with seizure annotations.

A window is labeled as "seizure" if at least `seizure_overlap_threshold`
fraction of the window overlaps with an annotated seizure period.
"""

import numpy as np
import logging

logger = logging.getLogger("EEG-Pipeline")


def segment_signal(data, fs, window_seconds, overlap_fraction):
    """
    Segment continuous EEG data into fixed-length overlapping windows.
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        fs: Sampling frequency in Hz
        window_seconds: Window length in seconds
        overlap_fraction: Fraction of overlap between consecutive windows
    
    Returns:
        list of numpy arrays: Each of shape (n_channels, window_samples)
        list of tuples: (start_sample, end_sample) for each window
    """
    window_samples = int(window_seconds * fs)
    step_samples = int(window_samples * (1 - overlap_fraction))
    n_samples = data.shape[1]
    
    windows = []
    window_positions = []
    
    start = 0
    while start + window_samples <= n_samples:
        window = data[:, start : start + window_samples]
        windows.append(window)
        window_positions.append((start, start + window_samples))
        start += step_samples
    
    return windows, window_positions


def label_windows(window_positions, fs, seizure_intervals, overlap_threshold=0.5):
    """
    Label each window as seizure (1) or non-seizure (0).
    
    A window is labeled "seizure" if the fraction of the window that
    overlaps with any seizure interval exceeds the threshold.
    
    Args:
        window_positions: List of (start_sample, end_sample) tuples
        fs: Sampling frequency in Hz
        seizure_intervals: List of (start_sec, end_sec) tuples from annotations
        overlap_threshold: Minimum overlap fraction to label as seizure
    
    Returns:
        numpy array: Binary labels (0 or 1) for each window
    """
    labels = np.zeros(len(window_positions), dtype=np.int32)
    
    if not seizure_intervals:
        return labels
    
    # Convert seizure intervals to samples
    seizure_samples = [
        (int(start * fs), int(end * fs)) for start, end in seizure_intervals
    ]
    
    for i, (win_start, win_end) in enumerate(window_positions):
        window_length = win_end - win_start
        overlap_total = 0
        
        for seiz_start, seiz_end in seizure_samples:
            # Calculate overlap between window and seizure
            overlap_start = max(win_start, seiz_start)
            overlap_end = min(win_end, seiz_end)
            
            if overlap_start < overlap_end:
                overlap_total += overlap_end - overlap_start
        
        # Label as seizure if overlap exceeds threshold
        if window_length > 0 and (overlap_total / window_length) >= overlap_threshold:
            labels[i] = 1
    
    return labels


def segment_and_label_file(data, fs, seizure_intervals, config):
    """
    Segment an entire EEG file and assign labels.
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        fs: Sampling frequency
        seizure_intervals: List of (start_sec, end_sec) for this file
        config: Configuration dictionary
    
    Returns:
        windows: List of numpy arrays
        labels: numpy array of binary labels
        positions: List of (start_sample, end_sample) tuples
    """
    seg_config = config["segmentation"]
    
    windows, positions = segment_signal(
        data,
        fs,
        window_seconds=seg_config["window_seconds"],
        overlap_fraction=seg_config["overlap_fraction"],
    )
    
    labels = label_windows(
        positions,
        fs,
        seizure_intervals,
        overlap_threshold=seg_config["seizure_overlap_threshold"],
    )
    
    return windows, labels, positions
