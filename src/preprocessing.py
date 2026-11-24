"""
EEG Signal Preprocessing Module.

Applies standard EEG preprocessing steps:
1. Bandpass filtering (0.5-50 Hz) to remove DC drift and high-frequency noise
2. Notch filtering (60 Hz) to remove powerline interference
3. Amplitude clipping for artifact rejection
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import logging

logger = logging.getLogger("EEG-Pipeline")


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply bandpass filter to multi-channel EEG data.
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        numpy array: Filtered data, same shape as input
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = np.zeros_like(data)
    
    for ch_idx in range(data.shape[0]):
        try:
            filtered[ch_idx] = filtfilt(b, a, data[ch_idx])
        except ValueError as e:
            logger.warning(f"Bandpass filter failed for channel {ch_idx}: {e}")
            filtered[ch_idx] = data[ch_idx]
    
    return filtered


def apply_notch_filter(data, notch_freq, fs, quality_factor=30.0):
    """
    Apply notch filter to remove powerline noise.
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        notch_freq: Frequency to notch out in Hz (e.g., 60)
        fs: Sampling frequency in Hz
        quality_factor: Quality factor of the notch filter
    
    Returns:
        numpy array: Filtered data
    """
    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered = np.zeros_like(data)
    
    for ch_idx in range(data.shape[0]):
        try:
            filtered[ch_idx] = filtfilt(b, a, data[ch_idx])
        except ValueError as e:
            logger.warning(f"Notch filter failed for channel {ch_idx}: {e}")
            filtered[ch_idx] = data[ch_idx]
    
    return filtered


def clip_amplitude(data, clip_std=10.0):
    """
    Clip extreme amplitude values (artifact rejection).
    Values beyond clip_std * std from the mean are clipped.
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        clip_std: Number of standard deviations for clipping threshold
    
    Returns:
        numpy array: Clipped data
    """
    clipped = data.copy()
    for ch_idx in range(data.shape[0]):
        ch_mean = np.mean(data[ch_idx])
        ch_std = np.std(data[ch_idx])
        if ch_std > 0:
            threshold = clip_std * ch_std
            clipped[ch_idx] = np.clip(
                data[ch_idx], ch_mean - threshold, ch_mean + threshold
            )
    return clipped


def preprocess_eeg(data, config):
    """
    Apply full preprocessing pipeline to EEG data.
    
    Steps:
    1. Bandpass filter (0.5-50 Hz)
    2. Notch filter (60 Hz)
    3. Amplitude clipping
    
    Args:
        data: numpy array of shape (n_channels, n_samples)
        config: Configuration dictionary
    
    Returns:
        numpy array: Preprocessed data
    """
    preproc = config["preprocessing"]
    fs = config["dataset"]["sampling_rate"]
    
    # Step 1: Bandpass filter
    logger.debug("Applying bandpass filter...")
    data = apply_bandpass_filter(
        data,
        lowcut=preproc["bandpass_low"],
        highcut=preproc["bandpass_high"],
        fs=fs,
        order=preproc["filter_order"],
    )
    
    # Step 2: Notch filter
    logger.debug("Applying notch filter...")
    data = apply_notch_filter(data, notch_freq=preproc["notch_freq"], fs=fs)
    
    # Step 3: Amplitude clipping
    logger.debug("Applying amplitude clipping...")
    data = clip_amplitude(data)
    
    return data
