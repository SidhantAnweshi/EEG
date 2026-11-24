"""
EEG Feature Extraction Module.

Extracts interpretable time-domain and frequency-domain features from
fixed-length EEG windows for seizure detection:

Time-domain features:
    - Mean, Variance, Std, RMS
    - Skewness, Kurtosis
    - Line Length (sum of absolute first differences)
    - Zero Crossings
    - Hjorth Parameters (Activity, Mobility, Complexity)

Frequency-domain features:
    - Band Powers (Delta, Theta, Alpha, Beta, Gamma)
    - Relative Band Powers (normalized by total power)
    - Spectral Entropy
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch
import logging

logger = logging.getLogger("EEG-Pipeline")


def compute_band_power(data, fs, band, nperseg=None):
    """
    Compute the average power in a specific frequency band using Welch's method.
    
    Args:
        data: 1D numpy array (single channel, single window)
        fs: Sampling frequency
        band: Tuple (low_freq, high_freq)
        nperseg: Segment length for Welch's method
    
    Returns:
        float: Average power in the band
    """
    if nperseg is None:
        nperseg = min(len(data), int(fs * 2))  # 2-second segments or data length
    
    nperseg = min(nperseg, len(data))
    
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    
    # Find frequency indices within the band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    
    if not np.any(band_mask):
        return 0.0
    
    # Integrate PSD over the band (trapezoidal integration)
    try:
        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
    except AttributeError:
        from scipy.integrate import trapezoid
        band_power = trapezoid(psd[band_mask], freqs[band_mask])
    
    return band_power


def compute_spectral_entropy(data, fs, nperseg=None):
    """
    Compute spectral entropy of the signal.
    
    Spectral entropy measures the "uniformity" of the power spectrum.
    Low entropy = power concentrated in few frequencies (rhythmic)
    High entropy = power spread across frequencies (noisy/complex)
    
    Args:
        data: 1D numpy array
        fs: Sampling frequency
        nperseg: Segment length for Welch's method
    
    Returns:
        float: Normalized spectral entropy [0, 1]
    """
    if nperseg is None:
        nperseg = min(len(data), int(fs * 2))
    
    nperseg = min(nperseg, len(data))
    
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    
    # Normalize PSD to form a probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-12)
    
    # Compute entropy
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(psd))
    if max_entropy > 0:
        entropy /= max_entropy
    
    return entropy


def compute_hjorth_parameters(data):
    """
    Compute Hjorth parameters: Activity, Mobility, Complexity.
    
    - Activity: Variance of the signal (signal power)
    - Mobility: Std of first derivative / Std of signal
      (proportion of standard deviation of the power spectrum)
    - Complexity: Mobility of first derivative / Mobility of signal
      (change in frequency)
    
    Args:
        data: 1D numpy array
    
    Returns:
        tuple: (activity, mobility, complexity)
    """
    # First derivative
    d1 = np.diff(data)
    # Second derivative
    d2 = np.diff(d1)
    
    # Activity = variance of signal
    activity = np.var(data)
    
    # Mobility = sqrt(var(d1) / var(signal))
    var_d1 = np.var(d1)
    if activity > 0:
        mobility = np.sqrt(var_d1 / activity)
    else:
        mobility = 0.0
    
    # Complexity = mobility(d1) / mobility(signal)
    var_d2 = np.var(d2)
    if var_d1 > 0:
        mobility_d1 = np.sqrt(var_d2 / var_d1)
    else:
        mobility_d1 = 0.0
    
    if mobility > 0:
        complexity = mobility_d1 / mobility
    else:
        complexity = 0.0
    
    return activity, mobility, complexity


def extract_window_features(window_data, fs, freq_bands):
    """
    Extract all features from a single window of multi-channel EEG data.
    
    Args:
        window_data: numpy array of shape (n_channels, n_samples)
        fs: Sampling frequency
        freq_bands: dict of {band_name: (low, high)} frequency bands
    
    Returns:
        numpy array: 1D feature vector
    """
    n_channels = window_data.shape[0]
    features = []
    
    for ch_idx in range(n_channels):
        ch_data = window_data[ch_idx]
        
        # ---- Time-domain features ----
        
        # Mean
        features.append(np.mean(ch_data))
        
        # Variance
        features.append(np.var(ch_data))
        
        # Standard deviation
        features.append(np.std(ch_data))
        
        # RMS (Root Mean Square)
        features.append(np.sqrt(np.mean(ch_data ** 2)))
        
        # Skewness
        features.append(float(sp_stats.skew(ch_data)))
        
        # Kurtosis
        features.append(float(sp_stats.kurtosis(ch_data)))
        
        # Line Length (sum of absolute differences)
        features.append(np.sum(np.abs(np.diff(ch_data))))
        
        # Zero Crossings
        zero_crossings = np.sum(np.abs(np.diff(np.sign(ch_data))) > 0)
        features.append(zero_crossings)
        
        # ---- Frequency-domain features ----
        
        # Band powers
        band_powers = {}
        total_power = 0
        for band_name, band_range in freq_bands.items():
            bp = compute_band_power(ch_data, fs, band_range)
            band_powers[band_name] = bp
            total_power += bp
            features.append(bp)
        
        # Relative band powers
        for band_name in freq_bands:
            if total_power > 0:
                features.append(band_powers[band_name] / total_power)
            else:
                features.append(0.0)
        
        # Spectral entropy
        features.append(compute_spectral_entropy(ch_data, fs))
        
        # ---- Hjorth parameters ----
        activity, mobility, complexity = compute_hjorth_parameters(ch_data)
        features.append(activity)
        features.append(mobility)
        features.append(complexity)
    
    return np.array(features, dtype=np.float64)


def get_feature_names(channel_names, freq_bands):
    """
    Generate human-readable feature names.
    
    Args:
        channel_names: List of channel names
        freq_bands: dict of frequency band names
    
    Returns:
        list: Feature name strings
    """
    feature_names = []
    
    time_features = [
        "mean", "variance", "std", "rms",
        "skewness", "kurtosis", "line_length", "zero_crossings"
    ]
    
    for ch in channel_names:
        # Time-domain
        for feat in time_features:
            feature_names.append(f"{ch}_{feat}")
        
        # Band powers
        for band_name in freq_bands:
            feature_names.append(f"{ch}_bandpower_{band_name}")
        
        # Relative band powers
        for band_name in freq_bands:
            feature_names.append(f"{ch}_rel_bandpower_{band_name}")
        
        # Spectral entropy
        feature_names.append(f"{ch}_spectral_entropy")
        
        # Hjorth
        feature_names.append(f"{ch}_hjorth_activity")
        feature_names.append(f"{ch}_hjorth_mobility")
        feature_names.append(f"{ch}_hjorth_complexity")
    
    return feature_names
