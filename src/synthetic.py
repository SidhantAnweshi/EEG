"""
Synthetic Data Generator for CHB-MIT EEG Pipeline Testing.

Creates synthetic scalp EEG recordings and summary files matching the
exact CHB-MIT format and structure. This allows end-to-end testing,
pipeline validation, and demonstration without needing the 25GB download.
"""

import os
import numpy as np
import mne
import logging

logger = logging.getLogger("EEG-Pipeline")


def create_synthetic_eeg(duration_sec=300, sfreq=256, ch_names=None):
    """
    Create a synthetic multi-channel EEG signal with background rhythms and seizure events.
    
    Args:
        duration_sec: Duration in seconds
        sfreq: Sampling rate in Hz
        ch_names: List of channel names
    
    Returns:
        numpy array: Shape (n_channels, n_samples)
    """
    if ch_names is None:
        ch_names = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"
        ]
    
    n_channels = len(ch_names)
    n_samples = int(duration_sec * sfreq)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    data = np.zeros((n_channels, n_samples))
    
    # Generate background signal (alpha + beta + pink noise)
    for i in range(n_channels):
        # 10 Hz alpha wave + 20 Hz beta wave + noise
        alpha = 20 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
        beta = 10 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2 * np.pi))
        noise = 5 * np.random.randn(n_samples)
        data[i] = alpha + beta + noise
    
    # Inject synthetic seizure activity (high-amplitude 3-6 Hz spike-and-wave)
    # Seizure period: between sec 100 and 130 (30 seconds)
    seizure_start_sample = int(100 * sfreq)
    seizure_end_sample = int(130 * sfreq)
    t_seiz = t[seizure_start_sample:seizure_end_sample]
    
    for i in range(n_channels):
        # High amplitude rhythmic spike wave
        spike_wave = 80 * np.sin(2 * np.pi * 4 * t_seiz) + 40 * np.sin(2 * np.pi * 8 * t_seiz)
        seiz_noise = 10 * np.random.randn(len(t_seiz))
        data[i, seizure_start_sample:seizure_end_sample] += spike_wave + seiz_noise
    
    return data, ch_names, (100, 130)


def generate_synthetic_dataset(data_dir="data/raw", patients=None):
    """
    Generate synthetic CHB-MIT formatted dataset (EDF files + summary text files).
    
    Args:
        data_dir: Directory where raw files will be saved
        patients: List of patient IDs to create
    """
    if patients is None:
        patients = ["chb01", "chb02", "chb03", "chb05", "chb08", "chb10"]
    
    logger.info(f"Generating synthetic dataset in {data_dir} for patients: {patients}")
    os.makedirs(data_dir, exist_ok=True)
    
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ"
    ]
    
    sfreq = 256
    
    for p_idx, patient_id in enumerate(patients):
        p_dir = os.path.join(data_dir, patient_id)
        os.makedirs(p_dir, exist_ok=True)
        
        # Create 2 files per patient
        files_info = []
        for file_idx in [1, 2]:
            file_name = f"{patient_id}_{file_idx:02d}.edf"
            file_path = os.path.join(p_dir, file_name)
            
            # Duration: 200 seconds per file
            duration = 200
            n_samples = duration * sfreq
            t = np.linspace(0, duration, n_samples, endpoint=False)
            
            data = np.zeros((len(ch_names), n_samples))
            
            # Background signal
            for c in range(len(ch_names)):
                data[c] = (
                    15 * np.sin(2 * np.pi * (8 + c % 4) * t) +
                    5 * np.sin(2 * np.pi * 22 * t) +
                    4 * np.random.randn(n_samples)
                )
            
            # File 1 has a seizure, File 2 is normal (or both have seizures for some)
            has_seizure = (file_idx == 1) or (p_idx % 2 == 1)
            seizure_start, seizure_end = 50, 80
            
            if has_seizure:
                s_start_sample = seizure_start * sfreq
                s_end_sample = seizure_end * sfreq
                t_s = t[s_start_sample:s_end_sample]
                for c in range(len(ch_names)):
                    data[c, s_start_sample:s_end_sample] += (
                        70 * np.sin(2 * np.pi * 3.5 * t_s) +
                        35 * np.sin(2 * np.pi * 7 * t_s)
                    )
                files_info.append((file_name, [(seizure_start, seizure_end)]))
            else:
                files_info.append((file_name, []))
            
            # Save as EDF using MNE (MNE expects data in Volts)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
            mne.export.export_raw(file_path, raw, fmt="edf", overwrite=True, verbose=False)
            logger.info(f"  Created synthetic EDF: {file_path}")
        
        # Write summary file
        summary_path = os.path.join(p_dir, f"{patient_id}-summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Channels Count: {len(ch_names)}\n")
            for c_idx, c_name in enumerate(ch_names):
                f.write(f"Channel {c_idx+1}: {c_name}\n")
            f.write("\n")
            
            for f_name, seiz_list in files_info:
                f.write(f"File Name: {f_name}\n")
                f.write(f"Number of Seizures in File: {len(seiz_list)}\n")
                for s_idx, (s_start, s_end) in enumerate(seiz_list):
                    f.write(f"Seizure {s_idx+1} Start Time: {s_start} seconds\n")
                    f.write(f"Seizure {s_idx+1} End Time: {s_end} seconds\n")
                f.write("\n")
        logger.info(f"  Created synthetic summary file: {summary_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()
