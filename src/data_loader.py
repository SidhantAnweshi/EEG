"""
Data loading and annotation parsing for the CHB-MIT Scalp EEG dataset.

The CHB-MIT dataset consists of:
- EDF files organized per patient (chb01/, chb02/, etc.)
- Summary text files (chbXX-summary.txt) with seizure annotations
- 256 Hz sampling rate, 23 EEG channels (International 10-20 system)
"""

import os
import re
import mne
import numpy as np
import logging

logger = logging.getLogger("EEG-Pipeline")


def parse_summary_file(summary_path):
    """
    Parse the CHB-MIT summary file to extract seizure annotations.
    
    Each patient directory has a summary file (e.g., chb01-summary.txt) with:
    - File names
    - Number of seizures in each file
    - Seizure start/end times in seconds
    
    Returns:
        dict: {filename: [(start_sec, end_sec), ...]} for files with seizures
    """
    seizure_annotations = {}
    
    if not os.path.exists(summary_path):
        logger.warning(f"Summary file not found: {summary_path}")
        return seizure_annotations
    
    with open(summary_path, "r") as f:
        content = f.read()
    
    # Split by "File Name:" to process each file block
    file_blocks = re.split(r"File Name:", content)
    
    for block in file_blocks[1:]:  # Skip the header before first file
        lines = block.strip().split("\n")
        filename = lines[0].strip()
        
        # Find number of seizures
        n_seizures = 0
        for line in lines:
            match = re.search(r"Number of Seizures in File:\s*(\d+)", line)
            if match:
                n_seizures = int(match.group(1))
                break
        
        if n_seizures > 0:
            seizures = []
            for line in lines:
                # Match seizure start time
                start_match = re.search(
                    r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", line
                )
                end_match = re.search(
                    r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", line
                )
                if start_match:
                    start_sec = int(start_match.group(1))
                if end_match:
                    end_sec = int(end_match.group(1))
                    seizures.append((start_sec, end_sec))
            
            if seizures:
                seizure_annotations[filename] = seizures
                logger.info(
                    f"  {filename}: {len(seizures)} seizure(s) - "
                    f"{[(s, e) for s, e in seizures]}"
                )
    
    return seizure_annotations


def discover_patients(data_dir):
    """
    Discover all patient directories in the dataset.
    
    Returns:
        list: Sorted list of patient IDs (e.g., ['chb01', 'chb02', ...])
    """
    patients = []
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return patients
    
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith("chb"):
            patients.append(item)
    
    logger.info(f"Discovered {len(patients)} patient directories: {patients}")
    return patients


def get_patient_files(data_dir, patient_id):
    """
    Get all EDF files for a specific patient.
    
    Returns:
        list: Sorted list of EDF file paths
    """
    patient_dir = os.path.join(data_dir, patient_id)
    edf_files = []
    
    if not os.path.exists(patient_dir):
        logger.warning(f"Patient directory not found: {patient_dir}")
        return edf_files
    
    for f in sorted(os.listdir(patient_dir)):
        if f.lower().endswith(".edf"):
            edf_files.append(os.path.join(patient_dir, f))
    
    return edf_files


def get_summary_path(data_dir, patient_id):
    """Get the path to the summary file for a patient."""
    patient_dir = os.path.join(data_dir, patient_id)
    
    # Try different naming patterns
    patterns = [
        f"{patient_id}-summary.txt",
        f"{patient_id.replace('chb', 'chb')}-summary.txt",
    ]
    
    for pattern in patterns:
        path = os.path.join(patient_dir, pattern)
        if os.path.exists(path):
            return path
    
    # Fallback: look for any summary file
    for f in os.listdir(patient_dir):
        if "summary" in f.lower() and f.endswith(".txt"):
            return os.path.join(patient_dir, f)
    
    return None


def load_edf_file(filepath, common_channels=None, verbose=False):
    """
    Load an EDF file using MNE and return the raw data.
    
    Args:
        filepath: Path to the EDF file
        common_channels: List of channel names to select (for consistency)
        verbose: Whether to show MNE output
    
    Returns:
        tuple: (data_array, channel_names, sfreq) or (None, None, None) on error
    """
    try:
        # Load with MNE, suppress output
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=verbose)
        
        # Get channel names from the file
        available_channels = raw.ch_names
        
        if common_channels is not None:
            # Find channels that are available in this file
            channels_to_use = [
                ch for ch in common_channels if ch in available_channels
            ]
            
            if len(channels_to_use) == 0:
                # Try case-insensitive matching
                ch_map = {ch.upper(): ch for ch in available_channels}
                channels_to_use = []
                for ch in common_channels:
                    if ch.upper() in ch_map:
                        channels_to_use.append(ch_map[ch.upper()])
            
            if len(channels_to_use) == 0:
                logger.warning(
                    f"No matching channels found in {filepath}. "
                    f"Available: {available_channels}"
                )
                return None, None, None
            
            # Pick only the common channels
            raw.pick(channels_to_use)
        
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        sfreq = raw.info["sfreq"]
        ch_names = raw.ch_names
        
        return data, ch_names, sfreq
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None, None, None


def inspect_dataset(data_dir, patient_ids=None):
    """
    Inspect the dataset structure and print detailed statistics.
    
    Args:
        data_dir: Path to raw data directory
        patient_ids: List of patient IDs to inspect (None = all)
    
    Returns:
        dict: Dataset statistics
    """
    stats = {
        "patients": {},
        "total_files": 0,
        "total_seizure_files": 0,
        "total_seizures": 0,
        "total_duration_hours": 0,
        "total_seizure_duration_seconds": 0,
    }
    
    if patient_ids is None:
        patient_ids = discover_patients(data_dir)
    
    for patient_id in patient_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Inspecting patient: {patient_id}")
        logger.info(f"{'='*60}")
        
        # Get EDF files
        edf_files = get_patient_files(data_dir, patient_id)
        
        # Parse annotations
        summary_path = get_summary_path(data_dir, patient_id)
        if summary_path:
            seizure_annotations = parse_summary_file(summary_path)
        else:
            logger.warning(f"No summary file found for {patient_id}")
            seizure_annotations = {}
        
        # Count seizures
        n_seizures = sum(len(v) for v in seizure_annotations.values())
        seizure_duration = sum(
            sum(end - start for start, end in seiz_list)
            for seiz_list in seizure_annotations.values()
        )
        
        # Inspect first EDF file for channel info
        channels_info = None
        sfreq = None
        file_duration_total = 0
        
        if edf_files:
            try:
                raw = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False)
                channels_info = raw.ch_names
                sfreq = raw.info["sfreq"]
                
                # Get duration of each file
                for edf_path in edf_files:
                    try:
                        raw_temp = mne.io.read_raw_edf(
                            edf_path, preload=False, verbose=False
                        )
                        file_duration_total += raw_temp.n_times / raw_temp.info["sfreq"]
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Could not read {edf_files[0]}: {e}")
        
        patient_stats = {
            "n_files": len(edf_files),
            "n_seizure_files": len(seizure_annotations),
            "n_seizures": n_seizures,
            "seizure_duration_sec": seizure_duration,
            "total_duration_hours": file_duration_total / 3600,
            "channels": channels_info,
            "sampling_rate": sfreq,
            "seizure_annotations": {
                k: [(int(s), int(e)) for s, e in v]
                for k, v in seizure_annotations.items()
            },
        }
        
        stats["patients"][patient_id] = patient_stats
        stats["total_files"] += len(edf_files)
        stats["total_seizure_files"] += len(seizure_annotations)
        stats["total_seizures"] += n_seizures
        stats["total_duration_hours"] += file_duration_total / 3600
        stats["total_seizure_duration_seconds"] += seizure_duration
        
        logger.info(f"  Files: {len(edf_files)}")
        logger.info(f"  Seizure files: {len(seizure_annotations)}")
        logger.info(f"  Total seizures: {n_seizures}")
        logger.info(f"  Seizure duration: {seizure_duration}s")
        logger.info(f"  Recording duration: {file_duration_total/3600:.2f} hours")
        if channels_info:
            logger.info(f"  Channels ({len(channels_info)}): {channels_info}")
        if sfreq:
            logger.info(f"  Sampling rate: {sfreq} Hz")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total patients: {len(stats['patients'])}")
    logger.info(f"Total EDF files: {stats['total_files']}")
    logger.info(f"Total seizure files: {stats['total_seizure_files']}")
    logger.info(f"Total seizures: {stats['total_seizures']}")
    logger.info(f"Total recording: {stats['total_duration_hours']:.2f} hours")
    logger.info(
        f"Total seizure duration: {stats['total_seizure_duration_seconds']}s "
        f"({stats['total_seizure_duration_seconds']/60:.1f} min)"
    )
    if stats['total_duration_hours'] > 0:
        seizure_ratio = (
            stats['total_seizure_duration_seconds'] / 
            (stats['total_duration_hours'] * 3600)
        ) * 100
        logger.info(f"Seizure ratio: {seizure_ratio:.3f}%")
    
    return stats
