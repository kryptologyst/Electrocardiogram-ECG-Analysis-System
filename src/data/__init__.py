"""Data utilities and preprocessing for ECG signals."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb
from scipy import signal
from scipy.signal import butter, filtfilt


class ECGSignalProcessor:
    """ECG signal preprocessing utilities."""
    
    def __init__(
        self,
        sampling_rate: int = 250,
        filter_low: float = 0.5,
        filter_high: float = 40.0,
        normalize: bool = True,
    ):
        """Initialize ECG signal processor.
        
        Args:
            sampling_rate: Sampling rate of ECG signals.
            filter_low: Low-pass filter cutoff frequency.
            filter_high: High-pass filter cutoff frequency.
            normalize: Whether to normalize signals.
        """
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.normalize = normalize
        
    def apply_bandpass_filter(
        self, 
        signal_data: np.ndarray, 
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
    ) -> np.ndarray:
        """Apply bandpass filter to ECG signal.
        
        Args:
            signal_data: Input ECG signal.
            lowcut: Low cutoff frequency.
            highcut: High cutoff frequency.
            
        Returns:
            Filtered signal.
        """
        if lowcut is None:
            lowcut = self.filter_low
        if highcut is None:
            highcut = self.filter_high
            
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize ECG signal to zero mean and unit variance.
        
        Args:
            signal_data: Input ECG signal.
            
        Returns:
            Normalized signal.
        """
        return (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    def preprocess_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline.
        
        Args:
            signal_data: Raw ECG signal.
            
        Returns:
            Preprocessed signal.
        """
        # Apply bandpass filter
        processed = self.apply_bandpass_filter(signal_data)
        
        # Normalize if requested
        if self.normalize:
            processed = self.normalize_signal(processed)
            
        return processed
    
    def detect_r_peaks(self, signal_data: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal.
        
        Args:
            signal_data: Preprocessed ECG signal.
            
        Returns:
            Array of R-peak indices.
        """
        from scipy.signal import find_peaks
        
        # Find peaks with minimum height and distance
        peaks, _ = find_peaks(
            signal_data,
            height=np.std(signal_data),
            distance=int(self.sampling_rate * 0.3)  # Minimum 300ms between peaks
        )
        
        return peaks


class ECGDataset(Dataset):
    """ECG dataset for training and evaluation."""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_samples: int = 1000,
        seq_length: int = 250,
        num_classes: int = 2,
        synthetic: bool = True,
        preprocessor: Optional[ECGSignalProcessor] = None,
    ):
        """Initialize ECG dataset.
        
        Args:
            data_dir: Directory containing real ECG data.
            num_samples: Number of samples to generate (for synthetic data).
            seq_length: Length of ECG sequences.
            num_classes: Number of classification classes.
            synthetic: Whether to use synthetic data.
            preprocessor: ECG signal preprocessor.
        """
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.synthetic = synthetic
        self.preprocessor = preprocessor or ECGSignalProcessor()
        
        if synthetic:
            self.data, self.labels = self._generate_synthetic_data(num_samples)
        else:
            self.data, self.labels = self._load_real_data(data_dir)
    
    def _generate_synthetic_data(
        self, 
        num_samples: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic ECG data for demonstration.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            Tuple of (data, labels).
        """
        data = []
        labels = []
        
        for i in range(num_samples):
            # Generate different types of ECG patterns
            if i % 4 == 0:  # Normal sinus rhythm
                signal_data = self._generate_normal_rhythm()
                label = 0
            elif i % 4 == 1:  # Atrial fibrillation
                signal_data = self._generate_afib_rhythm()
                label = 1
            elif i % 4 == 2:  # Ventricular tachycardia
                signal_data = self._generate_vtach_rhythm()
                label = 2
            else:  # Other arrhythmia
                signal_data = self._generate_other_rhythm()
                label = 3
            
            # Preprocess the signal
            processed_signal = self.preprocessor.preprocess_signal(signal_data)
            data.append(processed_signal)
            labels.append(label)
        
        return data, labels
    
    def _generate_normal_rhythm(self) -> np.ndarray:
        """Generate normal sinus rhythm ECG."""
        t = np.linspace(0, self.seq_length / 250, self.seq_length)
        
        # Normal ECG components
        p_wave = 0.1 * np.sin(2 * np.pi * 0.8 * t)
        qrs_complex = 0.5 * np.exp(-((t - 0.4) / 0.05) ** 2)
        t_wave = 0.2 * np.exp(-((t - 0.6) / 0.1) ** 2)
        
        # Combine components
        ecg = p_wave + qrs_complex + t_wave
        
        # Add noise
        noise = np.random.normal(0, 0.05, self.seq_length)
        return ecg + noise
    
    def _generate_afib_rhythm(self) -> np.ndarray:
        """Generate atrial fibrillation ECG."""
        t = np.linspace(0, self.seq_length / 250, self.seq_length)
        
        # Irregular rhythm with varying RR intervals
        rr_intervals = np.random.normal(0.8, 0.2, len(t))
        qrs_times = np.cumsum(rr_intervals)
        qrs_times = qrs_times[qrs_times < t[-1]]
        
        ecg = np.zeros_like(t)
        for qrs_time in qrs_times:
            idx = np.argmin(np.abs(t - qrs_time))
            if idx < len(ecg):
                ecg[idx] = 0.5
        
        # Add noise and baseline wander
        noise = np.random.normal(0, 0.1, self.seq_length)
        baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)
        
        return ecg + noise + baseline
    
    def _generate_vtach_rhythm(self) -> np.ndarray:
        """Generate ventricular tachycardia ECG."""
        t = np.linspace(0, self.seq_length / 250, self.seq_length)
        
        # Fast, regular rhythm
        rr_interval = 0.3  # 200 BPM
        qrs_times = np.arange(0, t[-1], rr_interval)
        
        ecg = np.zeros_like(t)
        for qrs_time in qrs_times:
            idx = np.argmin(np.abs(t - qrs_time))
            if idx < len(ecg):
                # Wide QRS complex
                start_idx = max(0, idx - 5)
                end_idx = min(len(ecg), idx + 5)
                ecg[start_idx:end_idx] = 0.3
        
        noise = np.random.normal(0, 0.08, self.seq_length)
        return ecg + noise
    
    def _generate_other_rhythm(self) -> np.ndarray:
        """Generate other arrhythmia ECG."""
        t = np.linspace(0, self.seq_length / 250, self.seq_length)
        
        # Premature ventricular contractions
        ecg = self._generate_normal_rhythm()
        
        # Add PVCs randomly
        pvc_indices = np.random.choice(
            range(50, self.seq_length - 50), 
            size=np.random.randint(1, 4), 
            replace=False
        )
        
        for idx in pvc_indices:
            # Wide, bizarre QRS complex
            start_idx = max(0, idx - 8)
            end_idx = min(len(ecg), idx + 8)
            ecg[start_idx:end_idx] += 0.4 * np.random.randn(end_idx - start_idx)
        
        return ecg
    
    def _load_real_data(self, data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
        """Load real ECG data from directory.
        
        Args:
            data_dir: Directory containing ECG data.
            
        Returns:
            Tuple of (data, labels).
        """
        # This would be implemented to load real ECG data
        # For now, fall back to synthetic data
        return self._generate_synthetic_data(100)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset.
        
        Args:
            idx: Index of item.
            
        Returns:
            Tuple of (signal, label).
        """
        signal = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        return signal, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in dataset.
        
        Returns:
            Dictionary mapping class to count.
        """
        from collections import Counter
        return dict(Counter(self.labels))
