"""
Audio signal processing routines:
- Resampling to uniform 22,050 Hz
- Duration alignment (5.0s, 110,250 samples)
- Peak and RMS normalization
- Energy-based active acoustic segment detection
- Piecewise Aggregate Approximation (PAA) for chaotic trajectory representation
"""

from soundsense.core.paths import ARTIFACTS_DIR, DATA_DIR

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, Any

from . import config


import wave
from pathlib import Path
import numpy as np
import librosa


def load_and_resample(file_path: str | Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load WAV audio via Python's native wave module (bypasses libsndfile Windows open errors) and resample."""
    path_str = str(file_path)

    with wave.open(path_str, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        orig_sr = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    if sample_width == 1:
        y = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        y = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw_u8[:, 0].astype(np.int32)
            | (raw_u8[:, 1].astype(np.int32) << 8)
            | (raw_u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
        y = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        y = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        y = y.reshape(-1, num_channels).mean(axis=1)

    if target_sr is not None and orig_sr != target_sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=target_sr)
        return y.astype(np.float32), target_sr

    return y.astype(np.float32), orig_sr
def peak_normalize(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize waveform to peak amplitude in [-1.0, 1.0]."""
    max_val = np.max(np.abs(audio))
    if max_val > eps:
        return audio / max_val
    return audio


def detect_active_segment(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    frame_len: int = 2048,
    hop_len: int = 512,
    threshold_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Detect the longest contiguous energetic / active acoustic segment.
    Uses short-time windowed energy calculations.
    """
    if len(audio) < frame_len:
        return {
            "start_sample": 0,
            "end_sample": len(audio),
            "start_time": 0.0,
            "end_time": len(audio) / sr,
            "active_ratio": 1.0,
            "max_energy": 0.0,
        }

    # Frame energy computation
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    energy = np.sum(frames ** 2, axis=0)
    max_energy = np.max(energy) if len(energy) > 0 else 0.0
    mean_energy = np.mean(energy) if len(energy) > 0 else 0.0

    threshold = max(mean_energy * threshold_ratio, 1e-6)
    active_mask = energy >= threshold

    # Find longest continuous run of active frames
    longest_start = 0
    longest_len = 0
    curr_start = 0
    curr_len = 0

    for i, is_act in enumerate(active_mask):
        if is_act:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > longest_len:
                longest_len = curr_len
                longest_start = curr_start
        else:
            curr_len = 0

    if longest_len > 0:
        start_sample = longest_start * hop_len
        end_sample = min(len(audio), (longest_start + longest_len) * hop_len + frame_len)
    else:
        start_sample = 0
        end_sample = len(audio)

    return {
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "start_time": round(float(start_sample / sr), 3),
        "end_time": round(float(end_sample / sr), 3),
        "active_ratio": round(float(np.mean(active_mask)), 4),
        "max_energy": round(float(max_energy), 6),
    }


def piecewise_aggregate_approximation(
    audio: np.ndarray,
    target_dim: int = config.PAA_DIM,
) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA).
    Condenses the raw 1D time-series into target_dim bins by computing the
    mean value in each partition. Preserves underlying dynamic trajectory
    for continuous phase-space embedding & Recurrence Plots.
    """
    n = len(audio)
    if n == target_dim:
        return audio.copy()

    # Exact division using array splits
    splits = np.array_split(audio, target_dim)
    paa = np.array([np.mean(segment) for segment in splits], dtype=np.float32)
    return paa


def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sr: int = config.SAMPLE_RATE,
) -> None:
    """Save audio waveform to disk using soundfile (16-bit PCM WAV)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
