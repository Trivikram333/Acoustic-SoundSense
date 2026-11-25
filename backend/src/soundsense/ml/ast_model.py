"""Fine-tuning and inference helpers for the Audio Spectrogram Transformer."""

from __future__ import annotations

from soundsense.core.paths import ARTIFACTS_DIR, DATA_DIR

import math
import random
import time
import io
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from . import config
from .huggingface_weights import ensure_huggingface_weights

BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "ast_esc50_mic_calibrated"
SAMPLE_RATE = 16000
_INFERENCE_CACHE = None
_CLAP_CLOCK_CACHE = None


def _decode_audio(source, target_sr: int = SAMPLE_RATE, target_duration: float = 5.0) -> np.ndarray:
    if isinstance(source, (bytes, bytearray)):
        audio, sr = sf.read(io.BytesIO(source), dtype="float32")
    elif isinstance(source, np.ndarray):
        audio, sr = source.astype(np.float32), target_sr
    else:
        audio, sr = sf.read(str(source), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    audio = np.nan_to_num(audio.astype(np.float32))
    peak = np.max(np.abs(audio)) if len(audio) else 0
    if peak > 1e-6:
        audio = audio / peak
    samples = int(target_sr * target_duration)
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    return audio[:samples]


def get_trained_ast():
    """Load the fine-tuned ESC-50 model; never substitute an unrelated label head."""
    global _INFERENCE_CACHE
    if _INFERENCE_CACHE is not None:
        return _INFERENCE_CACHE
    ensure_huggingface_weights()
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Google AST has not been fine-tuned yet. Train it with the developer Training view; expected {CHECKPOINT_DIR}."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True, attn_implementation='sdpa')
    model.to(device).eval()
    _INFERENCE_CACHE = (extractor, model, device)
    return _INFERENCE_CACHE


def predict_ast(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, float, float | None]:
    started = time.perf_counter()
    extractor, model, device = get_trained_ast()
    waveform = _decode_audio(audio, sample_rate, target_duration=max(5.0, len(audio) / sample_rate))
    window_samples = SAMPLE_RATE * 5
    hop_samples = SAMPLE_RATE * 2
    if len(waveform) <= window_samples:
        windows = [waveform]
    else:
        starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
        final_start = len(waveform) - window_samples
        if starts[-1] != final_start:
            starts.append(final_start)
        windows = [waveform[start:start + window_samples] for start in starts]

    # Prefer the clearest/highest-energy portion of a mic clip. Long mic takes
    # often contain several seconds of room tone before or after the target sound.
    energies = [float(np.sqrt(np.mean(window ** 2))) for window in windows]
    selected = windows[int(np.argmax(energies))]
    inputs = extractor(selected, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        embedding = model.audio_spectrogram_transformer(inputs.input_values.to(device)).pooler_output
        logits = model.classifier(embedding)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    global _CLAP_CLOCK_CACHE
    specialist_path = CHECKPOINT_DIR / "clap_clock_specialist.npz"
    if _CLAP_CLOCK_CACHE is None and specialist_path.is_file():
        stored = np.load(specialist_path)
        _CLAP_CLOCK_CACHE = {key: stored[key] for key in ("mean", "scale", "coef", "intercept")}
    clap_probability = None
    if _CLAP_CLOCK_CACHE is not None:
        specialist = _CLAP_CLOCK_CACHE
        z = (embedding[0].detach().cpu().numpy() - specialist["mean"]) / specialist["scale"]
        score = float(np.dot(z, specialist["coef"]) + specialist["intercept"])
        clap_probability = float(1.0 / (1.0 + np.exp(-np.clip(score, -40, 40))))
    return probs, (time.perf_counter() - started) * 1000, clap_probability


class _ESC50AudioDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False):
        self.rows = pd.read_csv(csv_path)
        self.augment = augment
        self.extractor = ASTFeatureExtractor.from_pretrained(BASE_MODEL)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = config.RAW_DATA_DIR / str(row["filename"])
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = _decode_audio(audio, SAMPLE_RATE)
        if self.augment:
            if random.random() < 0.5:
                audio = np.roll(audio, random.randint(-SAMPLE_RATE // 2, SAMPLE_RATE // 2))
            audio = np.clip(audio * random.uniform(0.8, 1.2) + np.random.normal(0, 0.002, audio.shape), -1, 1)
        values = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values[0]
        return values, int(row["target"])


def dataset_ready(fold: int) -> tuple[Path, Path]:
    train_csv = config.SPLITS_DIR / f"fold_{fold}_train.csv"
    val_csv = config.SPLITS_DIR / f"fold_{fold}_val.csv"
    if not config.RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(f"ESC-50 WAV files are missing: {config.RAW_DATA_DIR}")
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError(f"ESC-50 fold {fold} train/validation split files are missing under {config.SPLITS_DIR}")
    for split in (train_csv, val_csv):
        rows = pd.read_csv(split)
        if rows.empty or not all((config.RAW_DATA_DIR / str(name)).is_file() for name in rows["filename"]):
            raise FileNotFoundError(f"Audio referenced by {split.name} is missing; install the ESC-50 dataset first.")
    return train_csv, val_csv


def train_ast(
    *, fold: int, epochs: int, batch_size: int, learning_rate: float,
    augment: bool, progress: Callable[[int, int, dict], None],
) -> dict:
    """Fine-tune pretrained AST weights on one leakage-safe ESC-50 fold."""
    train_csv, val_csv = dataset_ready(fold)
    train_data = _ESC50AudioDataset(train_csv, augment=augment)
    val_data = _ESC50AudioDataset(val_csv)
    workers = 0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)

    labels = pd.read_csv(config.SPLITS_DIR / "dataset_manifest.csv")
    label_names = {int(row.target): str(row.category) for row in labels[["target", "category"]].drop_duplicates().itertuples(index=False)}
    id2label = {i: label_names.get(i, str(i)) for i in range(50)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = ASTForAudioClassification.from_pretrained(
        BASE_MODEL, num_labels=50, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_accuracy = -1.0
    best_loss = math.inf
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = train_count = 0
        for input_values, targets in train_loader:
            input_values, targets = input_values.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(input_values=input_values, labels=targets)
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = targets.size(0)
            train_loss_sum += float(output.loss.detach()) * count
            train_correct += int((output.logits.argmax(-1) == targets).sum())
            train_count += count

        model.eval()
        val_loss_sum = 0.0
        val_correct = val_count = 0
        with torch.inference_mode():
            for input_values, targets in val_loader:
                input_values, targets = input_values.to(device), targets.to(device)
                output = model(input_values=input_values, labels=targets)
                count = targets.size(0)
                val_loss_sum += float(output.loss) * count
                val_correct += int((output.logits.argmax(-1) == targets).sum())
                val_count += count

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        train_accuracy = 100.0 * train_correct / max(1, train_count)
        val_accuracy = 100.0 * val_correct / max(1, val_count)
        progress(epoch, epochs, {
            "trainLoss": round(train_loss, 4), "valLoss": round(val_loss, 4),
            "trainAcc": round(train_accuracy, 2), "valAcc": round(val_accuracy, 2),
        })
        if val_accuracy > best_accuracy:
            best_accuracy, best_loss = val_accuracy, val_loss
            model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
            train_data.extractor.save_pretrained(CHECKPOINT_DIR)

    global _INFERENCE_CACHE
    _INFERENCE_CACHE = None
    return {"fold": fold, "bestValidationAccuracy": round(best_accuracy, 2), "bestValidationLoss": round(best_loss, 4), "checkpoint": str(CHECKPOINT_DIR)}
