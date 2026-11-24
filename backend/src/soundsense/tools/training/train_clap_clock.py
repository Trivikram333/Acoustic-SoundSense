"""Train a fold-separated clapping versus clock-tick specialist for AST."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
from soundsense.ml import config
from soundsense.ml.ast_model import CHECKPOINT_DIR, SAMPLE_RATE, get_trained_ast


def mic_embedding(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = np.nan_to_num(audio)
    peak = np.max(np.abs(audio)) if len(audio) else 0
    if peak > 1e-6:
        audio = audio / peak
    total, width, hop = SAMPLE_RATE * 15, SAMPLE_RATE * 5, SAMPLE_RATE * 2
    audio = np.pad(audio[:total], (0, max(0, total - len(audio))))
    starts = list(range(0, total - width + 1, hop))
    starts.append(total - width)
    clips = [audio[start:start + width] for start in sorted(set(starts))]
    clip = max(clips, key=lambda item: float(np.mean(item * item)))
    extractor, ast, device = get_trained_ast()
    encoded = extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        result = ast.audio_spectrogram_transformer(encoded.input_values.to(device)).pooler_output
    return result[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic-clap", type=Path, required=True, help="User labeled microphone recording of clapping")
    args = parser.parse_args()
    frame = pd.read_csv(config.LABELS_CSV)
    frame = frame[frame.category.isin(["clapping", "clock_tick"])].copy()
    frame["target"] = (frame.category == "clapping").astype(int)
    cache = np.load(ARTIFACTS_DIR / "training_logs/ast_embeddings.npz")
    features = cache["features"]
    index = {name: i for i, name in enumerate(cache["filenames"].tolist())}

    def subset(folds):
        rows = frame[frame.fold.isin(folds)]
        x = np.stack([features[index[name]] for name in rows.filename])
        return x, rows.target.to_numpy(), rows

    x_train, y_train, _ = subset([3, 4, 5])
    x_val, y_val, _ = subset([2])
    x_test, y_test, _ = subset([1])
    user_feature = mic_embedding(args.mic_clap)
    x_train = np.concatenate([x_train, user_feature[None]], axis=0)
    y_train = np.concatenate([y_train, np.array([1])])

    specialist = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42),
    )
    specialist.fit(x_train, y_train)
    val_accuracy = float(specialist.score(x_val, y_val))
    test_accuracy = float(specialist.score(x_test, y_test))
    user_clap_probability = float(specialist.predict_proba(user_feature[None])[0, 1])

    scaler: StandardScaler = specialist.named_steps["standardscaler"]
    logistic: LogisticRegression = specialist.named_steps["logisticregression"]
    output = CHECKPOINT_DIR / "clap_clock_specialist.npz"
    np.savez(
        output,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=logistic.coef_[0].astype(np.float32),
        intercept=np.float32(logistic.intercept_[0]),
        classes=np.array(["clock_tick", "clapping"]),
    )
    report = {
        "trainSamples": len(x_train),
        "officialTrainSamples": len(x_train) - 1,
        "validationFold": 2,
        "testFold": 1,
        "validationFold2Accuracy": round(val_accuracy * 100, 2),
        "heldOutTestFold1Accuracy": round(test_accuracy * 100, 2),
        "micClappingProbability": round(user_clap_probability, 4),
    }
    (CHECKPOINT_DIR / "clap_clock_specialist_report.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
