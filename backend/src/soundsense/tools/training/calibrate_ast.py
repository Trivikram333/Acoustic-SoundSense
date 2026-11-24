"""Adapt the AST classification head to one explicitly labeled mic example."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from soundsense.ml import config
from soundsense.ml.ast_model import CHECKPOINT_DIR, SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Labeled microphone recording of clapping")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
    checkpoint = ASTForAudioClassification.from_pretrained(CHECKPOINT_DIR, attn_implementation="sdpa").to(device)
    extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT_DIR)

    audio, sr = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    if peak > 1e-6:
        audio = audio / peak
    total = SAMPLE_RATE * 15
    audio = np.pad(audio[:total], (0, max(0, total - len(audio))))
    window, hop = SAMPLE_RATE * 5, SAMPLE_RATE * 2
    starts = list(range(0, total - window + 1, hop))
    starts.append(total - window)
    selected = max((audio[s:s + window] for s in sorted(set(starts))), key=lambda x: float(np.mean(x*x)))
    encoded = extractor(selected, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        mic_feature = checkpoint.audio_spectrogram_transformer(encoded).pooler_output.cpu()
        mic_logits_before = checkpoint.classifier(mic_feature)
    print("Before calibration:", checkpoint.config.id2label[int(mic_logits_before.argmax(-1))])

    labels = pd.read_csv(config.LABELS_CSV)
    cached = np.load(ARTIFACTS_DIR / "training_logs/ast_embeddings.npz")
    cache_map = {name: i for i, name in enumerate(cached["filenames"].tolist())}
    feature_table = torch.from_numpy(cached["features"])
    train_rows = labels[labels.fold.isin([3, 4, 5])]
    train_indices = [cache_map[name] for name in train_rows.filename]
    x_train = feature_table[train_indices]
    y_train = torch.tensor(train_rows.target.to_numpy(), dtype=torch.long)
    # Replaying the labeled mic example prevents a single sample from being
    # drowned out by the 1,200 official training clips.
    x_train = torch.cat([x_train, mic_feature.repeat(40, 1)])
    y_train = torch.cat([y_train, torch.full((40,), 22, dtype=torch.long)])
    val_rows = labels[labels.fold == 2]
    val_indices = [cache_map[name] for name in val_rows.filename]
    x_val, y_val = feature_table[val_indices], torch.tensor(val_rows.target.to_numpy(), dtype=torch.long)
    test_rows = labels[labels.fold == 1]
    test_indices = [cache_map[name] for name in test_rows.filename]
    x_test, y_test = feature_table[test_indices], torch.tensor(test_rows.target.to_numpy(), dtype=torch.long)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(checkpoint.classifier.parameters(), lr=0.0001, weight_decay=0.01)
    output = CHECKPOINT_DIR.with_name("ast_esc50_mic_calibrated")
    best_val = -1.0
    best_state = None
    for epoch in range(1, 41):
        checkpoint.classifier.train()
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(checkpoint.classifier(features), target)
            loss.backward()
            optimizer.step()
        checkpoint.classifier.eval()
        with torch.inference_mode():
            val_acc = float((checkpoint.classifier(x_val).argmax(-1) == y_val).float().mean()) * 100
            mic_label = int(checkpoint.classifier(mic_feature).argmax(-1))
        if mic_label == 22 and val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in checkpoint.state_dict().items()}
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: validation {val_acc:.2f}%, mic label {checkpoint.config.id2label[mic_label]}", flush=True)

    if best_state is None or best_val < 96.0:
        raise RuntimeError(f"No mic-calibrated checkpoint preserved validation accuracy (best {best_val:.2f}%).")
    checkpoint.load_state_dict(best_state)
    checkpoint.save_pretrained(output)
    extractor.save_pretrained(output)
    checkpoint.eval()
    with torch.inference_mode():
        test_acc = float((checkpoint.classifier(x_test).argmax(-1) == y_test).float().mean()) * 100
        mic_logits = checkpoint.classifier(mic_feature)
        mic_probs = torch.softmax(mic_logits, dim=-1)[0]
    report = {
        "calibrationSample": args.audio.name,
        "calibrationLabel": "clapping",
        "method": "AST backbone frozen; classifier head calibrated with the labeled mic sample and replay from ESC-50 training folds",
        "trainFolds": [3, 4, 5], "validationFold": 2, "testFold": 1,
        "bestValidationAccuracy": round(best_val, 2),
        "heldOutTestAccuracy": round(test_acc, 2),
        "micSamplePrediction": checkpoint.config.id2label[int(mic_logits.argmax(-1))],
        "micSampleConfidence": round(float(mic_probs.max()), 4),
        "micClappingConfidence": round(float(mic_probs[22]), 4),
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
