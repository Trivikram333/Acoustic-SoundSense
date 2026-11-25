"""Independent CNN checkpoints with shared, reproducible audio preprocessing."""

from soundsense.core.paths import ARTIFACTS_DIR, DATA_DIR
import os
import time
from pathlib import Path
import numpy as np
import torch
from .models import build_model
from .audio_processor import peak_normalize
from .feature_extractor import extract_all_features
from .huggingface_weights import ensure_huggingface_weights

ROOT = ARTIFACTS_DIR
NAMES = {'resnet18': 'resnet', 'sresnet': 'sresnet', 'vgg16': 'vgg16', 'cnngru': 'cnn_gru'}
CACHE = {}


def features(audio, sr=22050):
    audio = np.pad(audio[:110250], (0, max(0, 110250-len(audio))))
    f = extract_all_features(peak_normalize(audio), sr=sr)
    return {k: torch.tensor(f[v], dtype=torch.float32)[None] for k, v in
            [('log_mel', 'log_mel'), ('rp', 'recurrence_plot')]}

def load_model(key):
    ensure_huggingface_weights()
    path = ROOT / 'checkpoints' / f'{key}_independent.pt'
    if key in {'vgg16', 'cnngru'} and not path.exists():
        path = ROOT / 'weights' / f'{NAMES[key]}_best_model.pth'
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint found for {key}. Configure a compatible local checkpoint"
            " or configure SOUNDSENSE_HF_REPO_ID and any required SOUNDSENSE_HF_FILE_MAP entries."
        )
    stamp = path.stat().st_mtime_ns
    if key not in CACHE or CACHE[key][0] != stamp:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model = build_model(NAMES[key], pretrained=False)
        model.load_state_dict(ckpt.get('state_dict', ckpt))
        CACHE[key] = (stamp, model.eval())
    return CACHE[key][1]

def predict(key, audio, sr):
    start = time.perf_counter()
    f = {k: v[None] for k,v in features(audio, sr).items()}
    with torch.inference_mode():
        p = load_model(key)(feats=f).softmax(-1)[0].numpy()
    return p, (time.perf_counter()-start)*1000
