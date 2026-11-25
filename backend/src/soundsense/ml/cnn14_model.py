"""PANNs CNN14 AudioSet backbone and SoundSense ESC-50 transfer head."""

from soundsense.core.paths import ARTIFACTS_DIR, DATA_DIR
import importlib.metadata
import importlib
import sys
import types
import time
from pathlib import Path
import torch
import numpy as np
from .huggingface_weights import ensure_huggingface_weights

ROOT = ARTIFACTS_DIR
BASE = ROOT / 'checkpoints/Cnn14_mAP_0.431.pth'
HEAD = ROOT / 'checkpoints/cnn14_esc50_head.pt'
_backbone = None
_head = None

def backbone():
    global _backbone
    if _backbone is None:
        ensure_huggingface_weights()
        # Import only the model module: upstream __init__ downloads labels into
        # the user's home folder, which ESC-50 transfer does not need.
        directory = importlib.metadata.distribution('panns-inference').locate_file('panns_inference')
        package = types.ModuleType('_soundsense_panns')
        package.__path__ = [str(directory)]
        sys.modules.setdefault('_soundsense_panns', package)
        cls = importlib.import_module('_soundsense_panns.models').Cnn14
        model = cls(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                    fmin=50, fmax=14000, classes_num=527)
        model.load_state_dict(torch.load(BASE, map_location='cpu', weights_only=True)['model'])
        _backbone = model.eval().requires_grad_(False)
    return _backbone

def predict(audio, sr=32000):
    global _head
    if sr != 32000:
        raise ValueError('CNN14 requires 32 kHz audio')
    start = time.perf_counter()
    if _head is None:
        state = torch.load(HEAD, map_location='cpu', weights_only=True)
        _head = torch.nn.Linear(2048, 50).eval()
        _head.load_state_dict(state['state_dict'])
    with torch.inference_mode():
        emb = backbone()(torch.tensor(np.asarray(audio),dtype=torch.float32)[None])['embedding']
        probs = _head(emb).softmax(-1)[0].numpy()
    return probs, (time.perf_counter()-start)*1000
