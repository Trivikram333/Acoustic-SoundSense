"""On-demand synchronization of model weights from a Hugging Face model repo."""

import json
import os
import shutil
import threading
from pathlib import Path, PurePosixPath

from soundsense.core.paths import ARTIFACTS_DIR

_SYNC_LOCK = threading.Lock()
_SYNCED_REPO = None
_WEIGHT_PATTERNS = [
    '*.pt', '**/*.pt',
    '*.pth', '**/*.pth',
    '*.bin', '**/*.bin',
    '*.safetensors', '**/*.safetensors',
    '*.npz', '**/*.npz',
    'config.json', '**/config.json',
    'preprocessor_config.json', '**/preprocessor_config.json',
]


def _safe_repo_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or '\\' in value or path.is_absolute() or '..' in path.parts:
        raise ValueError('Hugging Face file paths must be relative paths inside the repository.')
    return path


def _copy_file_map() -> None:
    raw_map = os.environ.get('SOUNDSENSE_HF_FILE_MAP')
    file_map = json.loads(raw_map) if raw_map else {}
    if not isinstance(file_map, dict):
        raise ValueError('SOUNDSENSE_HF_FILE_MAP must be a JSON object mapping repo paths to artifact paths.')

    for source_name, target_name in file_map.items():
        source_rel = _safe_repo_path(source_name)
        target_rel = _safe_repo_path(target_name)
        source = ARTIFACTS_DIR.joinpath(*source_rel.parts)
        target = ARTIFACTS_DIR.joinpath(*target_rel.parts)
        if not source.is_file():
            raise FileNotFoundError(f'Mapped Hugging Face checkpoint was not downloaded: {source_name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def ensure_huggingface_weights() -> None:
    """Download all supported weight files once, preserving repo directory structure."""
    global _SYNCED_REPO
    repo_id = os.environ.get('SOUNDSENSE_HF_REPO_ID')
    if not repo_id or _SYNCED_REPO == repo_id:
        return

    with _SYNC_LOCK:
        if _SYNCED_REPO == repo_id:
            return

        from huggingface_hub import snapshot_download

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type='model',
            token=os.environ.get('HF_TOKEN'),
            local_dir=str(ARTIFACTS_DIR),
            allow_patterns=_WEIGHT_PATTERNS,
        )

        # Backwards-compatible default for the single ResNet checkpoint previously configured.
        resnet_filename = os.environ.get('SOUNDSENSE_HF_FILENAME', 'resnet_best_mmodel.pth')
        source_rel = _safe_repo_path(resnet_filename)
        source = ARTIFACTS_DIR.joinpath(*source_rel.parts)
        target = ARTIFACTS_DIR / 'checkpoints' / 'resnet18_independent.pt'
        if source.is_file() and not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        _copy_file_map()
        _SYNCED_REPO = repo_id
