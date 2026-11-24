"""Shared paths, independent of the working directory."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / '.env')
def configured(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()

ARTIFACTS_DIR = configured('SOUNDSENSE_ARTIFACTS_DIR', PROJECT_ROOT / 'artifacts')
DATA_DIR = configured('SOUNDSENSE_DATA_DIR', PROJECT_ROOT / 'data')
RUNTIME_DIR = configured('SOUNDSENSE_RUNTIME_DIR', PROJECT_ROOT / 'var')
FRONTEND_DIR = PROJECT_ROOT / 'frontend'
PREPROCESSING_DIR = DATA_DIR / 'preprocessing'
UPLOADS_DIR = RUNTIME_DIR / 'uploads'
DB_PATH = configured('SOUNDSENSE_DB', RUNTIME_DIR / 'soundsense.db')
for directory in (UPLOADS_DIR, DB_PATH.parent, ARTIFACTS_DIR / 'checkpoints',
                  ARTIFACTS_DIR / 'weights', ARTIFACTS_DIR / 'training_logs'):
    directory.mkdir(parents=True, exist_ok=True)
