# SoundSense

Environmental sound classification web app: a static HTML/CSS/JavaScript frontend,
a FastAPI API, and local PyTorch inference and training tools.

## Repository structure

```text
backend/src/soundsense/
  main.py              # Compose API routers and static hosting
  api/                 # HTTP routes, authentication, request schemas
  core/                # Environment configuration and shared filesystem paths
  services/            # Training jobs, response helpers, process-local state
  ml/                  # Audio preprocessing, architectures, inference
  tools/
    training/          # Training and calibration command-line tools
    evaluation/        # Evaluation command-line tools
backend/tests/         # API, authentication, and path regression tests
frontend/
  index.html           # Browser entry point
  js/                  # Application, authentication, visual effects
  css/                 # Stylesheets
  assets/              # images/, audio/, data/
deployment/            # Container build configuration
docs/                  # Architecture, development, model workflow references
legacy/node/           # Archived Express prototype (not active)
artifacts/             # Local weights, checkpoints, training reports (ignored)
data/                  # Local dataset and preprocessing output (ignored)
var/                   # Local accounts DB, JWT key, uploads (ignored)
.github/workflows/     # Automated checks
```

## Run locally (Windows, Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\start.ps1
```

Existing installations can keep their current `.venv` and `.env`.
Open http://127.0.0.1:3001 and http://127.0.0.1:3001/docs.
The root `main.py` also remains a compatibility launcher.

For a terminal session, set `$env:PYTHONPATH = "$PWD/backend/src"`, then run
`python -m soundsense` using the virtual environment interpreter. Optional editable
installation: `python -m pip install -e .`; the frontend must remain alongside the
backend in this repository (a Python wheel alone is not a web deployment).

## Models and data

The local migration preserves your files; they are excluded from Git. A fresh clone
needs model weights and datasets installed separately. Put checkpoints under
`artifacts/checkpoints/`, older weights under `artifacts/weights/`, and reports under
`artifacts/training_logs/`. Set `SOUNDSENSE_ESC50_DIR` in `.env` to the ESC-50 dataset.
Generated audio/features/splits belong in `data/preprocessing/`.

Run tools as modules with `backend/src` on `PYTHONPATH`, for example:

```powershell
python -m soundsense.tools.training.train_ast
python -m soundsense.tools.evaluation.evaluate_models
python -m soundsense.ml.batch_preprocessor
```

See [development](docs/development.md), [architecture](docs/architecture.md), and
[model workflows](docs/model-workflows.md). Use module commands with `backend/src` on `PYTHONPATH`.
For a newcomer-oriented map of the repository and links by task, see the
[Project Guide](PROJECT_GUIDE.md).

## Verification and deployment

```powershell
$env:PYTHONPATH = "$PWD/backend/src"
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
docker compose up --build
```

The container does not include local datasets or trained weights. The Render
Blueprint uses the free web service plan and generates `SOUNDSENSE_SECRET`, but does
not attach persistent storage. Account data, uploads, and runtime files can be lost
when the service restarts or redeploys. Free-tier resource limits may also affect
model inference. Add required model files and datasets separately before using
inference.
Use one application worker: training progress and counters are process-local.

### Hugging Face model weights

When a prediction is requested, the backend can fetch supported weight files from
the configured Hugging Face model repository. Set `SOUNDSENSE_HF_REPO_ID` to the
full `owner/repository` identifier. For a private repository, set `HF_TOKEN` as a
secret environment variable in Render; never commit or share the token. The first
download fetches supported weight/config files into `artifacts/`, preserving the
repository's folder structure.

The simplest layout is to upload files under the matching `artifacts/` paths used by
the app, such as `checkpoints/resnet18_independent.pt`,
`checkpoints/sresnet_independent.pt`, `weights/vgg16_best_model.pth`, and
`checkpoints/ast_esc50_mic_calibrated/`. If names or folders differ, set
`SOUNDSENSE_HF_FILE_MAP` to a JSON object mapping repository paths to the expected
paths under `artifacts/`, for example
`{"resnet_best_mmodel.pth":"checkpoints/resnet18_independent.pt"}`. The supplied
`resnet_best_mmodel.pth` filename is also mapped to ResNet18 by default; override it
with `SOUNDSENSE_HF_FILENAME` if needed. Each checkpoint still must match its
architecture and expected state-dict format; one checkpoint cannot power every model
option. On Render's free tier, files may need to be downloaded again after restarts.

## Android app (Capacitor)

Install Node.js LTS and Android Studio with its Android SDK. Set
`window.SOUNDSENSE_API_BASE` in `frontend/js/api-config.js` to the HTTPS URL of the
deployed backend before building; leave it empty only when the frontend and API are
served from the same origin.

From the repository root, run:

```powershell
npm install
npm run android:sync
npm run android:open
```

Build and run the Android app from Android Studio. The Android app bundles the
frontend, but still needs an internet-accessible backend for API features. It
requests microphone permission for audio recording.

## Before uploading to GitHub

Only application source, documentation, configuration examples, and frontend assets
belong in this repository. `.gitignore` excludes local credentials, databases,
models, datasets, environments, and backups. Review `git status --short` and your
staged changes before committing. Large trained models should be distributed
separately through a model registry or explicitly configured Git LFS.
