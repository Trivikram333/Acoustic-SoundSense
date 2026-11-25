# SoundSense Project Guide

This guide is a map for anyone new to the repository. Start with the section that
matches what you want to do, then follow the links to the detailed documentation.

## Quick links

| If you want to… | Start here |
| --- | --- |
| Run SoundSense locally | [`README.md`](README.md#run-locally-windows-python-311), [`start.ps1`](start.ps1) |
| Change the page layout or content | [`frontend/index.html`](frontend/index.html) |
| Change frontend behavior | [`frontend/js/app.js`](frontend/js/app.js) |
| Change frontend styles or themes | [`frontend/css/styles.css`](frontend/css/styles.css) |
| Change an API endpoint | [`backend/src/soundsense/api/routes.py`](backend/src/soundsense/api/routes.py) |
| Change authentication | [`backend/src/soundsense/api/auth.py`](backend/src/soundsense/api/auth.py) |
| Change audio processing or inference | [`backend/src/soundsense/ml/`](backend/src/soundsense/ml/) |
| Train or evaluate a model | [`backend/src/soundsense/tools/`](backend/src/soundsense/tools/), [`docs/model-workflows.md`](docs/model-workflows.md) |
| Configure dataset and local paths | [`.env.example`](.env.example), [`backend/src/soundsense/core/paths.py`](backend/src/soundsense/core/paths.py) |
| Understand how parts fit together | [`docs/architecture.md`](docs/architecture.md) |
| Work on preprocessing | [`docs/preprocessing.md`](docs/preprocessing.md) |

## Repository map

```text
SoundSense/
├── frontend/                 Browser app served by FastAPI
│   ├── index.html             Page markup and theme preference bootstrap
│   ├── css/styles.css         Layout, components, responsive styles, themes
│   ├── js/app.js              Main UI, API calls, prediction and theme behavior
│   ├── js/auth.js             Frontend account and sign-in behavior
│   ├── js/fx.js               Visual effects
│   └── assets/                Images and frontend metadata
├── backend/src/soundsense/   Python application package
│   ├── main.py                FastAPI app construction and frontend hosting
│   ├── __main__.py            Enables `python -m soundsense`
│   ├── api/                   HTTP routes, schemas, authentication
│   ├── core/                  Shared settings and filesystem paths
│   ├── services/              Training job helpers and runtime state
│   ├── ml/                    Audio processing, model code, inference, data loading
│   └── tools/
│       ├── training/          Model training and calibration commands
│       └── evaluation/        Evaluation and reporting commands
├── backend/tests/             Backend regression tests
├── docs/                      Architecture, development, and ML workflow notes
├── data/                       Local dataset and generated preprocessing files
├── artifacts/                  Local model weights, checkpoints, and reports
├── var/                        Local database, uploads, and runtime state
├── deployment/                 Docker image definition
├── legacy/node/                Archived Node/Express prototype; not the active app
├── start.ps1                   Windows startup helper
├── main.py                     Compatibility startup entry point
├── compose.yaml                Docker Compose configuration
├── pyproject.toml              Python package/build configuration
├── requirements.txt            Runtime Python dependencies
└── requirements-dev.txt        Development dependencies
```

## Where common work belongs

### Frontend

The frontend is plain HTML, CSS, and JavaScript; there is no bundler or separate
frontend install step. `index.html` is the browser entry point. Most application
logic is in `js/app.js`; authentication helpers live in `js/auth.js`. Keep visual
changes in `css/styles.css` and images or static metadata in `assets/`.

### Backend and API

The active server is FastAPI. `backend/src/soundsense/main.py` builds the app,
registers the API, and serves the frontend. Put request handling and HTTP response
logic in `api/`; request/response shapes are in `api/schemas.py`. Shared paths and
environment-backed settings are in `core/paths.py`. Training job coordination is
in `services/`.

### Audio and models

`ml/` contains model definitions, checkpoint loading, feature extraction, audio
normalization, dataset reading, and prediction. The model implementations are in
files such as `ast_model.py`, `cnn14_model.py`, `models.py`, and
`independent_models.py`; `acoustic_inference.py` coordinates inference. Training
and evaluation entry points are grouped under `tools/training/` and
`tools/evaluation/`. Run them as Python modules from the project root; see
[`docs/model-workflows.md`](docs/model-workflows.md) for exact commands and model
requirements.

## Local files, data, and weights

The repository intentionally does not include private datasets, trained weights,
uploaded audio, local databases, virtual environments, or training output. The
folders may contain only `.gitkeep` in a clean checkout. Their usual roles are:

- `data/`: raw dataset locations (when configured) and generated preprocessing,
  splits, features, and reports.
- `artifacts/checkpoints/`: current model checkpoints.
- `artifacts/weights/`: compatible or older model weight files.
- `artifacts/training_logs/`: evaluation reports, logs, and cached training data.
- `var/`: local account database, upload files, and other runtime state.

Set `SOUNDSENSE_ESC50_DIR` in the root `.env` to the dataset directory. SoundSense
supports ESC-50 layouts containing `audio/` with `meta/esc50.csv`, or `wav_files/`
with `esc50_labels.csv`. Path defaults and overrides are documented in
[`backend/src/soundsense/core/paths.py`](backend/src/soundsense/core/paths.py) and
`.env.example`. Do not commit `.env`, user recordings, database files, or large
model artifacts.

## Run and develop

For the simplest Windows launch, run `start.ps1` from PowerShell in the repository
root. The app is served at `http://127.0.0.1:3001`; FastAPI's interactive API docs
are at `http://127.0.0.1:3001/docs`. Setup and alternatives are in the root
[`README.md`](README.md). Python development setup, configuration, and checks are
in [`docs/development.md`](docs/development.md).

Useful orientation docs:

- [`docs/architecture.md`](docs/architecture.md): request flow and component
  boundaries.
- [`docs/development.md`](docs/development.md): environment, configuration, and
  developer commands.
- [`docs/model-workflows.md`](docs/model-workflows.md): which models are runnable,
  required checkpoints, and training/evaluation commands.
- [`docs/preprocessing.md`](docs/preprocessing.md): dataset layout and feature
  preprocessing details.
- [`docs/launch-checklist.md`](docs/launch-checklist.md): deployment readiness
  reminders.

## What is active versus archived

Use the Python package under `backend/src/soundsense/` for backend work. The code in
`legacy/node/` is an archived Express prototype and is not used by `start.ps1`,
Docker Compose, or the current deployment configuration. Likewise, local model
files and reports are runtime assets rather than source code; their presence
depends on the machine and `.env` configuration.
