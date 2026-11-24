# Architecture

The browser calls `/api/*` on the same origin. FastAPI registers authentication and
application routers before mounting the frontend at `/`. Existing endpoint URLs
and public audio URLs are retained.

The `api` package owns HTTP request validation and response orchestration.
`services` owns training execution helpers and process-local job state. `ml` owns
signal processing, model construction, dataset access, and inference. `core.paths`
loads root `.env` once and resolves shared directories independently of the working
directory. Authentication currently owns SQLite persistence and token creation in
`api/auth.py`; this boundary is explicit and can be extracted into a repository
layer when the application needs more persistence operations.

Training/evaluation tools are importable modules under `tools`; importing the
application does not launch them. Frontend scripts live in `frontend/js`, styles in `frontend/css`, and images,
audio, and metadata in `frontend/assets`. HTML, CSS, and audio lookup paths are
updated together. There is no frontend build pipeline to run.

The Express prototype is archived under `legacy/node`; FastAPI is authoritative.
The architecture remains a single service rather than introducing unnecessary
microservices. Job state and counters are in memory, so run one worker. Some
non-AST training controls are existing simulations; this restructuring does not
turn simulated metrics into measured model performance.

Local state is deliberately separate: `var` contains private account data and
uploads, `artifacts` contains weights and training output, and `data` contains
datasets and generated preprocessing files. None is committed. Source code never
depends on the shell's current working directory to locate these files.
