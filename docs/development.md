# Development

Use Python 3.11 and the root `.venv`. Install `requirements-dev.txt`; the existing
`requirements.lock.txt` is the preserved environment snapshot and may include
platform-specific packages. It is not a portable cross-platform lockfile.

Set `PYTHONPATH` to the absolute `backend/src` directory, or install the root
project in editable mode. Run `python -m soundsense` from any working directory.
Change browser code under `frontend`, HTTP logic under `api`, training helpers
under `services`, and audio/model implementation under `ml`.

Configuration is in root `.env`; copy `.env.example` for a new checkout. Relative
path overrides resolve against the repository root. Existing `.env` values are
preserved by migration. `SOUNDSENSE_DB` overrides the default private database.
The JWT key remains beside that database unless `SOUNDSENSE_SECRET` is configured.

Run `python -m unittest discover -s backend/tests -v`. Tests use temporary account
storage and do not touch real users, uploaded files, or trained checkpoints.

For account administration, use `python -m soundsense.api.auth list` or
`python -m soundsense.api.auth promote someone@example.com`.

Container: `docker compose up --build`. Mount local models/data and persist `var`.
Supply production secrets through the hosting environment. Root `render.yaml`
points to `deployment/Dockerfile`; the archived Vercel/Node setup is not active.
