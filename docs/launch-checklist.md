# Launch checklist

Items to complete before the public showcase deployment:

- [ ] Configure persistent storage on the hosting platform for SQLite. Set `SOUNDSENSE_RUNTIME_DIR` to the mounted persistent directory so `soundsense.db` and `.jwt_secret` survive restarts and redeploys.
- [ ] Deploy, create a test account, save a prediction, restart/redeploy the service, then confirm the account can still sign in and its prediction history remains available.
