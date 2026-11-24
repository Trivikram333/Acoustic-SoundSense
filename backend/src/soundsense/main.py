"""ASGI entry point and HTTP application composition."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from soundsense.core.paths import FRONTEND_DIR, PREPROCESSING_DIR
from soundsense.ml import config
from soundsense.api.auth import router as auth_router
from soundsense.api.routes import router

app = FastAPI(title='SoundSense Backend', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True,
                   allow_methods=['GET','POST','PUT','DELETE','OPTIONS'],
                   allow_headers=['Content-Type','Authorization'])
app.include_router(auth_router)
app.include_router(router)
# Retain public audio URLs used by the existing frontend.
if PREPROCESSING_DIR.exists():
    app.mount('/Preproccesing', StaticFiles(directory=str(PREPROCESSING_DIR)), name='preprocessing')
if config.DATASET_DIR.exists():
    app.mount('/ESC-50', StaticFiles(directory=str(config.DATASET_DIR)), name='esc50')
if config.RAW_DATA_DIR.exists():
    app.mount('/audio/esc50', StaticFiles(directory=str(config.RAW_DATA_DIR)), name='audio_esc50')
if FRONTEND_DIR.exists():
    app.mount('/', StaticFiles(directory=str(FRONTEND_DIR), html=True), name='frontend')
