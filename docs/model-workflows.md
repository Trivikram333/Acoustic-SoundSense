# SoundSense

Environmental sound classification for the 50 ESC-50 classes.

VGG-16 and CNN-GRU now load their own compatible checkpoints from
`artifacts/weights/vgg16_best_model.pth` and `artifacts/weights/cnn_gru_best_model.pth`.
Run `.\.venv\Scripts\python.exe evaluate_models.py` to measure their fold-1
accuracy, confusion matrices, per-clip predictions, and CPU latency.
Their original training splits are unknown, so these are checkpoint evaluation
scores, not verified held-out test results. Reports are in
`artifacts/training_logs/vgg16_evaluation.json` and `artifacts/training_logs/cnngru_evaluation.json`.
Frontend comparison charts and the measured performance table read evaluation
reports via `/api/models/measured`; unavailable measurements display N/A.

## Run on this Windows computer

Start `start.ps1` in PowerShell, or run `.\.venv\Scripts\python.exe main.py`.
Open http://localhost:3001 and choose **AST** in the prediction model selector.
Upload a clip or use the microphone. Inference runs locally from the trained
checkpoint under `artifacts/checkpoints/ast_esc50_mic_calibrated/`.

Python dependencies are installed in `.venv`; `requirements.lock.txt` records
the versions installed. Recreate with Python 3.11 and
`python -m pip install -r requirements.lock.txt` inside a new virtual environment.
The FastAPI service serves both the frontend and API; Node is optional.

The local `.env` points `SOUNDSENSE_ESC50_DIR` to the existing dataset in
HorrorRecognition. Both `audio/` + `meta/esc50.csv` and
`wav_files/` + `esc50_labels.csv` are supported.

## AST training

Run `.\.venv\Scripts\python.exe train_ast.py` to train the classification head
on frozen pretrained AST embeddings. This CPU training uses folds 3, 4, 5 for
training, fold 2 for validation/checkpoint selection, and fold 1 only for final
evaluation. It runs 100 head-training epochs and writes a measured report to
the checkpoint folder.

The current checkpoint also includes the user supplied clapping recording as a
mic calibration example. Its validation and held-out test measurements are in
`artifacts/checkpoints/ast_esc50_mic_calibrated/calibration_report.json`.

A separate clap versus clock-tick specialist is trained from those ESC-50
categories, with the user recording added as a clap example. AST uses it to
resolve predictions when the main model ranks either of those classes first.
Retrain it with `.\.venv\Scripts\python.exe train_clap_clock.py --mic-clap "path\to\clapping.ogg"`.

The AST input uses 512 spectrogram frames for five-second ESC-50 clips;
pretrained positional embeddings are cropped to match. Waveforms use 16 kHz,
mono, five seconds, and peak normalization in both training and inference.
Embeddings are cached under `artifacts/training_logs/` to speed up repeated head training.

The developer Training view also supports full AST fine-tuning, with one fold
per job. Full fine-tuning requires considerably more compute and memory.
Other architecture training controls remain simulations.

## Independent ResNet-18 and SResNet

CNN14 uses the PANNs AudioSet-pretrained backbone from
https://zenodo.org/records/3987831 and the model implementation distributed by
https://github.com/qiuqiangkong/panns_inference (MIT code).
Run `.\.venv\Scripts\python.exe train_cnn14.py` to train its ESC-50 linear head
on frozen embeddings, using folds 3–5 for training, 2 for validation, and 1 for
testing. The backbone is `artifacts/checkpoints/Cnn14_mAP_0.431.pth`; the trained head is
`artifacts/checkpoints/cnn14_esc50_head.pt`. Both are required for deployment.
Results are saved to `artifacts/training_logs/cnn14_evaluation.json`. Select CNN14 in the
frontend after restarting the Python backend. CNN14 training uses this CLI;
the developer-panel CNN14 training option remains a demonstration.

Run `.\.venv\Scripts\python.exe train_independent.py` for real training (up to
40 epochs per model, early stopping after 8 epochs without validation improvement).
The CLI trains these models; their developer-panel training controls remain demo controls.
Training uses ESC-50 folds 3–5, validation uses fold 2, and the final evaluation
uses fold 1 (400 clips). Existing weights with unknown training provenance are
excluded. These are single-fold measurements, not five-fold benchmark scores.
Reports, per-clip test predictions, and confusion matrices are saved in
`artifacts/training_logs/resnet18_evaluation.json` and `artifacts/training_logs/sresnet_evaluation.json`.
Independent checkpoints are `artifacts/checkpoints/resnet18_independent.pt` and
`artifacts/checkpoints/sresnet_independent.pt`. Select the model in Prediction Model;
the frontend shows measured test accuracy after completion. Prediction requires
the Python backend and fails explicitly if its checkpoint is unavailable.
The checkpoints reload automatically when changed on disk.

AST ranks the 50 ESC-50 classes. Confidence is a softmax score; unknown sounds
can still receive a known class. Inputs longer than five seconds use their first
five seconds. The trained checkpoint must be copied when deploying to another
computer. Restart the server after replacing it outside the Training view.
