# ESC-50 Environmental Sound Classification — Preprocessing Pipeline

This directory contains the production data preprocessing and feature extraction suite for the ESC-50 dataset, engineered according to the **EURASIP 2024 Dual-Modal Feature Fusion (DMFF)** architecture.

---

## 1. Directory Structure

```text
backend/src/soundsense/ml/
├── raw/                               # Symlink/reference to raw 44.1 kHz ESC-50 audio
├── processed/
│   ├── audio_22k/                     # Resampled (22,050 Hz, 5.0s, mono) normalized audio files (.wav)
│   └── features/
│       ├── log_mel/                   # 128-band Log-Mel Spectrogram matrices (.npy) [128, 216]
│       ├── mfcc/                      # 40-coefficient MFCC matrices (.npy) [40, 216]
│       └── recurrence_plot/           # Threshold-Free continuous Recurrence Plots (.npy) [128, 128]
├── splits/
│   ├── dataset_manifest.csv           # Master verified dataset manifest
│   ├── fold_1_train.csv / val / test  # Fold 1 train/val/test splits (Zero data leakage)
│   ├── fold_2_train.csv / val / test  # Fold 2 splits
│   ├── fold_3_train.csv / val / test  # Fold 3 splits
│   ├── fold_4_train.csv / val / test  # Fold 4 splits
│   ├── fold_5_train.csv / val / test  # Fold 5 splits
│   └── split_statistics.json          # Split distribution statistics
├── reports/
│   ├── preprocessing_report.json      # Full execution telemetry and timing
│   └── feature_visualization.png      # Multi-modal diagnostic verification plot
├── config.py                          # Centralized hyperparameter and path definitions
├── audio_processor.py                 # Resampling, normalization, active segment detection, PAA
├── feature_extractor.py               # Log-Mel, MFCC, and Threshold-Free Recurrence Plot computation
├── data_augmentation.py               # Time stretch, pitch shift, and additive noise expansion
├── dataset_splits.py                  # ESC-50 5-fold cross-validation split generator
├── dataset_loader.py                  # High-throughput PyTorch Dataset & DataLoader
├── batch_preprocessor.py              # Multi-threaded batch processor runner
├── models.py                          # PyTorch models: ResNet-18, VGG-16, SResNet, CNN-GRU
├── download_checkpoints.py            # Checkpoint retriever & cache manager
└── verify_and_report.py               # Feature integrity verification and plotting script
```

---

## 2. Signal Processing Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Target Sample Rate** | `22,050 Hz` | Standardized audio sample rate across all 2,000 ESC-50 clips |
| **Duration / Samples** | `5.0 s` / `110,250` | Exact alignment via zero-padding or center truncation |
| **Channels** | `1` (Mono) | Phase-aligned single-channel audio |
| **Normalization** | Peak $[-1.0, 1.0]$ | $y_{\text{norm}} = y / \max(|y|)$ |
| **Active Segment** | Energy-based | Frame energy analysis detects longest energetic acoustic activity |
| **FFT Window ($N_{\text{fft}}$)** | `1024` | 46.4 ms spectral window |
| **Hop Length** | `512` | 23.2 ms frame step |
| **Mel Filterbanks** | `128` | Frequency range 0 Hz to 11,025 Hz (Nyquist) |
| **MFCC Coefficients** | `40` | Cepstral representation capturing formant and spectral envelope |
| **PAA Dimension ($K$)** | `128` | Piecewise Aggregate Approximation condenses waveform to length 128 |
| **Recurrence Plot** | `128 × 128` | Continuous Euclidean distance trajectory $R_{i,j} = \|x_i - x_j\|$ |

---

## 3. Dual-Modal Feature Fusion Acoustic Paradigm

Traditional environmental sound classifiers rely exclusively on linear time-frequency representations (STFT / Log-Mel), discarding phase dynamics and nonlinear physical generation processes.

This pipeline extracts complementary representations:
1. **Macroscopic Harmonic Modality (Log-Mel & MFCC)**:
   - Captures energy distributions, harmonic frequencies, and spectral timbre across 128 frequency bands.
   - Fed into the **SResNet** spatial CNN backbone with SwiGLU gating.
2. **Microscopic Chaotic Modality (Threshold-Free Recurrence Plot)**:
   - Uses Piecewise Aggregate Approximation (PAA) followed by unthresholded Euclidean phase-space distance mapping.
   - Captures nonlinear acoustic turbulence, friction, and recurrence patterns without losing subtle gradients through binarization.
   - Fed into the **Dual-Modal Feature Fusion (DMFF)** cross-attention mechanism.

---

## 4. 5-Fold Cross-Validation Protocol (Zero Data Leakage)

ESC-50 is partitioned into 5 balanced folds. For each evaluation fold $k \in \{1, 2, 3, 4, 5\}$:
- **Test Fold**: Fold $k$ (400 samples, exactly 8 per class)
- **Validation Fold**: Fold $(k \bmod 5) + 1$ (400 samples)
- **Training Folds**: Remaining 3 folds (1,200 samples)
- Each fold maintains strictly disjoint source recordings to avoid data leakage.

---

## 5. Usage

### Fine-tune AST for SoundSense

SoundSense's **AST (Audio Spectrogram Transformer)** option uses the pretrained
`MIT/ast-finetuned-audioset-10-10-0.4593` weights and fine-tunes a 50-class head
on ESC-50. The first prediction is available after training completes. AST uses
16 kHz mono input and saves its best validation checkpoint under
`artifacts/checkpoints/ast_esc50_mic_calibrated/`; this generated checkpoint is intentionally ignored by git.

1. Install the Python requirements from the project root: `pip install -r requirements.txt`.
2. Set `SOUNDSENSE_ESC50_DIR` to the ESC-50 dataset directory. SoundSense accepts
   either `wav_files/` with `esc50_labels.csv`, or ESC-50's original `audio/`
   with `meta/esc50.csv`. For the dataset already present on this computer, run
   in PowerShell:
   `$env:SOUNDSENSE_ESC50_DIR = 'C:\Users\Rohan Putta\OneDrive\Desktop\HorrorRecognition\SOUND_CLASSIFICATION\backend\dataset'`.
   Generate splits if needed with `python -m soundsense.ml.dataset_splits`.
3. Start the FastAPI app (`python main.py`), sign in as a developer, and choose
   **AST (Audio Spectrogram Transformer)** in Training & Model. Select one fold,
   use a batch size of 8 or less, and start training. The pretrained AST weights
   are downloaded from Hugging Face on the first run; a GPU is recommended.
4. After the job reports completion, choose AST in Classifier Studio to predict.

Training currently handles one fold per job. The job's validation metrics are
measured on that fold's validation split; they are not five-fold cross-validation
results. AST prediction requires the trained checkpoint and fails clearly when it
is missing. Other model choices still use the existing demo training flow.
AST predictions are closed-set (the model always ranks the 50 ESC-50 classes); its
results do not currently include out-of-distribution detection.

To grant a registered account the developer role, run
`python -m soundsense.api.auth promote your-email@example.com` from the project root.

### Run Batch Preprocessing
```bash
python -m soundsense.ml.batch_preprocessor --workers 8
```

### Verify Integrity & Generate Plots
```bash
python -m soundsense.ml.verify_and_report
```

### PyTorch Model Training & Inference Integration
```python
from soundsense.ml.dataset_loader import get_dataloaders
from soundsense.ml.models import ResNet18_ESC50, VGG16_ESC50

# 1. Initialize models
resnet18 = ResNet18_ESC50(in_channels=1, num_classes=50)
vgg16 = VGG16_ESC50(in_channels=1, num_classes=50)

# 2. Load fold 1 dataloaders
loaders = get_dataloaders(fold=1, batch_size=32)
train_loader = loaders["train"]

for batch in train_loader:
    log_mel = batch["log_mel"]            # torch.Size([32, 1, 128, 216])
    labels = batch["target"]              # torch.Size([32])
    
    # Forward pass
    outputs = resnet18(log_mel)           # torch.Size([32, 50])
    break
```
