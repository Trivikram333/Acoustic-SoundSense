"""Train an ESC-50 head on frozen pretrained AST embeddings, then evaluate once."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from soundsense.ml.ast_model import BASE_MODEL, CHECKPOINT_DIR, _decode_audio
from soundsense.ml import config


def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    started = time.time()
    source = os.environ.get('SOUNDSENSE_AST_SOURCE', BASE_MODEL)
    extractor = ASTFeatureExtractor.from_pretrained(source)
    labels = pd.read_csv(config.LABELS_CSV)
    names = dict(zip(labels.target.astype(int), labels.category))
    model = ASTForAudioClassification.from_pretrained(
        source, num_labels=50, id2label=names, label2id={v:k for k,v in names.items()},
        ignore_mismatched_sizes=True, attn_implementation='sdpa',
    ).eval()
    # ESC-50 clips are five seconds: retain pretrained positions for 512 frames
    # instead of attending over the default ten-second padded input.
    embeddings = model.audio_spectrogram_transformer.embeddings
    frequency_patches = (model.config.num_mel_bins - model.config.patch_size) // model.config.frequency_stride + 1
    old_time = (model.config.max_length - model.config.patch_size) // model.config.time_stride + 1
    new_time = (512 - model.config.patch_size) // model.config.time_stride + 1
    positions = embeddings.position_embeddings.detach()
    grid = positions[:, 2:].reshape(1, frequency_patches, old_time, -1)[:, :, :new_time]
    embeddings.position_embeddings = torch.nn.Parameter(torch.cat([positions[:, :2], grid.reshape(1, frequency_patches * new_time, -1)], dim=1))
    model.config.max_length = 512
    extractor.max_length = 512
    for p in model.audio_spectrogram_transformer.parameters():
        p.requires_grad = False
    cache = ARTIFACTS_DIR / 'training_logs/ast_embeddings.npz'
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        data = np.load(cache)
        features = torch.from_numpy(data['features'])
        if list(data['filenames']) != list(labels.filename):
            raise ValueError('Embedding cache does not match dataset order')
    else:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(labels), 4):
                rows = labels.iloc[start:start + 4]
                waves = [_decode_audio(config.RAW_DATA_DIR / name) for name in rows.filename]
                values = extractor(waves, sampling_rate=16000, return_tensors='pt').input_values
                chunks.append(model.audio_spectrogram_transformer(values).pooler_output.cpu())
                if start % 40 == 0:
                    print(f'Extracted {min(start+4,len(labels))}/{len(labels)} AST embeddings; elapsed {time.time()-started:.0f}s', flush=True)
        features = torch.cat(chunks)
        np.savez(cache, features=features.numpy(), filenames=labels.filename.to_numpy(dtype=str))
    targets = torch.tensor(labels.target.to_numpy(), dtype=torch.long)
    train_mask = torch.tensor(labels.fold.isin([3,4,5]).to_numpy())
    val_mask = torch.tensor((labels.fold == 2).to_numpy())
    test_mask = torch.tensor((labels.fold == 1).to_numpy())
    train_loader = DataLoader(TensorDataset(features[train_mask], targets[train_mask]), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.01)
    best = -1
    history = []
    for epoch in range(1, 101):
        model.classifier.train()
        loss_sum = correct = count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model.classifier(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(-1) == y).sum())
            count += len(y)
        model.classifier.eval()
        with torch.inference_mode():
            val_logits = model.classifier(features[val_mask])
            accuracy = float((val_logits.argmax(-1) == targets[val_mask]).float().mean()) * 100
        history.append(dict(epoch=epoch, trainLoss=loss_sum/count, trainAccuracy=correct/count*100, validationAccuracy=accuracy))
        if accuracy > best:
            best = accuracy
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CHECKPOINT_DIR)
            extractor.save_pretrained(CHECKPOINT_DIR)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/100: validation={accuracy:.2f}%, best={best:.2f}%', flush=True)
    best_model = ASTForAudioClassification.from_pretrained(CHECKPOINT_DIR).eval()
    with torch.inference_mode():
        test_logits = best_model.classifier(features[test_mask])
        test_accuracy = float((test_logits.argmax(-1) == targets[test_mask]).float().mean()) * 100
    report = dict(mode='frozen AST backbone; trained ESC-50 classifier head', seed=42, trainFolds=[3,4,5], validationFold=2, testFold=1, trainSamples=int(train_mask.sum()), validationSamples=int(val_mask.sum()), testSamples=int(test_mask.sum()), bestValidationAccuracy=best, testAccuracy=test_accuracy, epochs=100, elapsedSeconds=time.time()-started, history=history)
    (CHECKPOINT_DIR / 'training_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
