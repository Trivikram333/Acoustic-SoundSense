"""Evaluate existing independently loaded checkpoints; record provenance limitations."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from soundsense.ml.independent_models import ROOT, load_model, features, NAMES
from soundsense.ml.config import LABELS_CSV, RAW_DATA_DIR
from soundsense.ml.acoustic_inference import load_audio_waveform

def main():
    torch.set_num_threads(4)
    df=pd.read_csv(LABELS_CSV); df=df[df.fold.eq(1)].reset_index(drop=True)
    for key in ('vgg16','cnngru'):
        model=load_model(key); truth=[]; pred=[]; latency=[]
        with torch.inference_mode():
            for i,row in df.iterrows():
                start=time.perf_counter()
                y,sr=load_audio_waveform(RAW_DATA_DIR/row.filename)
                f={k:v[None] for k,v in features(y,sr).items()}
                p=int(model(feats=f).argmax(-1))
                latency.append((time.perf_counter()-start)*1000)
                truth.append(int(row.target)); pred.append(p)
                if i%100==0: print(key,i,'/400',flush=True)
        cm=np.zeros((50,50),dtype=int)
        for a,b in zip(truth,pred): cm[a,b]+=1
        path=ROOT/'weights'/f'{NAMES[key]}_best_model.pth'
        correct=sum(a==b for a,b in zip(truth,pred))
        report={'model':key,'status':'completed','testAccuracy':correct/len(df),'testCorrect':correct,'testCount':len(df),
                'evaluationFold':1,'protocol':'Existing checkpoint evaluation on ESC-50 fold 1; training split unknown; not verified held-out accuracy',
                'trainingProvenance':'unknown','parameters':sum(p.numel() for p in model.parameters()),
                'meanLatencyMs':float(np.mean(latency[1:])),'latencyProtocol':'CPU, 4 threads, batch 1, decode + features + forward, first clip excluded',
                'checkpointSHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'confusionMatrix':cm.tolist(),
                'testPredictions':[{'filename':n,'target':a,'prediction':b} for n,a,b in zip(df.filename,truth,pred)]}
        (ROOT/'training_logs'/f'{key}_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(key,correct,'/400',flush=True)

if __name__=='__main__': main()
