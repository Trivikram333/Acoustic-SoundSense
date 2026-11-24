"""Audit the compatible saved ResNet checkpoint without changing deployment."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import json
import hashlib
import time
import pandas as pd
import numpy as np
import torch
from soundsense.ml.models import build_model
from soundsense.ml.independent_models import ROOT, features
from soundsense.ml.config import RAW_DATA_DIR, LABELS_CSV
from soundsense.ml.acoustic_inference import load_audio_waveform

def main():
    torch.set_num_threads(4)
    path=ROOT/'weights/resnet_best_model-1.pth'
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model('resnet',pretrained=False)
    model.load_state_dict({('net.'+k[len('model.'):] if k.startswith('model.') else k):v for k,v in state.items()})
    model.eval(); df=pd.read_csv(LABELS_CSV); df=df[df.fold.eq(1)]
    records=[]; times=[]
    with torch.inference_mode():
        for i,row in enumerate(df.itertuples()):
            start=time.perf_counter(); audio,sr=load_audio_waveform(RAW_DATA_DIR/row.filename)
            f={k:v[None] for k,v in features(audio,sr).items()}
            pred=int(model(feats=f).argmax(-1)); times.append((time.perf_counter()-start)*1000)
            records.append({'filename':row.filename,'target':int(row.target),'prediction':pred})
            if i%100==0: print(i,'/400',flush=True)
    correct=sum(r['target']==r['prediction'] for r in records)
    report={'model':'resnet18','status':'completed','checkpoint':'weights/resnet_best_model-1.pth',
            'checkpointSHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'testAccuracy':correct/len(records),
            'testCorrect':correct,'testCount':len(records),'evaluationFold':1,'trainingProvenance':'unknown',
            'protocol':'Saved ResNet-18, model-to-net key mapping; original training split unknown; current log-mel + recurrence pipeline',
            'parameters':sum(p.numel() for p in model.parameters()),'meanLatencyMs':float(np.mean(times[1:])),
            'testPredictions':records}
    (ROOT/'training_logs/resnet18_saved_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SAVED RESNET',correct,'/400',flush=True)

if __name__=='__main__': main()
