"""Train independent models on folds 3-5, select on fold 2, test once on fold 1."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import argparse
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from soundsense.ml.config import LABELS_CSV, RAW_DATA_DIR
from soundsense.ml.audio_processor import load_and_resample
from soundsense.ml.independent_models import ROOT, NAMES, features
from soundsense.ml.models import build_model

def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)

def run(keys, epochs=40, patience=8):
    torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    out = ROOT / 'training_logs'; out.mkdir(exist_ok=True)
    df = pd.read_csv(LABELS_CSV)
    cache = out / 'cnn_features_v1.pt'
    if not cache.exists():
        rows = []
        for i,row in df.iterrows():
            y,sr = load_and_resample(RAW_DATA_DIR / row.filename)
            rows.append(features(y,sr))
            if i % 100 == 0: print(f'features {i}/{len(df)}', flush=True)
        torch.save({'filename':df.filename.tolist(), 'log_mel':torch.stack([f['log_mel'] for f in rows]),
                    'rp':torch.stack([f['rp'] for f in rows])}, cache)
    data = torch.load(cache, weights_only=True)
    assert data['filename'] == df.filename.tolist(), 'Feature manifest changed'
    loaders = {}
    for split,mask in [('train',df.fold.isin([3,4,5])),('val',df.fold.eq(2)),('test',df.fold.eq(1))]:
        ix = torch.tensor(np.flatnonzero(mask))
        ds = TensorDataset(data['log_mel'][ix],data['rp'][ix],torch.tensor(df.target.to_numpy())[ix])
        loaders[split] = DataLoader(ds,batch_size=16,shuffle=split=='train')
    def evaluate(model, split):
        model.eval(); truth=[]; pred=[]; loss=0
        with torch.inference_mode():
            for mel,rp,y in loaders[split]:
                logits = model(feats={'log_mel':mel,'rp':rp})
                loss += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
                truth.extend(y.tolist()); pred.extend(logits.argmax(-1).tolist())
        return sum(a==b for a,b in zip(truth,pred))/len(truth),loss/len(truth),truth,pred
    for key in keys:
        torch.manual_seed(42)
        model = build_model(NAMES[key],pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=0.5)
        report = {'model':key,'status':'training','seed':42,'initialization':'random; existing unverified weights excluded',
                  'trainFolds':[3,4,5],'validationFold':2,'testFold':1,'history':[],
                  'protocol':'single held-out fold; validation selection; not five-fold cross-validation',
                  'parameters':sum(p.numel() for p in model.parameters())}
        best=-1; stale=0; start=time.time()
        report_path=out / f'{key}_evaluation.json'
        ckpt_path=ROOT / 'checkpoints' / f'{key}_independent.pt'
        for epoch in range(1,epochs+1):
            model.train(); total_loss=0; correct=0; total=0
            for mel,rp,y in loaders['train']:
                # Mask short time/frequency bands on training spectrograms only.
                mel=mel.clone()
                t=random.randrange(max(1,mel.shape[-1]-12)); mel[:,:,:,t:t+12]=0
                f=random.randrange(116); mel[:,:,f:f+12,:]=0
                optimizer.zero_grad()
                logits=model(feats={'log_mel':mel,'rp':rp})
                loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=0.1)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                total_loss+=loss.item()*len(y); correct+=(logits.argmax(-1)==y).sum().item(); total+=len(y)
            acc,vl,_,_=evaluate(model,'val'); scheduler.step(vl)
            report['history'].append({'epoch':epoch,'trainLoss':total_loss/total,'trainAccuracy':correct/total,'validationAccuracy':acc,'validationLoss':vl})
            if acc > best:
                best=acc; stale=0
                torch.save({'state_dict':model.state_dict(),'model':key,'bestEpoch':epoch,'trainFolds':[3,4,5],'validationFold':2},ckpt_path)
                report['bestEpoch']=epoch; report['validationAccuracy']=acc
            else: stale+=1
            report['elapsedSeconds']=time.time()-start
            atomic_json(report_path,report)
            print(f'{key} epoch {epoch}: train={correct/total:.3f} val={acc:.3f} seconds={time.time()-start:.0f}',flush=True)
            if stale>=patience: break
        model.load_state_dict(torch.load(ckpt_path,weights_only=True)['state_dict'])
        acc,loss,truth,pred=evaluate(model,'test')
        report.update(status='completed',testAccuracy=acc,testCount=len(truth),testCorrect=sum(a==b for a,b in zip(truth,pred)),testLoss=loss)
        cm=np.zeros((50,50),dtype=int)
        for a,b in zip(truth,pred): cm[a,b]+=1
        report['confusionMatrix']=cm.tolist()
        report['testPredictions']=[{'filename':n,'target':a,'prediction':b} for n,a,b in zip(df[df.fold.eq(1)].filename,truth,pred)]
        atomic_json(report_path,report)
        print(f'{key} TEST {acc:.4f}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=list(NAMES),choices=list(NAMES))
    p.add_argument('--epochs',type=int,default=40); p.add_argument('--patience',type=int,default=8)
    a=p.parse_args(); run(a.models,a.epochs,a.patience)
