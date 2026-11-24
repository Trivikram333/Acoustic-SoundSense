"""Train an ESC-50 head on frozen pretrained CNN14 embeddings."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import hashlib
import json
import time
import pandas as pd
import numpy as np
import torch
from soundsense.ml.cnn14_model import ROOT, BASE, HEAD, backbone
from soundsense.ml.config import LABELS_CSV, RAW_DATA_DIR
from soundsense.ml.acoustic_inference import load_audio_waveform

def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    expected='541141fa2ee191a88f24a3219fff024e'
    assert hashlib.md5(BASE.read_bytes()).hexdigest()==expected, 'Pretrained checkpoint checksum mismatch'
    df=pd.read_csv(LABELS_CSV); out=ROOT/'training_logs'; out.mkdir(exist_ok=True)
    cache=out/'cnn14_embeddings.pt'
    if cache.exists():
        saved=torch.load(cache,weights_only=True)
        assert saved['filenames']==df.filename.tolist()
        x=saved['features']
    else:
        chunks=[]; model=backbone()
        with torch.inference_mode():
            for i in range(0,len(df),4):
                waves=[load_audio_waveform(RAW_DATA_DIR/n,target_sr=32000)[0] for n in df.filename.iloc[i:i+4]]
                chunks.append(model(torch.tensor(np.stack(waves)))['embedding'].clone())
                if i%100==0: print(f'embeddings {i}/{len(df)}',flush=True)
        x=torch.cat(chunks); torch.save({'features':x,'filenames':df.filename.tolist()},cache)
    y=torch.tensor(df.target.to_numpy()); train=torch.tensor(df.fold.isin([3,4,5]).to_numpy())
    val=torch.tensor(df.fold.eq(2).to_numpy()); test=torch.tensor(df.fold.eq(1).to_numpy())
    head=torch.nn.Linear(2048,50); opt=torch.optim.AdamW(head.parameters(),lr=0.001,weight_decay=0.01)
    best=-1; stale=0; history=[]
    for epoch in range(1,201):
        head.train(); ix=torch.where(train)[0]; ix=ix[torch.randperm(len(ix))]
        for batch in ix.split(64):
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(head(x[batch]),y[batch],label_smoothing=0.05)
            loss.backward(); opt.step()
        head.eval()
        with torch.inference_mode(): acc=(head(x[val]).argmax(-1)==y[val]).float().mean().item()
        history.append({'epoch':epoch,'validationAccuracy':acc})
        if acc>best:
            best=acc; stale=0; selected=epoch
            torch.save({'state_dict':head.state_dict(),'bestEpoch':epoch},HEAD)
        else: stale+=1
        if epoch%10==0: print(f'head epoch {epoch} validation {acc:.4f}',flush=True)
        if stale>=25: break
    head.load_state_dict(torch.load(HEAD,weights_only=True)['state_dict'])
    with torch.inference_mode(): pred=head(x[test]).argmax(-1)
    truth=y[test]; correct=int((pred==truth).sum()); cm=np.zeros((50,50),dtype=int)
    for a,b in zip(truth.tolist(),pred.tolist()): cm[a,b]+=1
    report={'model':'cnn14','status':'completed','method':'Frozen AudioSet CNN14 backbone; trained ESC-50 linear head',
            'trainFolds':[3,4,5],'validationFold':2,'testFold':1,'seed':42,'bestEpoch':selected,'validationAccuracy':best,
            'testAccuracy':correct/len(truth),'testCount':len(truth),'testCorrect':correct,'history':history,
            'pretrainedMD5':expected,'confusionMatrix':cm.tolist(),
            'testPredictions':[{'filename':n,'target':a,'prediction':b} for n,a,b in zip(df[test.numpy()].filename,truth.tolist(),pred.tolist())]}
    (out/'cnn14_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'TEST {correct}/{len(truth)} = {correct/len(truth):.4f}',flush=True)

if __name__=='__main__': main()
