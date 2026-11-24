"""Measure the deployed AST + clap/clock specialist on ESC-50 fold 1."""

from soundsense.core.paths import ARTIFACTS_DIR, PROJECT_ROOT
import json
import time
import numpy as np
import pandas as pd
import torch
from soundsense.ml.config import LABELS_CSV, RAW_DATA_DIR, BASE_DIR
from soundsense.ml.acoustic_inference import run_acoustic_inference
from soundsense.ml.ast_model import get_trained_ast

def main():
    torch.set_num_threads(4)
    df=pd.read_csv(LABELS_CSV); df=df[df.fold.eq(1)]
    records=[]; times=[]; cm=np.zeros((50,50),dtype=int)
    for i,row in enumerate(df.itertuples()):
        start=time.perf_counter()
        result=run_acoustic_inference(RAW_DATA_DIR/row.filename,'ast')
        times.append((time.perf_counter()-start)*1000)
        pred=result['category']['id']; target=int(row.target)
        if pred>=0: cm[target,pred]+=1
        records.append({'filename':row.filename,'target':target,'prediction':pred})
        if i%50==0: print(i,'/400',flush=True)
    correct=sum(r['target']==r['prediction'] for r in records)
    model=get_trained_ast()[1]
    report={'model':'ast','status':'completed','testAccuracy':correct/len(records),'testCount':len(records),'testCorrect':correct,
            'trainFolds':[3,4,5],'validationFold':2,'testFold':1,
            'protocol':'ESC-50 fold 1; calibrated AST head plus clap/clock specialist; one labeled microphone example also used in calibration',
            'method':'Frozen AudioSet AST backbone, ESC-50 head and microphone calibration, clap/clock specialist reranking',
            'parameters':sum(p.numel() for p in model.parameters()),'meanLatencyMs':float(np.mean(times[1:])),
            'latencyProtocol':'CPU 4 threads, batch 1, decode + features + forward + specialist; first clip excluded',
            'confusionMatrix':cm.tolist(),'testPredictions':records}
    (BASE_DIR/'training_logs/ast_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AST',correct,'/',len(records),flush=True)

if __name__=='__main__': main()
