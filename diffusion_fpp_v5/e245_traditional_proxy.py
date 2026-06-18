from pathlib import Path
import json, csv
import numpy as np

base=Path('/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix')
phase=Path('/root/autodl-tmp/fpp_ml_phase_cache_960')
out=Path('results/e245_traditional_single_frame_proxy_baselines')
out.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(245)
max_pixels_per_train_sample=2500
variants=['raw_xy','hilbert_phase_xy','ftp_phase_xy','hilbert_ftp_phase_xy']

def load(split):
    return {
        'instr': np.load(phase/f'phase_instr_{split}_float16.npy', mmap_mode='r'),
        'depth01': np.load(base/f'depth01_{split}_float16.npy', mmap_mode='r'),
        'depth_mm': np.load(base/f'depth_mm_{split}_float32.npy', mmap_mode='r'),
        'mask': np.load(base/f'mask_{split}_uint8.npy', mmap_mode='r'),
        'minmax': np.load(base/f'depth_minmax_{split}_float32.npy', mmap_mode='r'),
    }

def make_features(instr_chw, variant):
    # phase_instr_order:
    # 0 raw, 1 H sin, 2 H cos, 3 H residual, 4 H confidence,
    # 5 FTP sin, 6 FTP cos, 7 FTP residual, 8 FTP confidence,
    # 9 DWT energy, 10 fringe grad, 11 x, 12 y
    raw=instr_chw[0].astype(np.float32).reshape(-1)
    hs=instr_chw[1].astype(np.float32).reshape(-1)
    hc=instr_chw[2].astype(np.float32).reshape(-1)
    hr=instr_chw[3].astype(np.float32).reshape(-1)
    hq=instr_chw[4].astype(np.float32).reshape(-1)
    fs=instr_chw[5].astype(np.float32).reshape(-1)
    fc=instr_chw[6].astype(np.float32).reshape(-1)
    fr=instr_chw[7].astype(np.float32).reshape(-1)
    fq=instr_chw[8].astype(np.float32).reshape(-1)
    x=instr_chw[11].astype(np.float32).reshape(-1)
    y=instr_chw[12].astype(np.float32).reshape(-1)
    one=np.ones_like(x, dtype=np.float32)
    if variant=='raw_xy':
        cols=[one, raw, x, y, raw*x, raw*y, x*x, y*y, x*y]
    elif variant=='hilbert_phase_xy':
        cols=[one, hs, hc, hr, hq, x, y, x*x, y*y, x*y, hr*x, hr*y]
    elif variant=='ftp_phase_xy':
        cols=[one, fs, fc, fr, fq, x, y, x*x, y*y, x*y, fr*x, fr*y]
    elif variant=='hilbert_ftp_phase_xy':
        cols=[one, hs, hc, hr, hq, fs, fc, fr, fq, x, y, x*x, y*y, x*y, hr*x, hr*y, fr*x, fr*y]
    else:
        raise ValueError(variant)
    return np.stack(cols, axis=1)

def fit_variant(variant):
    tr=load('train')
    xs=[]; ys=[]
    n=tr['depth01'].shape[0]
    for i in range(n):
        m=tr['mask'][i,0].reshape(-1)>0
        idx=np.flatnonzero(m)
        if idx.size==0: continue
        if idx.size>max_pixels_per_train_sample:
            idx=rng.choice(idx, size=max_pixels_per_train_sample, replace=False)
        X=make_features(tr['instr'][i], variant)[idx]
        y=tr['depth01'][i,0].astype(np.float32).reshape(-1)[idx]
        ok=np.isfinite(X).all(axis=1) & np.isfinite(y)
        xs.append(X[ok]); ys.append(y[ok])
    X=np.concatenate(xs,axis=0).astype(np.float64)
    y=np.concatenate(ys,axis=0).astype(np.float64)
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=1e-6)
    return coef.astype(np.float64), {'train_pixels': int(y.size), 'rank': int(rank), 'feature_dim': int(X.shape[1])}

def eval_variant(variant, coef, split='test'):
    ds=load(split)
    rows=[]
    for i in range(ds['depth01'].shape[0]):
        X=make_features(ds['instr'][i], variant).astype(np.float64)
        pred01=(X@coef).astype(np.float32)
        pred01=np.clip(pred01,0.0,1.0).reshape(ds['depth01'][i,0].shape)
        mm_min=float(ds['minmax'][i,0]); mm_max=float(ds['minmax'][i,1])
        pred_mm=pred01*(mm_max-mm_min)+mm_min
        gt=ds['depth_mm'][i,0].astype(np.float32)
        m=ds['mask'][i,0].astype(bool)
        if m.sum()==0: continue
        err=pred_mm[m]-gt[m]
        rmse=float(np.sqrt(np.mean(err*err)))
        mae=float(np.mean(np.abs(err)))
        rows.append({'sample':i,'rmse':rmse,'mae':mae,'valid_pixels':int(m.sum())})
    return rows

def summarize(rows):
    return {k:{'mean':float(np.mean([r[k] for r in rows])), 'std':float(np.std([r[k] for r in rows]))} for k in ['rmse','mae']}

summary={
    'method_note': 'Traditional single-frame proxy baselines. They use only A0-derived cached Hilbert/FTP/raw features plus x/y at test time; a train-split empirical calibration maps features to per-sample normalized depth under the same FPP-ML-Bench normalization protocol. This is not a full calibrated FTP/Hilbert FPP renderer.',
    'normalization': 'fit target depth01 on train; test pred01 clipped to [0,1] and denormalized with sample depth_minmax, matching ML benchmark protocol',
    'variants': {}
}
for v in variants:
    print('FIT',v, flush=True)
    coef, meta=fit_variant(v)
    np.save(out/f'{v}_coef.npy', coef)
    for split in ['val','test']:
        rows=eval_variant(v,coef,split)
        with open(out/f'{split}_{v}_rows.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=['sample','rmse','mae','valid_pixels']); w.writeheader(); w.writerows(rows)
        summary['variants'].setdefault(v, {})[split]=summarize(rows)
    summary['variants'][v]['fit_meta']=meta
    print(v, 'test', summary['variants'][v]['test'], flush=True)
with open(out/'traditional_single_frame_proxy_summary.json','w',encoding='utf-8') as f:
    json.dump(summary,f,indent=2,ensure_ascii=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
