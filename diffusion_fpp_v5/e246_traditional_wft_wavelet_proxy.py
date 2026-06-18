from pathlib import Path
import json, csv, math
import numpy as np

try:
    from scipy import ndimage
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
    ndimage = None

base=Path('/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix')
phase=Path('/root/autodl-tmp/fpp_ml_phase_cache_960')
out=Path('results/e246_traditional_wft_wavelet_proxy_baselines')
out.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(246)
max_pixels_per_train_sample=1800
ridge_lambda=1e-4

variants=['wft_gaussian_xy','wavelet_gabor_bank_xy','dwt_grad_phase_xy','all_traditional_features_xy']

def load(split):
    return {
        'fringe': np.load(base/f'fringe_{split}_uint8.npy', mmap_mode='r'),
        'instr': np.load(phase/f'phase_instr_{split}_float16.npy', mmap_mode='r'),
        'depth01': np.load(base/f'depth01_{split}_float16.npy', mmap_mode='r'),
        'depth_mm': np.load(base/f'depth_mm_{split}_float32.npy', mmap_mode='r'),
        'mask': np.load(base/f'mask_{split}_uint8.npy', mmap_mode='r'),
        'minmax': np.load(base/f'depth_minmax_{split}_float32.npy', mmap_mode='r'),
    }

def estimate_fx(img):
    # Dominant horizontal fringe carrier from row-averaged spectrum. Ignore DC and very low frequencies.
    row = img.mean(axis=0).astype(np.float32)
    row = row - row.mean()
    spec = np.abs(np.fft.rfft(row))
    lo = max(2, int(0.005 * row.size))
    hi = max(lo + 1, int(0.45 * row.size))
    k = int(np.argmax(spec[lo:hi]) + lo)
    return k / float(row.size)

def gaussian_complex(z, sigma):
    if SCIPY_OK:
        return ndimage.gaussian_filter(z.real, sigma=sigma, mode='reflect') + 1j * ndimage.gaussian_filter(z.imag, sigma=sigma, mode='reflect')
    # Slow fallback: separable FFT-domain Gaussian approximation.
    h,w=z.shape
    fy=np.fft.fftfreq(h)[:,None]
    fx=np.fft.fftfreq(w)[None,:]
    filt=np.exp(-2*(math.pi**2)*(sigma**2)*(fx*fx+fy*fy))
    return np.fft.ifft2(np.fft.fft2(z)*filt)

def wft_features(img, sigmas=(5.0, 11.0)):
    h,w=img.shape
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    fx = estimate_fx(img)
    x = np.arange(w, dtype=np.float32)[None, :]
    carrier = np.exp(-1j * 2.0 * np.pi * fx * x).astype(np.complex64)
    z = img.astype(np.complex64) * carrier
    feats=[]
    for sigma in sigmas:
        local = gaussian_complex(z, sigma)
        phase_ang = np.angle(local).astype(np.float32)
        amp = np.abs(local).astype(np.float32)
        amp = amp / (np.percentile(amp, 99.0) + 1e-6)
        feats.extend([np.sin(phase_ang), np.cos(phase_ang), amp, phase_ang / np.pi])
    return feats

def laplace_gauss(img, sigma):
    if SCIPY_OK:
        return ndimage.gaussian_laplace(img, sigma=sigma, mode='reflect').astype(np.float32)
    h,w=img.shape
    fy=np.fft.fftfreq(h)[:,None]
    fx=np.fft.fftfreq(w)[None,:]
    rr=fx*fx+fy*fy
    filt=-(4*math.pi*math.pi)*rr*np.exp(-2*(math.pi**2)*(sigma**2)*rr)
    return np.fft.ifft2(np.fft.fft2(img)*filt).real.astype(np.float32)

def make_features(fringe_chw, instr_chw, variant):
    img = fringe_chw[0].astype(np.float32) / 255.0
    # phase_instr_order: raw, H sin/cos/res/conf, FTP sin/cos/res/conf, DWT, grad, x, y
    ins = instr_chw.astype(np.float32)
    h,w = img.shape
    x = ins[11].reshape(-1)
    y = ins[12].reshape(-1)
    one = np.ones_like(x, dtype=np.float32)
    raw = ins[0].reshape(-1)
    hs,hc,hr,hq = [ins[i].reshape(-1) for i in [1,2,3,4]]
    fs,fc,fr,fq = [ins[i].reshape(-1) for i in [5,6,7,8]]
    dwt = ins[9].reshape(-1)
    grad = ins[10].reshape(-1)
    cols=[one, x, y, x*x, y*y, x*y]
    if variant in ('wft_gaussian_xy','all_traditional_features_xy'):
        for f in wft_features(img, sigmas=(5.0, 11.0)):
            cols.append(f.reshape(-1).astype(np.float32))
    if variant in ('wavelet_gabor_bank_xy','all_traditional_features_xy'):
        imgn = (img - img.mean()) / (img.std() + 1e-6)
        for sigma in (2.0, 4.0, 8.0, 16.0):
            lg = laplace_gauss(imgn, sigma)
            # Normalize robustly per sample.
            scale = np.percentile(np.abs(lg), 99.0) + 1e-6
            cols.append(np.clip(lg / scale, -3, 3).reshape(-1).astype(np.float32))
        # Also include local energy proxies from cached DWT/gradient.
        cols.extend([dwt, grad])
    if variant in ('dwt_grad_phase_xy','all_traditional_features_xy'):
        cols.extend([hs,hc,hr,hq,fs,fc,fr,fq,dwt,grad,raw,hr*x,hr*y,fr*x,fr*y])
    return np.stack(cols, axis=1)

def fit_variant(variant):
    tr=load('train')
    xtx=None; xty=None; n_pix=0; dim=None
    for i in range(tr['depth01'].shape[0]):
        m=tr['mask'][i,0].reshape(-1)>0
        idx=np.flatnonzero(m)
        if idx.size==0: continue
        if idx.size>max_pixels_per_train_sample:
            idx=rng.choice(idx, size=max_pixels_per_train_sample, replace=False)
        X=make_features(tr['fringe'][i], tr['instr'][i], variant)[idx].astype(np.float64)
        y=tr['depth01'][i,0].astype(np.float32).reshape(-1)[idx].astype(np.float64)
        ok=np.isfinite(X).all(axis=1) & np.isfinite(y)
        X=X[ok]; y=y[ok]
        if X.size==0: continue
        if xtx is None:
            dim=X.shape[1]
            xtx=np.zeros((dim,dim), dtype=np.float64)
            xty=np.zeros((dim,), dtype=np.float64)
        xtx += X.T @ X
        xty += X.T @ y
        n_pix += int(y.size)
    if xtx is None:
        raise RuntimeError('no train pixels')
    reg=np.eye(dim)*ridge_lambda
    reg[0,0]=0.0
    coef=np.linalg.solve(xtx+reg, xty)
    return coef, {'train_pixels': n_pix, 'feature_dim': dim, 'ridge_lambda': ridge_lambda, 'scipy': SCIPY_OK}

def eval_variant(variant, coef, split):
    ds=load(split)
    rows=[]
    for i in range(ds['depth01'].shape[0]):
        X=make_features(ds['fringe'][i], ds['instr'][i], variant).astype(np.float64)
        pred01=np.clip(X@coef,0,1).astype(np.float32).reshape(ds['depth01'][i,0].shape)
        mn,mx=map(float, ds['minmax'][i])
        pred=pred01*(mx-mn)+mn
        gt=ds['depth_mm'][i,0].astype(np.float32)
        m=ds['mask'][i,0].astype(bool)
        err=pred[m]-gt[m]
        rows.append({'sample':i,'rmse':float(np.sqrt(np.mean(err*err))), 'mae':float(np.mean(np.abs(err))), 'valid_pixels':int(m.sum())})
    return rows

def summarize(rows):
    return {k:{'mean':float(np.mean([r[k] for r in rows])), 'std':float(np.std([r[k] for r in rows]))} for k in ['rmse','mae']}

summary={'method_note':'Expanded traditional single-frame proxy baselines: WFT-like Gaussian-windowed Fourier local demodulation, wavelet/LoG multiscale responses, DWT/gradient/Fourier/Hilbert cached single-frame features. Train-split ridge calibration maps features to normalized depth; no teacher phase or GT at test time.', 'variants':{}}
for v in variants:
    print('FIT',v, flush=True)
    coef, meta=fit_variant(v)
    np.save(out/f'{v}_coef.npy', coef)
    summary['variants'][v]={'fit_meta':meta}
    for split in ['val','test']:
        rows=eval_variant(v,coef,split)
        with open(out/f'{split}_{v}_rows.csv','w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['sample','rmse','mae','valid_pixels']); wr.writeheader(); wr.writerows(rows)
        summary['variants'][v][split]=summarize(rows)
    print(v, 'test', summary['variants'][v]['test'], flush=True)
with open(out/'traditional_wft_wavelet_proxy_summary.json','w',encoding='utf-8') as f:
    json.dump(summary,f,indent=2,ensure_ascii=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
