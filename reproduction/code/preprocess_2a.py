"""Port of EEG-Conformer/preprocessing/BCIIV2a.m to the open BNCI 001-2014 .mat release.

Mirrors the MATLAB script step for step:
  epoch = s(Pos+500 : Pos+1499, 1:22)   -> [2,6] s after trial onset, 22 EEG channels
  NaN -> 0
  cheby2(6, 60, [4 40] Hz) + filtfilt, applied per trial
Saves (1000, 22, 288) 'data' + (288,) 'label' so conformer.py's loader works unchanged.
"""
import argparse, os                                     # [MOD]
import numpy as np, scipy.io as sio
from scipy.signal import cheby2, filtfilt

ap = argparse.ArgumentParser()                          # [MOD] paths were hardcoded
ap.add_argument('--raw', default='data_raw', help='dir holding A0{1..9}{T,E}.mat from BNCI 001-2014')
ap.add_argument('--out', default='data_proc', help='dir to write the epoched+filtered .mat files')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

FS, WL, WH, NS, NCH = 250, 4, 40, 1000, 22
b, a = cheby2(6, 60, [WL * 2 / FS, WH * 2 / FS], btype='bandpass')
PADLEN = 3 * (max(len(a), len(b)) - 1)          # MATLAB filtfilt default

for sub in range(1, 10):
    for sess in ('T', 'E'):
        runs = sio.loadmat(f'{args.raw}/A0{sub}{sess}.mat')['data']
        X, y = [], []
        for i in range(runs.shape[1]):
            r = runs[0, i][0, 0]
            if r['trial'].size == 0:
                continue                         # EOG calibration run
            s, tr, lab = r['X'], r['trial'].ravel(), r['y'].ravel()
            for p, l in zip(tr, lab):
                X.append(s[p - 1 + 500: p - 1 + 1500, :NCH])
                y.append(l)
        data = np.stack(X, axis=2).astype(np.float64)   # (1000, 22, 288)
        data[np.isnan(data)] = 0
        for j in range(data.shape[2]):
            data[:, :, j] = filtfilt(b, a, data[:, :, j], axis=0, padlen=PADLEN)
        label = np.array(y, dtype=np.float64).reshape(-1, 1)  # column vector, as classlabel is in BCIIV2a.m
        sio.savemat(f'{args.out}/A0{sub}{sess}.mat',
                    {'data': data, 'label': label})
        print(f'A0{sub}{sess}: data{data.shape} label{label.shape} '
              f'classes={np.unique(label).astype(int)} std={data.std():.4g}', flush=True)
