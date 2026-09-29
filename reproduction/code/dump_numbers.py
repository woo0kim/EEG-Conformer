import glob, json, collections, numpy as np
import os, sys

# [MOD] results directory is argv[1] (or $EEGCONF_RESULTS), not a hardcoded scratch path
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EEGCONF_RESULTS', 'results')

PAPER_T2 = [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]
runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(RESULTS, 'curve_*.npz'))):
    d = np.load(f)
    if str(d['tag']) != 'PILOT':
        runs[(str(d['tag']), int(d['subject']))].append(d)

best = lambda d: d['test'].max() * 100
tail = lambda d: d['test'][-100:].mean() * 100
fin  = lambda d: d['test'][-1] * 100
aver = lambda d: d['test'].mean() * 100
vsel = lambda d: d['test'][int(np.nanargmax(d['val']))] * 100

def per_sub(tag, fn):
    m, s_ = [], []
    for s in range(1, 10):
        v = [fn(d) for d in runs.get((tag, s), [])]
        m.append(float(np.mean(v)) if v else None)
        s_.append(float(np.std(v, ddof=1)) if len(v) > 1 else None)
    return m, s_

def flat(tag, fn):
    return [fn(d) for k, v in runs.items() if k[0] == tag for d in v]

out = {'n_runs': {t: sum(len(v) for k, v in runs.items() if k[0] == t) for t in 'ABC'},
       'paper_T2': PAPER_T2, 'paper_T2_avg': float(np.mean(PAPER_T2))}
for tag, fn, nm in [('A', best, 'A_best'), ('A', tail, 'A_tail'), ('A', fin, 'A_final'),
                    ('A', aver, 'A_aver'), ('B', vsel, 'B_sel'), ('B', best, 'B_best'),
                    ('C', best, 'C_best'), ('C', tail, 'C_tail')]:
    m, s_ = per_sub(tag, fn)
    out[nm] = m
    out[nm + '_sd'] = s_
    vals = [x for x in m if x is not None]
    out[nm + '_avg'] = float(np.mean(vals)) if vals else None

for nm, vals in [('gap_best_minus_tail', [best(d) - tail(d) for k, v in runs.items() if k[0] == 'A' for d in v]),
                 ('gap_best_minus_final', [best(d) - fin(d) for k, v in runs.items() if k[0] == 'A' for d in v]),
                 ('gap_B_best_minus_valsel', [best(d) - vsel(d) for k, v in runs.items() if k[0] == 'B' for d in v])]:
    if vals:
        out[nm] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)),
                   'min': float(np.min(vals)), 'max': float(np.max(vals)), 'n': len(vals)}

eps = [int(np.argmax(d['test'])) for k, v in runs.items() if k[0] == 'A' for d in v]
if eps:
    out['argmax_epoch'] = {'median': int(np.median(eps)), 'min': int(min(eps)), 'max': int(max(eps)),
                           'frac_after_1000': float(np.mean([e > 1000 for e in eps]))}
sec = [float(d['sec_per_epoch']) for v in runs.values() for d in v]
if sec:
    out['sec_per_epoch'] = {'mean': float(np.mean(sec)), 'min': float(np.min(sec)), 'max': float(np.max(sec))}

conv = []
for k, v in runs.items():
    if k[0] != 'A':
        continue
    for d in v:
        t = d['test'] * 100
        idx = np.where(t >= t[-100:].mean() - 1.0)[0]
        conv.append(int(idx[0]) if len(idx) else len(t))
if conv:
    out['converge_epoch_median'] = int(np.median(conv))

# spread across seeds of the paper-style statistic
spread = [np.std([best(d) for d in runs[('A', s)]], ddof=1) for s in range(1, 10) if len(runs.get(('A', s), [])) > 1]
if spread:
    out['A_best_seed_sd_mean'] = float(np.mean(spread))
    out['A_best_seed_sd_max'] = float(np.max(spread))

for k in ('A_best', 'A_tail', 'B_sel', 'C_best'):
    a = out.get(k + '_avg')
    if a is not None:
        out[k + '_kappa'] = round((a / 100 - 0.25) / 0.75, 4)
out['paper_kappa'] = round((out['paper_T2_avg'] / 100 - 0.25) / 0.75, 4)

json.dump(out, open(os.path.join(RESULTS, 'numbers.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
