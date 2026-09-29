import glob, numpy as np, collections, json
import os, sys

# [MOD] results directory is argv[1] (or $EEGCONF_RESULTS), not a hardcoded scratch path
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EEGCONF_RESULTS', 'results')

PAPER_T2 = [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]
runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(RESULTS, 'curve_*.npz'))):
    d = np.load(f)
    tag = str(d['tag'])
    if tag == 'PILOT':
        continue
    runs[(tag, int(d['subject']))].append(d)

def agg(tag, fn):
    """fn(curve_dict) -> scalar per run; returns per-subject mean, sd, n."""
    out = {}
    for s in range(1, 10):
        vals = [fn(d) for d in runs.get((tag, s), [])]
        out[s] = (np.mean(vals) * 100 if vals else np.nan,
                  np.std(vals, ddof=1) * 100 if len(vals) > 1 else 0.0, len(vals))
    return out

best   = lambda d: d['test'].max()
final  = lambda d: d['test'][-1]
tail   = lambda d: d['test'][-100:].mean()
aver   = lambda d: d['test'].mean()
def valsel(d):
    v = d['val']
    return d['test'][int(np.nanargmax(v))]

print('runs found:', {k: len(v) for k, v in sorted(runs.items())})
print()

A_best, A_fin, A_tail, A_aver = agg('A', best), agg('A', final), agg('A', tail), agg('A', aver)
B_sel, B_best = agg('B', valsel), agg('B', best)
C_best, C_fin = agg('C', best), agg('C', final)

hdr = (f"{'Sub':>4} {'paper':>7} | {'A:best':>13} {'A:final':>13} {'A:tail100':>13} {'A:mean-ep':>13}"
       f" | {'B:val-sel':>13} | {'C:best(perm)':>13} {'C:final':>13}")
print(hdr); print('-' * len(hdr))
def c(t, s): 
    m, sd, n = t[s]
    return f'{m:6.2f}±{sd:4.2f}' if n else '      n/a    '
for s in range(1, 10):
    print(f'{s:>4} {PAPER_T2[s-1]:7.2f} | {c(A_best,s):>13} {c(A_fin,s):>13} {c(A_tail,s):>13} {c(A_aver,s):>13}'
          f' | {c(B_sel,s):>13} | {c(C_best,s):>13} {c(C_fin,s):>13}')

def gm(t):
    v = [t[s][0] for s in range(1, 10) if not np.isnan(t[s][0])]
    return np.mean(v) if v else np.nan
print('-' * len(hdr))
print(f"{'AVG':>4} {np.mean(PAPER_T2):7.2f} | {gm(A_best):11.2f}   {gm(A_fin):11.2f}   {gm(A_tail):11.2f}   "
      f"{gm(A_aver):11.2f}   | {gm(B_sel):11.2f}   | {gm(C_best):11.2f}   {gm(C_fin):11.2f}")

print()
print('kappa (from grand-mean accuracy, 4-class):')
for nm, t in [('paper', None), ('A:best', A_best), ('A:tail100', A_tail), ('B:val-sel', B_sel)]:
    a = np.mean(PAPER_T2) if t is None else gm(t)
    print(f'   {nm:12} acc={a:6.2f}  kappa={(a/100-0.25)/0.75:.4f}')

print()
print('=== selection optimism (same runs, protocol A) ===')
gaps = [(best(d) - tail(d)) * 100 for k, v in runs.items() if k[0] == 'A' for d in v]
print(f'  best-epoch minus tail-100 mean : {np.mean(gaps):.2f} pp  (sd {np.std(gaps, ddof=1):.2f}, n={len(gaps)})')
gaps2 = [(best(d) - final(d)) * 100 for k, v in runs.items() if k[0] == 'A' for d in v]
print(f'  best-epoch minus final epoch   : {np.mean(gaps2):.2f} pp  (sd {np.std(gaps2, ddof=1):.2f}, n={len(gaps2)})')
gaps3 = [(best(d) - valsel(d)) * 100 for k, v in runs.items() if k[0] == 'B' for d in v]
print(f'  [B] best-epoch minus val-selected: {np.mean(gaps3):.2f} pp (sd {np.std(gaps3, ddof=1):.2f}, n={len(gaps3)})')

print()
print('=== where does the "best" epoch land? (protocol A) ===')
eps = [int(np.argmax(d['test'])) for k, v in runs.items() if k[0] == 'A' for d in v]
if eps:
    print(f'  argmax epoch: median {int(np.median(eps))}, min {min(eps)}, max {max(eps)}, '
          f'{100*np.mean([e>1000 for e in eps]):.0f}% land after epoch 1000')

print()
print('=== label-permutation control (protocol C) ===')
cb = [best(d) * 100 for k, v in runs.items() if k[0] == 'C' for d in v]
cf = [final(d) * 100 for k, v in runs.items() if k[0] == 'C' for d in v]
ca = [aver(d) * 100 for k, v in runs.items() if k[0] == 'C' for d in v]
if cb:
    print(f'  best over 2000 epochs : {np.mean(cb):.2f} ± {np.std(cb, ddof=1):.2f}  (chance = 25.00)  n={len(cb)}')
    print(f'  final epoch           : {np.mean(cf):.2f} ± {np.std(cf, ddof=1):.2f}')
    print(f'  mean over all epochs  : {np.mean(ca):.2f} ± {np.std(ca, ddof=1):.2f}')

print()
sec = [float(d['sec_per_epoch']) for v in runs.values() for d in v]
if sec:
    print(f'=== runtime: {np.mean(sec):.3f} s/epoch (A40) over {len(sec)} runs; paper claims 0.27 s/epoch (RTX 3090)')

conv = []
for k, v in runs.items():
    if k[0] != 'A':
        continue
    for d in v:
        t = d['test']; fin = t[-100:].mean()
        idx = np.where(t >= fin - 0.01)[0]
        conv.append(idx[0] if len(idx) else len(t))
if conv:
    print(f'=== convergence: test acc first reaches (tail-100 mean - 1pp) at epoch {int(np.median(conv))} (median); paper says ~250')

json.dump({'A_best': {s: A_best[s][0] for s in A_best}, 'A_tail': {s: A_tail[s][0] for s in A_tail},
           'B_sel': {s: B_sel[s][0] for s in B_sel}, 'C_best': {s: C_best[s][0] for s in C_best}},
          open(os.path.join(RESULTS, 'summary.json'), 'w'), indent=1)
