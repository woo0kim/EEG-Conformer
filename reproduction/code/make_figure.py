import glob, collections, numpy as np
import os, sys

# [MOD] results directory is argv[1] (or $EEGCONF_RESULTS), not a hardcoded scratch path
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EEGCONF_RESULTS', 'results')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8981', '#e6e5e0'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'     # validated categorical slots 1,2,3
PAPER = [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]
BB = dict(facecolor=SURF, edgecolor='none', pad=2)

runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(RESULTS, 'curve_*.npz'))):
    d = np.load(f)
    if str(d['tag']) != 'PILOT':
        runs[(str(d['tag']), int(d['subject']))].append(d)

best = lambda d: d['test'].max() * 100
tail = lambda d: d['test'][-100:].mean() * 100
vsel = lambda d: d['test'][int(np.nanargmax(d['val']))] * 100

def per_sub(tag, fn):
    return [np.mean([fn(d) for d in runs[(tag, s)]]) if runs.get((tag, s)) else np.nan
            for s in range(1, 10)]

a_best, b_sel, c_best = per_sub('A', best), per_sub('B', vsel), per_sub('C', best)
gm = np.nanmean

def clean(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#d5d4cf')
    ax.tick_params(length=0, colors=INK2, labelsize=9)

fig = plt.figure(figsize=(11, 8.0), facecolor=SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.50)

# ---------------- panel A ----------------
ax = fig.add_subplot(gs[0]); clean(ax)
series = [('Paper, Table II', PAPER + [gm(PAPER)], S1),
          ("Reproduced — paper's rule (best epoch on test)", a_best + [gm(a_best)], S2),
          ('Reproduced — validation-selected epoch', b_sel + [gm(b_sel)], S3)]
x = np.arange(10); w = 0.26
for k, (name, vals, col) in enumerate(series):
    ax.bar(x + (k - 1) * w, vals, w * 0.90, label=name, color=col, linewidth=0)
for k, (_, vals, col) in enumerate(series):          # relief: label only the Avg group
    ax.text(9 + (k - 1) * w, vals[9] + 1.5, f'{vals[9]:.1f}', ha='center', va='bottom',
            fontsize=8.5, color=INK, fontweight='bold', rotation=90)

permavg = gm(c_best)
ax.axhline(permavg, color=MUTED, lw=1.3, ls=(0, (5, 3)))
ax.axhline(25, color=MUTED, lw=1.0, ls=':')
ax.text(10.2, permavg, f'scrambled labels,\nsame rule: {permavg:.1f}%', va='center',
        fontsize=8.5, color=INK2, bbox=BB)
ax.text(10.2, 25, 'chance 25%', va='center', fontsize=8.5, color=MUTED, bbox=BB)

ax.set_xticks(x); ax.set_xticklabels([f'S{i}' for i in range(1, 10)] + ['Avg'], color=INK2)
ax.set_xlim(-0.6, 11.7); ax.set_ylim(20, 102)
ax.set_ylabel('Test accuracy (%)', fontsize=9.5, color=INK2)
ax.set_title('BCI IV 2a: reported vs. reproduced accuracy, by epoch-selection rule',
             fontsize=12, color=INK, loc='left', pad=34)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
          loc='lower left', bbox_to_anchor=(0, 1.005), handlelength=1.1, columnspacing=1.4)

# ---------------- panel B: the run with the largest gap ----------------
ax2 = fig.add_subplot(gs[1]); clean(ax2)
cand = [(best(d) - tail(d), k[1], d) for k, v in runs.items() if k[0] == 'A' for d in v]
gap, sub, d = max(cand, key=lambda t_: t_[0])
t = d['test'] * 100
ax2.plot(t, color=S1, lw=0.55, alpha=0.7)
mx = int(np.argmax(t)); tl = t[-100:].mean()
ax2.axvline(mx, color=S2, lw=1.0, ls=(0, (3, 3)), zorder=3)
ax2.axhline(tl, color=S3, lw=2.2, zorder=4)
ax2.plot([mx], [t[mx]], 'o', ms=9, color=S2, zorder=6,
         markeredgecolor=SURF, markeredgewidth=2)
lbl = 'reported "best accuracy" = %.2f%%, at epoch %d (dashed line)' % (t[mx], mx)
ax2.text(0.025, 0.95, lbl, transform=ax2.transAxes, fontsize=9.5, color=INK,
         va='center', bbox=BB, zorder=7)
ax2.text(1985, tl - 2.5, 'mean of last 100 epochs = %.2f%%' % tl, ha='right', va='top',
         fontsize=9.5, color=INK, bbox=BB, zorder=7)
ax2.set_xlim(0, 2000); ax2.set_ylim(20, 104)
ax2.set_xlabel('Training epoch', fontsize=9.5, color=INK2)
ax2.set_ylabel('Test accuracy (%)', fontsize=9.5, color=INK2)
ax2.set_title('Subject %d, one run: the test set is scored every epoch and the maximum is reported' % sub,
              fontsize=12, color=INK, loc='left', pad=10)

fig.savefig(os.path.join(RESULTS, 'validation_figure.png'), dpi=200,
            bbox_inches='tight', facecolor=SURF)
print('figure written (panel B: subject %d, gap %.2f pp)' % (sub, gap))
