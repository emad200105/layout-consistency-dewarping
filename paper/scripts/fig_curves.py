import gzip, json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, FIGS
from style import plt, LAY, ABL, COL1
U = RESULTS / 'training_logs'
SPE = 17733
def load(files):
    d = {}
    for f in files:
        for l in gzip.open(f, 'rt'):
            r = json.loads(l)
            if r.get('split') in ('val', 'val_epoch'):
                d[r['step']] = r
    s = sorted(d)
    return np.array(s), d
sl, dl = load([U / 'layout_session1.log.jsonl.gz', U / 'layout_session2.log.jsonl.gz'])
sa, da = load([U / 'ablation_session1.log.jsonl.gz', U / 'ablation_session2.log.jsonl.gz'])
common = np.array(sorted(set(sl) & set(sa)))
el = np.array([dl[s]['epe_px'] for s in common]); ea = np.array([da[s]['epe_px'] for s in common])
ep = common / SPE
print('evals', len(common), 'last', common[-1], el[-1], ea[-1])
gap = 100 * (ea - el) / ea
def smooth(x, k=15):
    return np.convolve(np.pad(x, (k//2, k//2), mode='edge'), np.ones(k)/k, mode='valid')
fig, ax = plt.subplots(2, 1, figsize=(COL1, 2.9), sharex=True, gridspec_kw={'height_ratios': [1.25, 1]})
ax[0].plot(ep, el, color=LAY, lw=0.8, label=r'with $\mathcal{L}_{\mathrm{lay}}$ ($\lambda$=0.2)')
ax[0].plot(ep, ea, color=ABL, lw=0.8, label=r'without ($\lambda$=0)')
ax[0].set_yscale('log'); ax[0].set_ylabel('val EPE (px)')
ax[0].set_yticks([3, 4, 6, 10, 20]); ax[0].set_yticklabels(['3', '4', '6', '10', '20'])
ax[0].set_ylim(3.2, 20); ax[0].legend(loc='upper right', frameon=False)
lr = lambda s: 0.5 * (1 + np.cos(np.pi * np.minimum(s, 540000) / 540000))
a2 = ax[0].twinx(); a2.plot(ep, lr(common), color='0.55', ls=':', lw=0.8); a2.set_ylim(0, 1.05)
a2.set_ylabel('LR / LR$_{max}$', color='0.45'); a2.tick_params(colors='0.45'); a2.grid(False)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].plot(ep, gap, color='0.75', lw=0.5)
ax[1].plot(ep, smooth(gap), color='k', lw=1.0, label='EPE gap (rolling mean)')
ax[1].set_ylabel('EPE gap (%)'); ax[1].set_xlabel('epoch')
ax[1].set_ylim(-12, 15); ax[1].set_xlim(0, 30)
ax[1].text(29.6, 11.5, 'positive: ablation worse', ha='right', fontsize=6.5, color='0.3')
fig.align_ylabels(ax)
fig.savefig(FIGS / 'curves.pdf'); fig.savefig(FIGS / 'curves.png', dpi=200)
print('final 5 gap mean', gap[-5:].mean(), 'ep>=25 gap mean', gap[ep>=25].mean(), 'ep 10-20', gap[(ep>=10)&(ep<20)].mean())
