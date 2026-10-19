import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'pdf.fonttype': 42,
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.grid': True,
    'grid.linewidth': 0.3, 'grid.alpha': 0.5,
})
LAY = '#1f5fa8'   # layout arm
ABL = '#c4572a'   # ablation
COL1 = 3.45       # IEEE column width, inches
COL2 = 7.16
