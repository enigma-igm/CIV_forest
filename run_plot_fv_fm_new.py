from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

##### requires the master table of fvfm (much faster)
##### else needs to run on IGM where all production files are

#maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
#outfig = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/paper_plots/fvfm_all.pdf'

### setting the figure
font = {'family' : 'serif', 'weight' : 'normal'}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(14, 11))
ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
xytick_size = 16
xlabel_fontsize = 20
ylabel_fontsize = 22
legend_fontsize = 14
linewidth = 2

fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
outfig = 'paper_plots/fvfm_all.pdf'

### looking at just a subset of all models
logM = np.array([8.50, 9.00, 9.50, 10.00, 10.50, 11.00])
R = np.array([0.1, 0.50, 1.00, 1.50, 2.00, 3.00])
color_idx_logM = np.linspace(0, 1, len(logM))
color_idx_R = np.linspace(0, 1, len(R))
color_idx_logM = color_idx_logM[::-1] # reversing the order so lighter color corresponds to smaller M
color_idx_R = color_idx_R[::-1] # reversing order so lighter color corresponds to smaller R

fm_vs_R = []
fv_vs_R = []
fm_vs_logM = []
fv_vs_logM = []

fvfm_master_logM = np.round(fvfm_master['logM'], 2)
fvfm_master_R = np.round(fvfm_master['R_Mpc'], 2)

for i_R, Rval in enumerate(R):
    want_i = np.where(fvfm_master_R == Rval)[0]
    ax1.plot(fvfm_master['logM'][want_i], fvfm_master['fm'][want_i], '-', lw=linewidth, color=plt.cm.viridis(color_idx_R[i_R]), label='R = %0.2f' % Rval)
ax1.legend(fontsize=legend_fontsize)
ax1.set_ylabel(r'f$_\mathrm{m}$', fontsize=ylabel_fontsize)
ax1.tick_params(axis="both", labelsize=xytick_size)
ax1.xaxis.set_tick_params(labelbottom=True)

for i_logM, logMval in enumerate(logM):
    want_i = np.where(fvfm_master_logM == logMval)[0]
    ax2.plot(fvfm_master['R_Mpc'][want_i], fvfm_master['fm'][want_i], '-', lw=linewidth, color=plt.cm.viridis(color_idx_logM[i_logM]), label='log(M) = %0.2f' % logMval)
ax2.legend(fontsize=legend_fontsize)
ax2.tick_params(axis="both", labelsize=xytick_size)
ax2.xaxis.set_tick_params(labelbottom=True)

for i_R, Rval in enumerate(R):
    want_i = np.where(fvfm_master_R == Rval)[0]
    ax3.plot(fvfm_master['logM'][want_i], fvfm_master['fv'][want_i], '-', lw=linewidth, color=plt.cm.viridis(color_idx_R[i_R]), label='R = %0.2f' % Rval)
ax3.legend(fontsize=legend_fontsize)
ax3.set_xlabel(r'log(M) [M$_{\odot}$]', fontsize=xlabel_fontsize)
ax3.set_ylabel(r'f$_\mathrm{v}$', fontsize=ylabel_fontsize)
ax3.tick_params(axis="both", labelsize=xytick_size)

for i_logM, logMval in enumerate(logM):
    want_i = np.where(fvfm_master_logM == logMval)[0]
    ax4.plot(fvfm_master['R_Mpc'][want_i], fvfm_master['fv'][want_i], '-', lw=linewidth, color=plt.cm.viridis(color_idx_logM[i_logM]), label=r'log(M) = %0.2f' % logMval)
ax4.legend(fontsize=legend_fontsize)
ax4.set_xlabel('R [Mpc]', fontsize=xlabel_fontsize)
ax4.tick_params(axis="both", labelsize=xytick_size)
ax4.xaxis.set_tick_params(labelbottom=True)

plt.tight_layout()
#plt.show()
plt.savefig(outfig)