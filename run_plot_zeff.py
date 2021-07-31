from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import halos_skewers
import matplotlib as mpl

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

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(13.5,6.5))
ax1, ax2 = axes[0], axes[1]
xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2.5

### files
outfig = 'paper_plots/logZeff_new.pdf'

fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
fvfm_master_logM = np.round(fvfm_master['logM'], 2)
fvfm_master_R = np.round(fvfm_master['R_Mpc'], 2)
logZ_fid = -3.5

# looking at just a subset of all models
logM = np.array([8.50, 9.00, 9.50, 10.00, 10.50, 11.00])
R = np.array([0.1, 0.50, 1.00, 1.50, 2.00, 3.00])
color_idx_logM = np.linspace(0, 1, len(logM))
color_idx_R = np.linspace(0, 1, len(R))
color_idx_logM = color_idx_logM[::-1] # reversing the order so lighter color corresponds to smaller M
color_idx_R = color_idx_R[::-1] # reversing order so lighter color corresponds to smaller R

# all models
#logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

for i_R, Rval in enumerate(R):
    want_iR = np.where(fvfm_master_R == Rval)[0]
    logZ_eff = halos_skewers.calc_igm_Zeff(fvfm_master['fm'][want_iR], logZ_fid=logZ_fid)
    ax1.plot(fvfm_master['logM'][want_iR], logZ_eff, '-', lw=2, color=plt.cm.viridis(color_idx_R[i_R]), label='R = %0.2f Mpc' % Rval)

ax1.set_xlabel(r'log(M) [M$_{\odot}$]', fontsize=xylabel_fontsize)
#ax1.set_ylabel(r'log(Z$_{\mathrm{eff}}$)', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'[C/H]$_{\mathrm{eff}}$', fontsize=xylabel_fontsize)
ax1.legend(fontsize=legend_fontsize, ncol=2)
ax1.tick_params(axis="both", labelsize=xytick_size)

for i_logM, logMval in enumerate(logM):
    want_ilogM = np.where(fvfm_master_logM == logMval)[0]
    logZ_eff = halos_skewers.calc_igm_Zeff(fvfm_master['fm'][want_ilogM], logZ_fid=logZ_fid)
    ax2.plot(fvfm_master['R_Mpc'][want_ilogM], logZ_eff, '-', lw=2, color=plt.cm.viridis(color_idx_logM[i_logM]), label=r'log(M) = %0.2f M$_{\odot}$' % logMval)

ax2.set_xlabel('R [Mpc]', fontsize=xylabel_fontsize)
ax2.legend(fontsize=legend_fontsize)
ax2.tick_params(axis="both", labelsize=xytick_size)

plt.tight_layout()
plt.savefig(outfig)
plt.show()

"""
plot_vs_R = False # if False, then plot vs logM (at fixed R)
plt.figure(figsize=(8,6))
if plot_vs_R:
    for i_logM, logMval in enumerate(logM):
        want_logM = np.where(fvfm_all['logM'] == logMval)[0]
        logZ_eff, _ = halos_skewers.calc_igm_Zeff(fvfm_all['fm'][want_logM], logZ_fid=logZ_fid)
        plt.plot(fvfm_all['R_Mpc'][want_logM], logZ_eff, 'o-', ms=3, label='logM = %0.2f' % logMval)

    plt.xlabel('R (Mpc)', fontsize=15)
    plt.legend(fontsize=10)

else:
    fvfm_all_R = []
    for i in range(len(fvfm_all)):
        fvfm_all_R.append(round(fvfm_all['R_Mpc'][i], 2))
 
    for i_R, Rval in enumerate(R):
        want_R = np.where(np.array(fvfm_all_R) == round(Rval, 2))[0]
        logZ_eff, _ = halos_skewers.calc_igm_Zeff(fvfm_all['fm'][want_R], logZ_fid=logZ_fid)
        plt.plot(fvfm_all['logM'][want_R], logZ_eff, 'o-', ms=3, label='R = %0.2f Mpc' % Rval)

    plt.xlabel('logM', fontsize=15)
    plt.legend(fontsize=10, ncol=4)

plt.ylabel(r'logZ$_{eff}$', fontsize=15)
plt.title(r'logZ$_{fid}$ = %0.1f' % logZ_fid)
plt.tight_layout()
plt.show()
"""
