from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers
import matplotlib.style as style

##### requires the master table of fvfm (much faster)
##### else needs to run on IGM where all production files are

#maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
#outfig = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/paper_plots/fvfm_all.pdf'

fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
outfig = 'paper_plots/fvfm_all.pdf'

# looking at just a subset of all models
logM = np.array([8.50, 9.00, 9.50, 10.00, 10.50, 11.00])
R = np.array([0.1, 0.50, 1.00, 1.50, 2.00, 3.00])
color_idx_logM = np.linspace(0, 1, len(logM))
color_idx_R = np.linspace(0, 1, len(R))

fm_vs_R = []
fv_vs_R = []
fm_vs_logM = []
fv_vs_logM = []

fvfm_master_logM = np.round(fvfm_master['logM'], 2)
fvfm_master_R = np.round(fvfm_master['R_Mpc'], 2)

plt.figure(figsize=(13,10.5))
plt.subplot(221)
for i_R, Rval in enumerate(R):
    want_i = np.where(fvfm_master_R == Rval)[0]
    plt.plot(fvfm_master['logM'][want_i], fvfm_master['fm'][want_i], '-', lw=2, color=plt.cm.viridis(color_idx_R[i_R]), label='R = %0.2f Mpc' % Rval)
plt.legend(fontsize=13)
plt.xlabel(r'log$_{10}M$', fontsize=18)
plt.ylabel(r'$f_m$', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)

plt.subplot(222)
for i_logM, logMval in enumerate(logM):
    want_i = np.where(fvfm_master_logM == logMval)[0]
    plt.plot(fvfm_master['R_Mpc'][want_i], fvfm_master['fm'][want_i], '-', lw=2, color=plt.cm.viridis(color_idx_logM[i_logM]), label=r'log$_{10}M$ = %0.2f' % logMval)
plt.legend(fontsize=13)
plt.xlabel('R (Mpc)', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)

plt.subplot(223)
for i_R, Rval in enumerate(R):
    want_i = np.where(fvfm_master_R == Rval)[0]
    plt.plot(fvfm_master['logM'][want_i], fvfm_master['fv'][want_i], '-', lw=2, color=plt.cm.viridis(color_idx_R[i_R]), label='R = %0.2f Mpc' % Rval)
plt.xlabel(r'log$_{10}M$', fontsize=18)
plt.ylabel(r'$f_v$', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)

plt.subplot(224)
for i_logM, logMval in enumerate(logM):
    want_i = np.where(fvfm_master_logM == logMval)[0]
    plt.plot(fvfm_master['R_Mpc'][want_i], fvfm_master['fv'][want_i], '-', lw=2, color=plt.cm.viridis(color_idx_logM[i_logM]), label=r'log$_{10}M$ = %0.2f' % logMval)
plt.xlabel('R (Mpc)', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)

plt.tight_layout()
#plt.show()
plt.savefig(outfig)