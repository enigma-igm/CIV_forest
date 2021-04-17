from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import halos_skewers
import matplotlib.style as style
style.use('tableau-colorblind10')

outfig = 'paper_plots/logZeff.pdf'

fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
fvfm_master_logM = np.round(fvfm_master['logM'], 2)
fvfm_master_R = np.round(fvfm_master['R_Mpc'], 2)

logZ_fid = -3.5
plot_vs_R = False # if False, then plot vs logM (at fixed R)

# looking at just a subset of all models
logM = np.array([8.50, 9.00, 9.50, 10.00, 10.50, 11.00])
R = np.array([0.1, 0.50, 1.00, 1.50, 2.00, 3.00])

# all models
#logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

plt.figure(figsize=(13.5,6.5))
plt.subplot(121)
for i_R, Rval in enumerate(R):
    want_iR = np.where(fvfm_master_R == Rval)[0]
    logZ_eff = halos_skewers.calc_igm_Zeff(fvfm_master['fm'][want_iR], logZ_fid=logZ_fid)
    plt.plot(fvfm_master['logM'][want_iR], logZ_eff, 'o-', ms=3, label='R = %0.2f Mpc' % Rval)

plt.xlabel('logM', fontsize=18)
plt.ylabel(r'logZ$_{eff}$', fontsize=18)
plt.legend(fontsize=13, ncol=2)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)

plt.subplot(122)
for i_logM, logMval in enumerate(logM):
    want_ilogM = np.where(fvfm_master_logM == logMval)[0]
    logZ_eff = halos_skewers.calc_igm_Zeff(fvfm_master['fm'][want_ilogM], logZ_fid=logZ_fid)
    plt.plot(fvfm_master['R_Mpc'][want_ilogM], logZ_eff, 'o-', ms=3, label='logM = %0.2f' % logMval)

plt.xlabel('R (Mpc)', fontsize=15)
plt.legend(fontsize=12)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)

plt.tight_layout()
plt.savefig(outfig)

"""
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
