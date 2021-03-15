from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import halos_skewers

fvfm_all = Table.read('nyx_sim_data/igm_cluster/fvfm_all.fits')
logZ_fid = -3.5
plot_vs_R = False # if False, then plot vs logM (at fixed R)

# looking at just a subset of all models
logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.50, 0.1, 3, 0.4)
R = R[:-1] # the last element is outside model boundaries

# all models
logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

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
