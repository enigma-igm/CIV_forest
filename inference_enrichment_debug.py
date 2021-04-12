import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from enigma.reion_forest import utils as reion_utils
from matplotlib import pyplot as plt
import numpy as np
import os

modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
savefig_path = 'plots/enrichment/inference_enrichment_debug/'

logM_guess, R_guess, logZ_guess = 10.21, 0.35, -3.91
#logM_guess, R_guess, logZ_guess = 10.61, 0.65, -3.71
logM_guess, R_guess, logZ_guess = 9.89, 0.98, -3.57 # fiducial
logM_guess, R_guess, logZ_guess = 10.50, 0.90, -3.57
want_savefig = False

seed = 5382029
nlogM, nR, nlogZ = 251, 201, 161
#nlogM, nR, nlogZ = 251, 251, 251
#nlogM, nR, nlogZ = 451, 451, 451

if want_savefig:
    newpath = savefig_path + 'seed_%d_%0.2f_%0.2f_%0.2f/' % (seed, logM_guess, R_guess, logZ_guess)
    os.mkdir(newpath)
    savefig_lnlike1d = newpath + 'lnlike_1d.png'
    savefig_lnlike_surface = newpath + 'lnlike_surface.png'
    savefig_lnlike_contour = newpath + 'lnlike_contour.png'
else:
    savefig_lnlike1d = None
    savefig_lnlike_surface = None
    savefig_lnlike_contour = None

########
init_out = infen.init(modelfile, logM_guess, R_guess, logZ_guess, seed)

logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out

lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, True)

ilogZ_coarse = find_closest(logZ_coarse, logZ_data)
ilogM_coarse = find_closest(logM_coarse, logM_data)
iR_coarse = find_closest(R_coarse, R_data)
print('ilogM_coarse, iR_coarse, ilogZ_coarse', ilogM_coarse, iR_coarse, ilogZ_coarse)

ilogZ_fine = find_closest(logZ_fine, logZ_data)
ilogM_fine = find_closest(logM_fine, logM_data)
iR_fine = find_closest(R_fine, R_data)
print('ilogM_fine, iR_fine, ilogZ_fine', ilogM_fine, iR_fine, ilogZ_fine)

######## Plot 2PCF for mock and mean model
covar_mean = inference.covar_model_3d([logM_guess, R_guess, logZ_guess], logM_coarse, R_coarse, logZ_coarse, covar_array)
xi_err = np.sqrt(np.diag(covar_mean))
xi_model_mean = xi_model_array[ilogM_coarse, iR_coarse, ilogZ_coarse]

plt.figure(figsize=(14,5))
plt.errorbar(vel_corr, xi_data, yerr=xi_err, marker='o', ms=6, alpha=0.6, color='black', ecolor='black', capthick=2, capsize=4, mec='none', ls='none', label='mock data', zorder=20)
plt.plot(vel_corr, xi_model_mean, linewidth=2.0, linestyle='-', color='red', label='mean model')
vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
plt.axvline(vel_doublet.value, color='b', linestyle='--', linewidth=2)
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=18)
plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
plt.title(r'logZ $=%0.2f$, logM $=%0.2f$, R $=%0.2f$' % (logZ_data, logM_data, R_data), fontsize=18)
plt.tight_layout()
plt.show()

exit()
# plot 1D likelihood at true mock data values
infen.plot_likelihood_data(lnlike_fine, logM_fine, R_fine, logZ_fine, logM_data, R_data, logZ_data, savefig=savefig_lnlike1d)
# plot surface plots at true values
inference.lnlike_plot_3d(logM_fine, R_fine, logZ_fine, lnlike_fine, ilogM_fine, iR_fine, ilogZ_fine, savefig=savefig_lnlike_surface)

# plot contour plots
levels = [0.68, 0.95] # None
plt.figure(figsize=(12,6))
plt.subplot(231)
inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine[ilogM_fine,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
plt.subplot(232)
inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine[:, iR_fine, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
plt.subplot(233)
inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine[:,:,ilogZ_fine], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)

plt.subplot(234)
inference.plot_contour_lnlike(R_coarse, logZ_coarse, lnlike_coarse[ilogM_coarse,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
plt.subplot(235)
inference.plot_contour_lnlike(logM_coarse, logZ_coarse, lnlike_coarse[:, iR_coarse, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
plt.subplot(236)
inference.plot_contour_lnlike(logM_coarse, R_coarse, lnlike_coarse[:,:,ilogZ_coarse], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)

plt.tight_layout()

if savefig_lnlike_contour != None:
    plt.savefig(savefig_lnlike_contour)
else:
    plt.show()