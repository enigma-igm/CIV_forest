import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from enigma.reion_forest import utils as reion_utils
from matplotlib import pyplot as plt
import numpy as np
import os

#modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
savefig_path = 'plots/enrichment/inference_enrichment_debug/'

logM_guess, R_guess, logZ_guess = 9.68, 0.68, -3.48 # fiducial model for paper
#logM_guess, R_guess, logZ_guess = 9.5, 0.5, -3.5
#logM_guess, R_guess, logZ_guess = 9.5, 1.5, -3.5
#logM_guess, R_guess, logZ_guess = 9.5, 2.5, -3.5

#logM_guess, R_guess, logZ_guess = 10.5, 0.5, -3.5
#logM_guess, R_guess, logZ_guess = 10.5, 1.5, -3.5
#logM_guess, R_guess, logZ_guess = 10.5, 2.5, -3.5
#logM_guess, R_guess, logZ_guess = 10.5, 2.7, -3.5
logM_guess, R_guess, logZ_guess = 10.89, 0.2, -4.4 # upper limit case (truncated as limited by spline interpolation)

want_savefig = False

#seed = 5382029 # note: nmock=26 in this case (caught this bug too late)
#seed = 4355455 # nmock=26
seed = 5377192 # 977242, 9194375

#nlogM, nR, nlogZ = 251, 201, 161
nlogM, nR, nlogZ = 251, 291, 251 # for finer grid

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

lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, interp_lnlike=True)

"""
# hack for calling lnlike_fine produced using ARBInterp.py
lnlike_fine = np.load(savefig_path + 'seed_%d_%0.2f_%0.2f_%0.2f/' % (seed, logM_guess, R_guess, logZ_guess) + \
                      'finer_grid/lnlike_fine_arbinterp.npy')
logM_fine = logM_fine[10:240]
R_fine = R_fine[10:280]
logZ_fine = logZ_fine[10:240]
"""

ilogZ_coarse = find_closest(logZ_coarse, logZ_data)
ilogM_coarse = find_closest(logM_coarse, logM_data)
iR_coarse = find_closest(R_coarse, R_data)
print('ilogM_coarse, iR_coarse, ilogZ_coarse', ilogM_coarse, iR_coarse, ilogZ_coarse)

ilogZ_fine = find_closest(logZ_fine, logZ_data)
ilogM_fine = find_closest(logM_fine, logM_data)
iR_fine = find_closest(R_fine, R_data)
print('ilogM_fine, iR_fine, ilogZ_fine', ilogM_fine, iR_fine, ilogZ_fine)

# plot 1D likelihood at true mock data values
infen.plot_likelihood_data(lnlike_fine, logM_fine, R_fine, logZ_fine, logM_data, R_data, logZ_data, savefig=savefig_lnlike1d)

# plot surface plots at true values
#plotprob=True
#inference.lnlike_plot_3d(logM_fine, R_fine, logZ_fine, lnlike_fine, ilogM_fine, iR_fine, ilogZ_fine, savefig=savefig_lnlike_surface, plotprob=plotprob)

# plot contour plots
levels = [0.68, 0.95] # None
plt.figure(figsize=(12,4))
plt.subplot(131)
inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine[ilogM_fine,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
plt.subplot(132)
inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine[:, iR_fine, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
plt.subplot(133)
inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine[:,:,ilogZ_fine], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)

"""
plt.subplot(234)
inference.plot_contour_lnlike(R_coarse, logZ_coarse, lnlike_coarse[ilogM_coarse,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
plt.subplot(235)
inference.plot_contour_lnlike(logM_coarse, logZ_coarse, lnlike_coarse[:, iR_coarse, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
plt.subplot(236)
inference.plot_contour_lnlike(logM_coarse, R_coarse, lnlike_coarse[:,:,ilogZ_coarse], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)
"""
plt.tight_layout()
if savefig_lnlike_contour != None:
    plt.savefig(savefig_lnlike_contour)
else:
    plt.show()