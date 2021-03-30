import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from matplotlib import pyplot as plt
import os

modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
savefig_path = 'plots/enrichment/inference_enrichment_debug/'

logM_guess, R_guess, logZ_guess = 10.21, 0.35, -3.91
logM_guess, R_guess, logZ_guess = 10.61, 0.65, -3.71
logM_guess, R_guess, logZ_guess = 10.6, 1.09, -3.50
want_savefig = True

#seed = 5382029
#seed = 4597271
seed = 8365108 # 5822201
nlogM, nR, nlogZ = 251, 201, 161

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
logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out

lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, True)

ilogZ_fine = find_closest(logZ_fine, logZ_data)
ilogM_fine = find_closest(logM_fine, logM_data)
iR_fine = find_closest(R_fine, R_data)
print('ilogM_fine, iR_fine, ilogZ_fine', ilogM_fine, iR_fine, ilogZ_fine)

infen.plot_likelihood_data(lnlike_fine, logM_fine, R_fine, logZ_fine, logM_data, R_data, logZ_data, savefig=savefig_lnlike1d)

inference.lnlike_plot_3d(logM_fine, R_fine, logZ_fine, lnlike_fine, ilogM_fine, iR_fine, ilogZ_fine, savefig=savefig_lnlike_surface)

levels = [0.68, 0.95] # None

plt.figure(figsize=(12,4))
plt.subplot(131)
inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine[ilogM_fine,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
plt.subplot(132)
inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine[:, iR_fine, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
plt.subplot(133)
inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine[:,:,ilogZ_fine], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)

plt.tight_layout()

if savefig_lnlike_contour != None:
    plt.savefig(savefig_lnlike_contour)
else:
    plt.show()

"""
1. logM_guess, R_guess, logZ_guess = 9.0, 1.00, -3.50
seed = 3116325

2. logM_guess, R_guess, logZ_guess = 9.9, 1.09, -3.50
seed = 8405705

(plotted)
3. logM_guess, R_guess, logZ_guess = 10.6, 1.09, -3.50
seed = 8365108

(plotted)
4. logM_guess, R_guess, logZ_guess = 9.9, 0.45, -3.50
seed = 988019


"""