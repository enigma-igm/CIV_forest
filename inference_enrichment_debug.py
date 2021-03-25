import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference

modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
logM_guess, R_guess, logZ_guess = 10.21, 0.35, -3.91
#logM_guess, R_guess, logZ_guess = 10.61, 0.65, -3.71
#logM_guess, R_guess, logZ_guess = 9.89, 0.98, -3.57

seed = 5382029
seed = 4597271
seed = None
nlogM, nR, nlogZ = 251, 201, 161

########
init_out = infen.init(modelfile, logM_guess, R_guess, logZ_guess, seed)
logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out

lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, True)

ilogZ_fine = find_closest(logZ_fine, logZ_data)
ilogM_fine = find_closest(logM_fine, logM_data)
iR_fine = find_closest(R_fine, R_data)
print('ilogM_fine, iR_fine, ilogZ_fine', ilogM_fine, iR_fine, ilogZ_fine)

infen.plot_likelihood_data(lnlike_fine, logM_fine, R_fine, logZ_fine, logM_data, R_data, logZ_data)

inference.lnlike_plot_3d(logM_fine, R_fine, logZ_fine, lnlike_fine, ilogM_fine, iR_fine, ilogZ_fine)