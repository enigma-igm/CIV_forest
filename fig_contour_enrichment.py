import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from matplotlib import pyplot as plt
import itertools
import numpy as np

seed = 1048272
seed = None
if seed == None:
    seed = np.random.randint(0, 10000000)

nlogM, nR, nlogZ = 251, 201, 161
modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'

logM_guess_ls = [9.5, 10.5]
R_guess_ls = [0.3, 1.5]
logZ_guess_ls = [-3.50]

contour_color = itertools.cycle(('black', 'green', 'red', 'magenta', 'blue', 'saddlebrown', 'darkorange', 'dodgerblue', 'purple', 'lightgreen', 'cornflowerblue'))

plt.figure(figsize=(14,5))
for ilogM, logM_guess in enumerate(logM_guess_ls):
    for iR, R_guess in enumerate(R_guess_ls):
        for ilogZ, logZ_guess in enumerate(logZ_guess_ls):

            print("==========", logM_guess, R_guess, logZ_guess, "==========")
            init_out = infen.init(modelfile, logM_guess, R_guess, logZ_guess, seed)
            logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out
            lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, True)

            ilogZ_fine = find_closest(logZ_fine, logZ_data)
            ilogM_fine = find_closest(logM_fine, logM_data)
            iR_fine = find_closest(R_fine, R_data)

            levels = [0.68, 0.95]
            c = next(contour_color)

            plt.subplot(131)
            inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine[ilogM_fine, :, :], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels, color=c)
            plt.subplot(132)
            inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine[:, iR_fine, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels, color=c)
            plt.subplot(133)
            inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine[:, :, ilogZ_fine], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels, color=c)

plt.tight_layout()
plt.show()

"""
more_contours1.png: 

seed = 1048272
nlogM, nR, nlogZ = 251, 251, 251
logM_guess_ls = [9.5, 10.5]
R_guess_ls = [0.25, 0.75]
logZ_guess_ls = [-3.5]

"""