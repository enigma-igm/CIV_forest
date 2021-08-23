import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
import corner

from scipy import optimize
from IPython import embed
from enigma.reion_forest.compute_model_grid import read_model_grid
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
import random
import time
import inference_enrichment as infen

############## Setting up ##############
modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'

seed = 5382029
#seed = 8808697
#seed=None

logM_guess, R_guess, logZ_guess = 9.89, 0.98, -3.57
#logM_guess, R_guess, logZ_guess = 9.09, 0.35, -3.3

nlogM_fine, nR_fine, nlogZ_fine = 251, 251, 251
nlogM_fine, nR_fine, nlogZ_fine = 251, 291, 251 # for finer grid, giving increment of 0.01 for each param

fixedpar = ['logM', 'R', 'logZ']
spline_order_x, spline_order_y = 2, 2

############## Reading in ##############
if seed == None:
    seed = np.random.randint(0, 10000000)
    print("Using random seed", seed)
else:
    print("Using random seed", seed)

rand = np.random.RandomState(seed)

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)

logZ_coarse = params['logZ'][0]
logM_coarse = params['logM'][0]
R_coarse = params['R_Mpc'][0]
R_coarse = np.round(R_coarse, 2) # need to force this to avoid floating point issue

vel_corr = params['vel_mid'].flatten()
vel_min = params['vmin_corr'][0]
vel_max = params['vmax_corr'][0]
nlogZ = params['nlogZ'][0]
nlogM = params['nlogM'][0]
nR = params['nR'][0]

# Pick the data that we will run with
nmock = xi_mock_array.shape[2]
imock = rand.choice(np.arange(nmock), size=1)
print('imock', imock)

# find the closest model values to guesses
ilogZ_mod = find_closest(logZ_coarse, logZ_guess)
ilogM_mod =  find_closest(logM_coarse, logM_guess)
iR_mod = find_closest(R_coarse, R_guess)
print('ilogM, iR, ilogZ', ilogM_mod, iR_mod, ilogZ_mod)

logZ_data = logZ_coarse[ilogZ_mod]
logM_data = logM_coarse[ilogM_mod]
R_data = R_coarse[iR_mod]
print('logM_data, R_data, logZ_data', logM_data, R_data, logZ_data)

xi_data = xi_mock_array[ilogM_mod, iR_mod, ilogZ_mod, imock, :].flatten()
xi_mask = np.ones_like(xi_data, dtype=bool)  # in case you want to mask any xi value, otherwise all True

############## Interpolating model grid ##############

# Interpolate the likelihood onto a fine grid to speed up the MCMC
nlogM = logM_coarse.size
logM_fine_min = logM_coarse.min()
logM_fine_max = logM_coarse.max()
dlogM_fine = (logM_fine_max - logM_fine_min) / (nlogM_fine - 1)
logM_fine = logM_fine_min + np.arange(nlogM_fine) * dlogM_fine

nR = R_coarse.size
R_fine_min = R_coarse.min()
R_fine_max = R_coarse.max()
dR_fine = (R_fine_max - R_fine_min) / (nR_fine - 1)
R_fine = R_fine_min + np.arange(nR_fine) * dR_fine

nlogZ = logZ_coarse.size
logZ_fine_min = logZ_coarse.min()
logZ_fine_max = logZ_coarse.max()
dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

print('dlogM_fine', dlogM_fine)
print('dR', dR_fine)
print('dlogZ_fine', dlogZ_fine)

############## Holding one param fixed ##############

fixed_lnlike_coarse = []
fixed_lnlike_fine = []
levels = [0.68, 0.95]

plt.figure(figsize=(12,4))

for ipar, par in enumerate(fixedpar):
    print('Now holding %s fixed' % par)
    if par == 'logM':
        lnlike_coarse = np.zeros((nR, nlogZ))
        for iR, R_val in enumerate(R_coarse):
            for ilogZ, logZ_val in enumerate(logZ_coarse):
                lnlike_coarse[iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM_mod, iR, ilogZ, :], \
                                                                 lndet_array[ilogM_mod, iR, ilogZ],
                                                                 icovar_array[ilogM_mod, iR, ilogZ, :, :])
        lnlike_fine = inference.interp_lnlike(R_fine, logZ_fine, R_coarse, logZ_coarse, lnlike_coarse, kx=spline_order_x, ky=spline_order_y)
        plt.subplot(131)
        inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine, R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)

    elif par == 'R':
        lnlike_coarse = np.zeros((nlogM, nlogZ))
        for ilogM, logM_val in enumerate(logM_coarse):
            for ilogZ, logZ_val in enumerate(logZ_coarse):
                lnlike_coarse[ilogM, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR_mod, ilogZ, :], \
                                                                    lndet_array[ilogM, iR_mod, ilogZ],
                                                                    icovar_array[ilogM, iR_mod, ilogZ, :, :])
        lnlike_fine = inference.interp_lnlike(logM_fine, logZ_fine, logM_coarse, logZ_coarse, lnlike_coarse, kx=spline_order_x, ky=spline_order_y)
        plt.subplot(132)
        inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine, logM_data, logZ_data, 'logM', 'logZ', levels=levels)

    elif par == 'logZ':
        lnlike_coarse = np.zeros((nlogM, nR))
        for ilogM, logM_val in enumerate(logM_coarse):
            for iR, R_val in enumerate(R_coarse):
                lnlike_coarse[ilogM, iR] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ_mod, :], \
                                                                 lndet_array[ilogM, iR, ilogZ_mod],
                                                                 icovar_array[ilogM, iR, ilogZ_mod, :, :])
        lnlike_fine = inference.interp_lnlike(logM_fine, R_fine, logM_coarse, R_coarse, lnlike_coarse, kx=spline_order_x, ky=spline_order_y)
        plt.subplot(133)
        inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine, logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)

    fixed_lnlike_coarse.append(lnlike_coarse)
    fixed_lnlike_fine.append(lnlike_fine)

plt.tight_layout()
plt.show()

"""
if fixedpar == 'logZ': # holding logZ fixed at the true value

    lnlike_coarse = np.zeros((nlogM, nR))
    for ilogM, logM_val in enumerate(logM_coarse):
        for iR, R_val in enumerate(R_coarse):
            lnlike_coarse[ilogM, iR] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ_mod, :], \
                                                             lndet_array[ilogM, iR, ilogZ_mod], icovar_array[ilogM, iR, ilogZ_mod, :, :])

    print('interpolating lnlike')
    lnlike_fine = inference.interp_lnlike(logM_fine, R_fine, logM_coarse, R_coarse, lnlike_coarse, kx=2, ky=2) # RectBivariateSpline

    levels = [0.68, 0.95]
    inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine, logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)
    plt.tight_layout()
    plt.show()

elif fixedpar == 'logM':

    lnlike_coarse = np.zeros((nR, nlogZ))
    for iR, R_val in enumerate(R_coarse):
        for ilogZ, logZ_val in enumerate(logZ_coarse):
            lnlike_coarse[iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM_mod, iR, ilogZ, :], \
                                                             lndet_array[ilogM_mod, iR, ilogZ],
                                                             icovar_array[ilogM_mod, iR, ilogZ, :, :])

    print('interpolating lnlike')
    lnlike_fine = inference.interp_lnlike(R_fine, logZ_fine, R_coarse, logZ_coarse, lnlike_coarse, kx=2, ky=2)  # RectBivariateSpline

    levels = [0.68, 0.95]
    inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine, R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
    plt.tight_layout()
    plt.show()

elif fixedpar == 'R':

    lnlike_coarse = np.zeros((nlogM, nlogZ))
    for ilogM, logM_val in enumerate(logM_coarse):
        for ilogZ, logZ_val in enumerate(logZ_coarse):
            lnlike_coarse[ilogM, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR_mod, ilogZ, :], \
                                                             lndet_array[ilogM, iR_mod, ilogZ],
                                                             icovar_array[ilogM, iR_mod, ilogZ, :, :])

    print('interpolating lnlike')
    lnlike_fine = inference.interp_lnlike(logM_fine, logZ_fine, logM_coarse, logZ_coarse, lnlike_coarse, kx=2, ky=2)  # RectBivariateSpline

    levels = [0.68, 0.95]
    inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine, logM_data, logZ_data, 'logM', 'logZ', levels=levels)
    plt.tight_layout()
    plt.show()
"""