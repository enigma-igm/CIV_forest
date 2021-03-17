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

######## Setting up #########
seed = 125913 # random seed to pick the mock data set
seed = None

if seed == None:
    seed = np.random.randint(0, 10000000)
    print("Using random seed", seed)

rand = np.random.RandomState(seed)
random.seed(seed)

def init(modelfile, logZ_guess, logM_guess, R_guess):

    # Read in the model grid
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

    linearZprior = False

    # find the closest model values to guesses
    ilogZ = find_closest(logZ_coarse, logZ_guess)
    ilogM =  find_closest(logM_coarse, logM_guess)
    iR = find_closest(R_coarse, R_guess)

    logZ_data = logZ_coarse[ilogZ]
    logM_data = logM_coarse[ilogM]
    R_data = R_coarse[iR]

    xi_data = xi_mock_array[ilogM, iR, ilogZ, imock, :].flatten()
    xi_mask = np.ones_like(xi_data, dtype=bool)  # in case you want to mask any xi value, otherwise all True

    init_out = logZ_coarse, logM_coarse, R_coarse, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array
    return init_out

def interp_likelihood(init_out, nlogZ_fine=501, nlogM_fine=501, nR_fine=501, plot=False):

    # unpack input
    logZ_coarse, logM_coarse, R_coarse, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

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

    print('dlogZ_fine', dlogZ_fine)
    print('dlogM_fine', dlogM_fine)
    print('dR', dR_fine)

    # Loop over the coarse grid and evaluate the likelihood at each location
    lnlike_coarse = np.zeros((nlogM, nR, nlogZ))
    for ilogM, logM_val in enumerate(logM_coarse):
        for iR, R_val in enumerate(R_coarse):
            for ilogZ, logZ_val in enumerate(logZ_coarse):
                lnlike_coarse[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ, :],
                                                            lndet_array[ilogM, iR, ilogZ],
                                                            icovar_array[ilogM, iR, ilogZ, :, :])

    print('interpolating lnlike')
    lnlike_fine = inference.interp_lnlike_3d(logM_fine, R_fine, logZ_fine, logM_coarse, R_coarse, logZ_coarse, lnlike_coarse)

    start = time.time()
    print('interpolating model')
    xi_model_fine = inference.interp_model_3d(logM_fine, R_fine, logZ_fine, logM_coarse, R_coarse, logZ_coarse, xi_model_array)
    end = time.time()
    print((end-start)/60.)

    if plot: # make a 2d surface plot of the likelihood
        pass
        # TODO
        #logZ_fine_2d, xhi_fine_2d = np.meshgrid(logZ_fine, xhi_fine)
        #lnlikefile = figpath + 'lnlike.pdf'
        #inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    return lnlike_fine, xi_model_fine, logM_fine, R_fine, logZ_fine

def mcmc_inference(nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior):

    # find optimal starting points for each walker
    logM_fine_min, logM_fine_max = logM_fine.min(), logM_fine.max()
    R_fine_min, R_fine_max = R_fine.min(), R_fine.max()
    logZ_fine_min, logZ_fine_max = logZ_fine.min(), logZ_fine.max()

    # DOUBLE CHECK
    bounds = [(logM_fine_min, logM_fine_max), (R_fine_min, R_fine_max), (logZ_fine_min, logZ_fine_max)] if not linear_prior else \
        [(0, 10**logM_fine_max), (0, R_fine_max), (0, 10**logZ_fine_max)]

    chi2_func = lambda *args: -2 * inference.lnprob_3d(*args)
    args = (lnlike_fine, logM_fine, R_fine, logZ_fine, linear_prior)

    result_opt = optimize.differential_evolution(chi2_func, bounds=bounds, popsize=25, recombination=0.7, disp=True, polish=True, args=args, seed=rand)
    ndim = 3

    # initialize walkers
    pos = [[np.clip(result_opt.x[i] + 1e-2 * (bounds[i][1] - bounds[i][0]) * rand.randn(1)[0], bounds[i][0], bounds[i][1])
         for i in range(ndim)] for i in range(nwalkers)]

    np.random.seed(rand.randint(0, seed, size=1)[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, inference.lnprob_3d, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    tau = sampler.get_autocorr_time()

    print('Autocorrelation time')
    print('tau_logM = {:7.2f}, tau_R = {:7.2f}, tau_logZ = {:7.2f}'.format(tau[0], tau[1], tau[2]))

    flat_samples = sampler.get_chain(discard=burnin, thin=250, flat=True) # numpy array

    if linear_prior: # convert the samples to linear units
        param_samples = flat_samples.copy()
        param_samples[:, 0] = np.log10(param_samples[:, 0]) # logM
        param_samples[:, 2] = np.log10(param_samples[:, 2]) # logZ
    else:
        param_samples = flat_samples

    return flat_samples