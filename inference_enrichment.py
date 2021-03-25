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

### debugging config (3.25.2021)
#modelfile = 'nyx_sim_data/igm_cluster/enrichment/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
#logM_guess, R_guess, logZ_guess = 9.89, 0.98, -3.57
#seed = 5382029
# nlogM, nR, nlogZ = 251, 201, 161

def init(modelfile, logM_guess, R_guess, logZ_guess, seed=None):

    if seed == None:
        seed = np.random.randint(0, 10000000)
        print("Using random seed", seed)
    else:
        print("Using random seed", seed)

    rand = np.random.RandomState(seed)

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
    print('imock', imock)

    linearZprior = False

    # find the closest model values to guesses
    ilogZ = find_closest(logZ_coarse, logZ_guess)
    ilogM =  find_closest(logM_coarse, logM_guess)
    iR = find_closest(R_coarse, R_guess)
    print('ilogM, iR, ilogZ', ilogM, iR, ilogZ)

    logZ_data = logZ_coarse[ilogZ]
    logM_data = logM_coarse[ilogM]
    R_data = R_coarse[iR]
    print('logM_data, R_data, logZ_data', logM_data, R_data, logZ_data)

    xi_data = xi_mock_array[ilogM, iR, ilogZ, imock, :].flatten()
    xi_mask = np.ones_like(xi_data, dtype=bool)  # in case you want to mask any xi value, otherwise all True

    init_out = logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array

    return init_out

def interp_likelihood(init_out, nlogM_fine, nR_fine, nlogZ_fine, interp_lnlike=False, interp_ximodel=False):

    # ~10 sec to interpolate 3d likelihood for nlogM_fine, nR_fine, nlogZ_fine = 251, 201, 161
    # dlogM_fine 0.01
    # dR 0.015
    # dlogZ_fine 0.015625

    # unpack input
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out

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

    # Loop over the coarse grid and evaluate the likelihood at each location for the chosen mock data
    # Needs to be repeated for each chosen mock data
    if interp_lnlike:
        lnlike_coarse = np.zeros((nlogM, nR, nlogZ))
        for ilogM, logM_val in enumerate(logM_coarse):
            for iR, R_val in enumerate(R_coarse):
                for ilogZ, logZ_val in enumerate(logZ_coarse):
                    lnlike_coarse[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ, :],
                                                                lndet_array[ilogM, iR, ilogZ],
                                                                icovar_array[ilogM, iR, ilogZ, :, :])

        print('interpolating lnlike')
        start = time.time()
        lnlike_fine = inference.interp_lnlike_3d(logM_fine, R_fine, logZ_fine, logM_coarse, R_coarse, logZ_coarse, lnlike_coarse)
        end = time.time()
        print((end - start) / 60.)
    else:
        lnlike_fine = None

    # Only needs to be done once, unless the fine grid is change
    if interp_ximodel:
        start = time.time()
        print('interpolating model')
        xi_model_fine = inference.interp_model_3d(logM_fine, R_fine, logZ_fine, logM_coarse, R_coarse, logZ_coarse, xi_model_array)
        end = time.time()
        print((end-start)/60.)
    else:
        xi_model_fine = None

    return lnlike_coarse, lnlike_fine, xi_model_fine, logM_fine, R_fine, logZ_fine

def plot_marginal_likelihood(xparam, yparam, lnlike_fine, summing_axis, xparam_label, yparam_label):
    # double check
    # plot the normalized likelihood?

    #lnlike_fine_new = lnlike_fine - lnlike_fine.max()
    #lnlike_norm = integrate.trapz(np.exp(lnlike_fine_new), logZ_fine)  # summing L
    #plogZ = np.exp(lnlike_fine_new) / lnlike_norm

    xparam_2d, yparam_2d = np.meshgrid(xparam, yparam, indexing='ij')
    lnlike_2d = np.sum(lnlike_fine, axis=summing_axis)
    inference.lnlike_plot_general(xparam_2d, yparam_2d, xparam_label, yparam_label, lnlike_2d)

def plot_single_likelihood(lnlike_3d, grid_arr, param_name, ind_par1, ind_par2):

    nlogM, nR, nlogZ = np.shape(lnlike_3d)

    if param_name == 'logM':
        plt.plot(grid_arr, lnlike_3d[:, ind_par1, ind_par2])
    elif param_name == 'R_Mpc':
        plt.plot(grid_arr, lnlike_3d[ind_par1, :, ind_par2])
    elif param_name == 'logZ':
        plt.plot(grid_arr, lnlike_3d[ind_par1, ind_par2, :])
    plt.ylabel('lnL', fontsize=12)
    plt.xlabel(param_name, fontsize=12)

def plot_likelihoods(lnlike_fine, logM_fine, R_fine, logZ_fine):
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    for i in range(0, 150, 50):
        for j in range(0, 150, 50):
            plot_single_likelihood(lnlike_fine, logM_fine, 'logM', i, j)

    plt.subplot(132)
    for i in range(0, 150, 50):
        for j in range(0, 150, 50):
            plot_single_likelihood(lnlike_fine, R_fine, 'R_Mpc', i, j)

    plt.subplot(133)
    for i in range(0, 150, 50):
        for j in range(0, 150, 50):
            plot_single_likelihood(lnlike_fine, logZ_fine, 'logZ', i, j)

    plt.tight_layout()
    plt.show()

def plot_likelihood_data(lnlike, logM_grid, R_grid, logZ_grid, logM_data, R_data, logZ_data):

    ilogM = find_closest(logM_grid, logM_data)
    iR = find_closest(R_grid, R_data)
    ilogZ = find_closest(logZ_grid, logZ_data)
    print(ilogM, iR, ilogZ)

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.plot(logM_grid, lnlike[:, iR, ilogZ])
    plt.axvline(logM_data, ls='--', c='k', label='logM_data=% 0.2f' % logM_data)
    plt.legend()
    plt.xlabel('logM', fontsize = 13)
    plt.ylabel('lnL', fontsize=13)
    plt.ylim([2400, 2800])

    plt.subplot(132)
    plt.plot(R_grid, lnlike[ilogM, :, ilogZ])
    plt.axvline(R_data, ls='--', c='k', label = 'R_data=%0.2f' % R_data)
    plt.legend()
    plt.xlabel('R (Mpc)', fontsize=13)
    plt.ylim([2400, 2800])

    plt.subplot(133)
    plt.plot(logZ_grid, lnlike[ilogM, iR])
    plt.axvline(logZ_data, ls='--', c='k', label = 'logZ_data=%0.2f' % logZ_data)
    plt.legend()
    plt.xlabel('logZ', fontsize=13)
    plt.ylim([2400, 2800])

    plt.tight_layout()
    plt.show()

def mcmc_inference(nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior, ball_size=0.01, seed=None):

    if seed == None:
        seed = np.random.randint(0, 10000000)
        print("Using random seed", seed)
    else:
        print("Using random seed", seed)

    rand = np.random.RandomState(seed)

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
    # for my own understanding #
    pos = []
    for i in range(nwalkers):
        tmp = []
        for j in range(ndim):
            perturb_pos = result_opt.x[j] + (ball_size * (bounds[j][1] - bounds[j][0]) * rand.randn(1)[0])
            tmp.append(np.clip(perturb_pos, bounds[j][0], bounds[j][1]))
        pos.append(tmp)
    embed()
    #pos = [[np.clip(result_opt.x[i] + 1e-2 * (bounds[i][1] - bounds[i][0]) * rand.randn(1)[0], bounds[i][0], bounds[i][1])
    #     for i in range(ndim)] for i in range(nwalkers)]

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

    return sampler, param_samples, bounds

def plot_mcmc(sampler, param_samples, init_out, linear_prior):

    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, lndet_array, icovar_array = init_out

    # (1) Make the walker plot, use the true values in the chain
    var_label = ['logM', 'R_Mpc', 'logZ']
    truths = [10**(logM_data), R_data, 10**(logZ_data)] if linear_prior else [logM_data, R_data, logZ_data]
    print("truths", truths)
    chain = sampler.get_chain()

    #walker_outfig = 'walker_ballsize_0.1_nwalker20.pdf'
    #inference.walker_plot(chain, truths, var_label, walker_outfig)

    # (2) Make the corner plot, again use the true values in the chain
    fig = corner.corner(param_samples, labels=var_label, truths=truths, levels=(0.68,), color='k', \
                        truth_color='darkgreen', \
                        show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20}, \
                        data_kwargs={'ms': 1.0, 'alpha': 0.1})

    #cornerfile = figpath + 'corner_plot.pdf'
    for ax in fig.get_axes():
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.tick_params(labelsize=12)
    plt.show()
    plt.close()

    # (3) Make the corrfunc plot
    # Need to modify inference.corrfunc_plot()

    """
    lower = np.array([bounds[0][0], bounds[1][0]])
    upper = np.array([bounds[0][1], bounds[1][1]])
    param_limits = [lower, upper]
    
    corrfile = figpath + 'corr_func_data.pdf'
    inference.corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse,
                            covar_array,
                            xhi_data, logZ_data, corrfile, rand=rand)
    # Lower limit on metallicity for pristine case
    if linearZprior:
        ixhi_prior = flat_samples[:, 0] > 0.95
        logZ_95 = np.percentile(param_samples[ixhi_prior, 1], 95.0)
        print('Obtained 95% upper limit of {:6.4f}'.format(logZ_95))
    """