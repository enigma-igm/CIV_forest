'''
Functions here:
    - init
    - interp_likelihood
    - mcmc_inference
    - plot_mcmc
    - do_arbinterp
    - do_all
    - interp_likelihood_fixedlogZ
    - plot_marginal_likelihood
    - plot_single_likelihood
    - plot_likelihoods
    - plot_likelihood_data
    - prep_for_arbinterp
    - prep_for_arbinterp2
    - plot_corner_nonthinned
'''

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
import halos_skewers
import time
from astropy.io import fits

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
    if seed == 5382029 or seed == 4355455: # there is a bug for getting nmock for debug seeds
        print("Using debug seed", seed)
        nmock = 26
    else:
        nmock = xi_mock_array.shape[3]

    imock = rand.choice(np.arange(nmock), size=1)
    print('imock', imock)

    #linearZprior = False

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

    init_out = logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
               covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess

    return init_out

def interp_likelihood(init_out, nlogM_fine, nR_fine, nlogZ_fine, interp_lnlike=False, interp_ximodel=False):

    # ~10 sec to interpolate 3d likelihood for nlogM_fine, nR_fine, nlogZ_fine = 251, 201, 161
    # dlogM_fine 0.01
    # dR 0.015
    # dlogZ_fine 0.015625

    # unpack input
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

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
    lnlike_coarse = np.zeros((nlogM, nR, nlogZ))
    for ilogM, logM_val in enumerate(logM_coarse):
        for iR, R_val in enumerate(R_coarse):
            for ilogZ, logZ_val in enumerate(logZ_coarse):
                lnlike_coarse[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask,
                                                                        xi_model_array[ilogM, iR, ilogZ, :],
                                                                        lndet_array[ilogM, iR, ilogZ],
                                                                        icovar_array[ilogM, iR, ilogZ, :, :])
    if interp_lnlike:
        #lnlike_coarse = np.zeros((nlogM, nR, nlogZ))
        #for ilogM, logM_val in enumerate(logM_coarse):
        #    for iR, R_val in enumerate(R_coarse):
        #        for ilogZ, logZ_val in enumerate(logZ_coarse):
        #            lnlike_coarse[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ, :],
        #                                                        lndet_array[ilogM, iR, ilogZ],
        #                                                        icovar_array[ilogM, iR, ilogZ, :, :])
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


def mcmc_inference(nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior, ball_size=0.01, \
                   seed=None, savefits_chain=None):

    if seed == None:
        seed = np.random.randint(0, 10000000)
        print("Using random seed", seed)
    else:
        print("Using random seed", seed)

    print("Using ball size", ball_size)

    rand = np.random.RandomState(seed)

    # find optimal starting points for each walker
    logM_fine_min, logM_fine_max = logM_fine.min(), logM_fine.max()
    R_fine_min, R_fine_max = R_fine.min(), R_fine.max()
    logZ_fine_min, logZ_fine_max = logZ_fine.min(), logZ_fine.max()

    # DOUBLE CHECK
    #bounds = [(logM_fine_min, logM_fine_max), (R_fine_min, R_fine_max), (logZ_fine_min, logZ_fine_max)] if not linear_prior else \
    #    [(0, 10**logM_fine_max), (0, R_fine_max), (0, 10**logZ_fine_max)]

    # (8/16/21) linear_prior only on logZ
    bounds = [(logM_fine_min, logM_fine_max), (R_fine_min, R_fine_max), (logZ_fine_min, logZ_fine_max)] if not linear_prior else \
        [(logM_fine_min, logM_fine_max), (R_fine_min, R_fine_max), (0, 10**logZ_fine_max)]

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
        #param_samples[:, 0] = np.log10(param_samples[:, 0]) # logM  # (8/16/21) linear_prior only on logZ
        param_samples[:, 2] = np.log10(param_samples[:, 2]) # logZ
    else:
        param_samples = flat_samples

    if savefits_chain != None:
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(), name='all_chain'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(flat=True), name='all_chain_flat'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(discard=burnin, flat=True), name='all_chain_discard_burnin'))
        hdulist.append(fits.ImageHDU(data=param_samples, name='param_samples'))
        hdulist.writeto(savefits_chain, overwrite=True)

    return sampler, param_samples, bounds

def plot_mcmc(sampler, param_samples, init_out, params, logM_fine, R_fine, logZ_fine, xi_model_fine, linear_prior, seed=None):
    # seed here used to choose random nrand(=50) mcmc realizations to plot on the 2PCF measurement

    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

    ##### (1) Make the walker plot, use the true values in the chain
    var_label = ['log(M)', 'R', '[C/H]']
    #truths = [10**(logM_data), R_data, 10**(logZ_data)] if linear_prior else [logM_data, R_data, logZ_data]
    truths = [logM_data, R_data, 10**(logZ_data)] if linear_prior else [logM_data, R_data, logZ_data] # (8/16/21) linear_prior only on logZ
    print("truths", truths)
    chain = sampler.get_chain()
    inference.walker_plot(chain, truths, var_label, walkerfile=None)

    ##### (2) Make the corner plot, again use the true values in the chain
    fig = corner.corner(param_samples, labels=var_label, truths=truths, levels=(0.68, ), color='k', \
                        truth_color='darkgreen', \
                        show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20}, \
                        data_kwargs={'ms': 1.0, 'alpha': 0.1})
    for ax in fig.get_axes():
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.tick_params(labelsize=12)

    plt.show()
    plt.close()

    ##### (3) Make the corrfunc plot with mcmc realizations
    fv, fm = halos_skewers.get_fvfm(np.round(logM_data,2), np.round(R_data,2))
    logZ_eff = halos_skewers.calc_igm_Zeff(fm, logZ_fid=logZ_data)
    print("logZ_eff", logZ_eff)
    inference.corrfunc_plot_3d(xi_data, param_samples, params, logM_fine, R_fine, logZ_fine, xi_model_fine, logM_coarse, R_coarse,
                     logZ_coarse, covar_array, logM_data, R_data, logZ_data, logZ_eff, nrand=50, seed=seed)

################ run all the driver functions leading to mcmc ################
import configparser

def do_arbinterp(logM_coarse, R_coarse, logZ_coarse, lnlike_coarse, coarse_outcsv, \
                 logM_fine, R_fine, logZ_fine, want_fine_outcsv, arbinterp_outnpy):

    prep_for_arbinterp(logM_coarse, R_coarse, logZ_coarse, lnlike_coarse, coarse_outcsv)
    trunc_nlogM, trunc_nR, trunc_nlogZ = prep_for_arbinterp2(logM_coarse, R_coarse, logZ_coarse, logM_fine, R_fine, logZ_fine, want_fine_outcsv)

    from ARBTools.ARBInterp import tricubic

    start = time.time()
    field = np.genfromtxt(coarse_outcsv, delimiter=',')
    Run = tricubic(field)
    allpts = np.genfromtxt(want_fine_outcsv, delimiter=',')

    print("########## starting interpolation ########## ")
    out_norm, out_grad = Run.Query(allpts)  # ~10 min compute time
    out_norm2 = out_norm[:, 0]
    out_norm2 = np.reshape(out_norm2, (trunc_nlogM, trunc_nR, trunc_nlogZ))

    np.save(arbinterp_outnpy, out_norm2)
    end = time.time()
    print((end - start) / 60.)

def do_all(config_file, run_mcmc=True):
    """
    modelfile_path = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/'
    modelfile = modelfile_path + 'fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'


    #seed = 4355455  # using the incorrect nmock=26
    #logM_guess, R_guess, logZ_guess = 9.12, 0.45, -3.50

    seed = 5382029
    logM_guess, R_guess, logZ_guess = 9.89, 0.98, -3.57

    nlogM, nR, nlogZ = 251, 291, 251
    interp_lnlike = False  # If False, need to provide filename containing pre-interpolated lnlikelihood;
                           # if True, then interpolate here
    interp_ximodel = False  # Same
    nsteps = 150000
    burnin = 1000
    nwalkers = 30
    linear_prior = False
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    modelfile = config['DEFAULT']['modelfile']
    seed = int(config['DEFAULT']['seed'])
    logM_guess, R_guess, logZ_guess = float(config['DEFAULT']['logm_guess']), float(config['DEFAULT']['r_guess']), float(config['DEFAULT']['logz_guess'])
    nlogM, nR, nlogZ = int(config['DEFAULT']['nlogm']), int(config['DEFAULT']['nr']), int(config['DEFAULT']['nlogz'])
    interp_lnlike = config['DEFAULT']['interp_lnlike']
    interp_ximodel = config['DEFAULT']['interp_ximodel']
    lnlike_file_name = config['DEFAULT']['lnlike_file_name']
    ximodel_file_name = config['DEFAULT']['ximodel_file_name']
    nsteps = int(config['DEFAULT']['nsteps'])
    burnin = int(config['DEFAULT']['burnin'])
    nwalkers = int(config['DEFAULT']['nwalkers'])
    linear_prior = config['DEFAULT']['linear_prior']
    savefits_chain = config['DEFAULT']['savefits_chain']
    #savefits_chain = None
    ball_size = float(config['DEFAULT']['ball_size'])

    # convert from string to bool
    interp_lnlike = False if interp_lnlike == 'False' else True
    interp_ximodel = False if interp_ximodel == 'False' else True
    linear_prior = False if linear_prior == 'False' else True
    print("interp_lnlike, interp_ximodel, linear_prior", interp_lnlike, interp_ximodel, linear_prior)

    init_out = init(modelfile, logM_guess, R_guess, logZ_guess, seed)

    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out

    lnlike_coarse, lnlike_fine, ximodel_fine, logM_fine, R_fine, logZ_fine = \
        interp_likelihood(init_out, nlogM, nR, nlogZ, interp_lnlike=interp_lnlike, interp_ximodel=interp_ximodel)

    ori_logM_fine = logM_fine
    ori_R_fine = R_fine
    ori_logZ_fine = logZ_fine

    if interp_lnlike == False:
        #lnlike_file_name = 'plots/enrichment/inference_enrichment_debug/seed_%d_%0.2f_%0.2f_%0.2f/' \
        #                   % (seed, logM_guess, R_guess, logZ_guess) + 'finer_grid/lnlike_fine_arbinterp.npy'
        print("Reading in pre-computed interpolated likelihood", lnlike_file_name)
        lnlike_fine = np.load(lnlike_file_name)
        logM_fine = logM_fine[10:240]
        R_fine = R_fine[10:280]
        logZ_fine = logZ_fine[10:240]

    if interp_ximodel == False:
        #ximodel_file_name = 'plots/enrichment/inference_enrichment_debug/seed_%d_%0.2f_%0.2f_%0.2f/' \
        #                    % (seed, logM_guess, R_guess, logZ_guess) + 'finer_grid/xi_model_fine.npy'
        print("Reading in pre-computed interpolated ximodel_fine", ximodel_file_name)
        ximodel_fine = np.load(ximodel_file_name)

    params, _, _, _, _, _ = read_model_grid(modelfile)

    if run_mcmc:
        sampler, param_samples, bounds = mcmc_inference(nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, \
                                                        lnlike_fine, linear_prior, ball_size=ball_size, seed=seed, \
                                                        savefits_chain=savefits_chain)

        # using ori_[param]_fine because consistent with shape of ximodel_fine and to avoid indexing error
        plot_mcmc(sampler, param_samples, init_out, params, ori_logM_fine, ori_R_fine, ori_logZ_fine, ximodel_fine, linear_prior,
                  seed=seed)


        coarse_out = lnlike_coarse, logM_coarse, R_coarse, logZ_coarse
        fine_out = lnlike_fine, ori_logM_fine, ori_R_fine, ori_logZ_fine, logM_fine, R_fine, logZ_fine, ximodel_fine
        return init_out, coarse_out, fine_out, params, sampler, param_samples

    else:
        return init_out, params, ori_logM_fine, ori_R_fine, ori_logZ_fine, ximodel_fine, linear_prior, seed

################ checking, debugging ################
def interp_likelihood_fixedlogZ(init_out, ilogZ, nlogM_fine, nR_fine, interp_lnlike=False, interp_ximodel=False):

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

    print('dlogM_fine', dlogM_fine)
    print('dR', dR_fine)

    # Loop over the coarse grid and evaluate the likelihood at each location for the chosen mock data
    # Needs to be repeated for each chosen mock data
    if interp_lnlike:
        lnlike_coarse = np.zeros((nlogM, nR))
        for ilogM, logM_val in enumerate(logM_coarse):
            for iR, R_val in enumerate(R_coarse):
                lnlike_coarse[ilogM, iR] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ilogM, iR, ilogZ, :], \
                                                                 lndet_array[ilogM, iR, ilogZ], icovar_array[ilogM, iR, ilogZ, :, :])

        print('interpolating lnlike')
        start = time.time()
        lnlike_fine = inference.interp_lnlike(logM_fine, R_fine, logM_coarse, R_coarse, lnlike_coarse) # RectBivariateSpline
        end = time.time()
        print((end - start) / 60.)
    else:
        lnlike_fine = None

    """"
    # Only needs to be done once, unless the fine grid is change
    if interp_ximodel:
        start = time.time()
        print('interpolating model')
        xi_model_fine = inference.interp_model_3d(logM_fine, R_fine, logZ_fine, logM_coarse, R_coarse, logZ_coarse, xi_model_array)
        end = time.time()
        print((end-start)/60.)
    else:
        xi_model_fine = None
    """

    return lnlike_coarse, lnlike_fine, logM_fine, R_fine

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

def plot_likelihood_data(lnlike, logM_grid, R_grid, logZ_grid, logM_data, R_data, logZ_data, savefig=None):

    ilogM = find_closest(logM_grid, logM_data)
    iR = find_closest(R_grid, R_data)
    ilogZ = find_closest(logZ_grid, logZ_data)
    print(ilogM, iR, ilogZ)

    plt.figure(figsize=(12, 8))
    # plot lnL
    plt.subplot(231)
    plt.plot(logM_grid, lnlike[:, iR, ilogZ], '.-')
    plt.axvline(logM_data, ls='--', c='k', label='logM_data=% 0.2f' % logM_data)
    plt.legend()
    plt.xlabel('logM', fontsize = 13)
    plt.ylabel('lnL', fontsize=13)

    plt.subplot(232)
    plt.plot(R_grid, lnlike[ilogM, :, ilogZ], '.-')
    plt.axvline(R_data, ls='--', c='k', label = 'R_data=%0.2f' % R_data)
    plt.legend()
    plt.xlabel('R (Mpc)', fontsize=13)

    plt.subplot(233)
    plt.plot(logZ_grid, lnlike[ilogM, iR], '.-')
    plt.axvline(logZ_data, ls='--', c='k', label = 'logZ_data=%0.2f' % logZ_data)
    plt.legend()
    plt.xlabel('logZ', fontsize=13)

    # plot prob
    delta_lnL = lnlike - lnlike.max()
    Prob = np.exp(delta_lnL)  # un-normalized
    lnlike = Prob

    plt.subplot(234)
    plt.plot(logM_grid, lnlike[:, iR, ilogZ], '.-')
    plt.axvline(logM_data, ls='--', c='k', label='logM_data=% 0.2f' % logM_data)
    plt.xlabel('logM', fontsize=13)
    plt.ylabel('Prob', fontsize=13)

    plt.subplot(235)
    plt.plot(R_grid, lnlike[ilogM, :, ilogZ], '.-')
    plt.axvline(R_data, ls='--', c='k', label='R_data=%0.2f' % R_data)
    plt.xlabel('R (Mpc)', fontsize=13)

    plt.subplot(236)
    plt.plot(logZ_grid, lnlike[ilogM, iR], '.-')
    plt.axvline(logZ_data, ls='--', c='k', label='logZ_data=%0.2f' % logZ_data)
    plt.xlabel('logZ', fontsize=13)

    plt.tight_layout()

    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()

def prep_for_arbinterp(logM_coarse, R_coarse, logZ_coarse, lnlike_coarse, outtxt):

    field = np.zeros((len(logM_coarse) * len(R_coarse) * len(logZ_coarse), 4))
    n = 0
    for ilogZ, logZ in enumerate(logZ_coarse):
        for iR, R in enumerate(R_coarse):
            for ilogM, logM in enumerate(logM_coarse):
                field[n] = [np.round(logM,2), np.round(R,2), np.round(logZ,2), lnlike_coarse[ilogM, iR, ilogZ]]
                n += 1

    np.savetxt(outtxt, field, fmt=['%0.2f', '%0.2f', '%0.2f', '%f'], delimiter=',')

def prep_for_arbinterp2(logM_coarse, R_coarse, logZ_coarse, logM_fine, R_fine, logZ_fine, outtxt):

    im_lo = np.argwhere(np.round(logM_fine,2)==np.round(logM_coarse[1],2))[0][0] # 2nd elem
    im_hi = np.argwhere(np.round(logM_fine, 2) == np.round(logM_coarse[-2], 2))[0][0] # 2nd-to-last elem

    ir_lo = np.argwhere(np.round(R_fine, 2) == np.round(R_coarse[1], 2))[0][0]
    ir_hi = np.argwhere(np.round(R_fine, 2) == np.round(R_coarse[-2], 2))[0][0]

    iz_lo = np.argwhere(np.round(logZ_fine, 2) == np.round(logZ_coarse[1], 2))[0][0]
    iz_hi = np.argwhere(np.round(logZ_fine, 2) == np.round(logZ_coarse[-2], 2))[0][0]

    print(im_lo, im_hi, ir_lo, ir_hi, iz_lo, iz_hi)

    allpts = []
    new_logM_fine = logM_fine[im_lo:im_hi]
    new_R_fine = R_fine[ir_lo:ir_hi]
    new_logZ_fine = logZ_fine[iz_lo:iz_hi]
    print(len(new_logM_fine), len(new_R_fine), len(new_logZ_fine))

    for ilogM, logM_val in enumerate(new_logM_fine):
        for iR, R_val in enumerate(new_R_fine):
            for ilogZ, logZ_val in enumerate(new_logZ_fine):
                pts = np.array([logM_val, R_val, logZ_val])
                allpts.append(pts)

    print(allpts[0])
    print(allpts[-1])
    print(len(allpts))

    np.savetxt(outtxt, allpts, fmt=['%0.2f', '%0.2f', '%0.2f'], delimiter=',')
    return len(new_logM_fine), len(new_R_fine), len(new_logZ_fine)

def plot_corner_nonthinned(mcmc_chain_filename, config_file, linear_prior=False):

    config = configparser.ConfigParser()
    config.read(config_file)
    modelfile = config['DEFAULT']['modelfile']
    seed = int(config['DEFAULT']['seed'])
    logM_guess, R_guess, logZ_guess = float(config['DEFAULT']['logm_guess']), float(
        config['DEFAULT']['r_guess']), float(config['DEFAULT']['logz_guess'])

    init_out = init(modelfile, logM_guess, R_guess, logZ_guess, seed)
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out

    mcmc = fits.open(mcmc_chain_filename)
    param_samples = mcmc['ALL_CHAIN_DISCARD_BURNIN'].data
    var_label = ['log(M)', 'R', '[C/H]']
    truths = [10 ** (logM_data), R_data, 10 ** (logZ_data)] if linear_prior else [logM_data, R_data, logZ_data]

    corner.corner(param_samples, labels=var_label, truths=truths, levels=(0.68,), color='k', \
                        truth_color='darkgreen', \
                        show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20}, \
                        data_kwargs={'ms': 1.0, 'alpha': 0.1})

    plt.show()

def mcmc_upperlim_boundary():

    savefits_chain = 'plots/enrichment/inference_enrichment_debug/seed_5377192_10.89_0.20_-4.40/mcmc_chain_linearprior.fits'
    chain = fits.open(savefits_chain)
    ps = chain['param_samples'].data
    allchain_noburn = chain['ALL_CHAIN_DISCARD_BURNIN'].data
    logM_ps, R_ps, logZ_ps = ps[:, 0], ps[:, 1], ps[:, 2]
    logM_all, R_all, logZ_all = allchain_noburn[:, 0], allchain_noburn[:, 1], np.log10(allchain_noburn[:, 2])

    # model boundaries
    logM_min, logM_max = 8.5, 11
    R_min, R_max = 0.1, 3.0
    logZ_min, logZ_max = -4.5, -2

    # param samples
    plt.subplot(231)
    plt.hist(logM_ps, color='k', bins=50, histtype='step')
    plt.axvline(logM_min)
    plt.axvline(logM_max)

    plt.subplot(232)
    plt.hist(R_ps, color='k', bins=50, histtype='step')
    plt.axvline(R_min)
    plt.axvline(R_max)

    plt.subplot(233)
    plt.hist(10**logZ_ps, color='k', bins=50, histtype='step')
    plt.axvline(10**logZ_min)
    plt.axvline(10**logZ_max)

    # all chain
    plt.subplot(234)
    plt.hist(logM_all, color='k', bins=50, histtype='step')
    plt.axvline(logM_min)
    plt.axvline(logM_max)

    plt.subplot(235)
    plt.hist(R_all, color='k', bins=50, histtype='step')
    plt.axvline(R_min)
    plt.axvline(R_max)

    plt.subplot(236)
    plt.hist(10**logZ_all, color='k', bins=50, histtype='step')
    plt.axvline(10**logZ_min)
    plt.axvline(10**logZ_max)

    plt.show()

