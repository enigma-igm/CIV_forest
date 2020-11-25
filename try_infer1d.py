import numpy as np
import emcee
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest.compute_model_grid import read_model_grid
from enigma.reion_forest import inference
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# log prior
def lnprior_1d(theta, logZ_fine, linearZprior):
    if linearZprior:
        retval  = 1.0 if (0.0 <= 10**theta <= np.power(10.0,logZ_fine.max())) else -np.inf
    else:
        retval = 1.0 if (logZ_fine.min() <= theta[1] <= logZ_fine.max()) else -np.inf
    return retval

# log likelihood
def lnlike_1d(theta, lnlike_fine, logZ_fine, linearZprior):

    #logZ = np.log10(np.fmax(theta[1], 1e-12)) if linearZprior else theta[1]
    logZ = np.log10(np.fmax(theta, 1e-12)) if linearZprior else theta
    iZ_fine = find_closest(logZ_fine, logZ)
    lnL = lnlike_fine[iZ_fine]
    return lnL

# interpolating coarse log likelihood into finer grids
def interp_lnlike_1d(logZ_fine, logZ_coarse, lnlike_coarse):

    # Interpolate the lnL onto the fine grid to speed up the MCMC
    lnlike_interp_func = interp1d(logZ_coarse, lnlike_coarse)
    lnlike_fine = lnlike_interp_func(logZ_fine)

    return lnlike_fine

# log prob
def lnprob_1d(theta, lnlike_fine, logZ_fine, linearZprior):

    lp = lnprior_1d(theta, logZ_fine, linearZprior)
    if not np.isfinite(lp):
        print('return inf')
        return -np.inf
    return lp + lnlike_1d(theta, lnlike_fine,  logZ_fine, linearZprior)

def init_var(modelfile, logZ_guess):
    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    #xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr'][0]
    vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    # nhi = params['nhi'][0]

    # Pick the data that we will run with
    rand = np.random.RandomState(111)
    nmock = xi_mock_array.shape[2]
    imock = rand.choice(np.arange(nmock), size=1)
    #logZ_guess = -3.31

    # find the closest model values to guesses
    # ixhi = find_closest(xhi_coarse, xhi_guess)
    ixhi = 0
    iZ = find_closest(logZ_coarse, logZ_guess)
    # xhi_data = xhi_coarse[ixhi]
    logZ_data = logZ_coarse[iZ]
    xi_data = xi_mock_array[ixhi, iZ, imock, :].flatten()
    xi_mask = np.ones_like(xi_data, dtype=bool)

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    nlogZ_fine = 1001
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

    ixhi = 0
    lnlike_coarse = np.zeros(nlogZ)
    for iZ, logZ in enumerate(logZ_coarse):
        lnlike_coarse[iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ixhi, iZ, :], lndet_array[ixhi, iZ], \
                                                  icovar_array[ixhi, iZ, :, :])

    return logZ_fine, logZ_coarse, lnlike_coarse

def infer(logZ_fine, logZ_coarse, lnlike_coarse):
    nwalkers = 40
    ndim = 1
    nsteps = 130000
    burnin = 1000
    pos = np.random.uniform(-6, -2, nwalkers)

    linearZprior = False
    lnlike_fine = interp_lnlike_1d(logZ_fine, logZ_coarse, lnlike_coarse)
    args = lnlike_fine, logZ_fine, linearZprior

    return pos, lnlike_fine, logZ_fine, linearZprior

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_1d, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)

import scipy
def plot_prob(lnlike, logZ):
    like_ratio = np.exp(lnlike - lnlike.max()) # linear; L/L_max
    norm = scipy.integrate.trapz(like_ratio)
    print(norm)

    plt.figure()
    plt.plot(logZ, like_ratio/norm, 'k.-')
    plt.xlabel('logZ', fontsize=13)
    plt.ylabel(r'$L/L_{max}$', fontsize=13)

    plt.figure()
    plt.plot(logZ, lnlike)
    plt.xlabel('logZ', fontsize=13)
    plt.ylabel('lnL', fontsize=13)

    plt.show()