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
from scipy.interpolate import interp1d
from enigma.reion_forest.inference import lnprior_1d, lnlike_1d, interp_lnlike_1d, lnprob_1d
from enigma.reion_forest import utils as reion_utils
from scipy import integrate, interpolate
from scipy.special import logsumexp # = np.log(np.sum(np.exp(input)))
import random
from scipy.stats import norm

######## Setting up #########
seed = 2182061 # random seed to pick the mock data set
seed = None

if seed == None:
    seed = np.random.randint(0, 10000000)
    print("Using random seed", seed)

rand = np.random.RandomState(seed)
random.seed(seed)

# Path to model grid
path = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/'
modelfile = path + 'igm_cluster/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=30k (Hires)
#modelfile = path + 'igm_cluster/corr_func_models_fwhm_30.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=10k (Xshooter)
#modelfile = path + 'igm_cluster/corr_func_models_fwhm_60.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=5k (Deimos)

logZ_guess = -5.0 # the minimum value from grid; see "igm_cluster_compute_model_grid_civ.sh"
#linearZprior = True

n_sample = 15000 # for sampling random logZ values according to the derived P(logZ)
#############################

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
logZ_coarse = params['logZ'].flatten()
#xhi_coarse = params['xhi'].flatten()
vel_corr = params['vel_mid'].flatten()
vel_min = params['vmin_corr'][0]
vel_max = params['vmax_corr'][0]
nlogZ = params['nlogZ'][0]
#nhi = params['nhi'][0]

# Pick the data that we will run with
nmock = xi_mock_array.shape[2] # np.shape(xi_mock_array) = nhi, nlogZ, nmock, ncorr
imock = rand.choice(np.arange(nmock), size=1)
print('imock=%d' % imock)

# find the closest model values to guesses
ixhi = 0 # xhi = find_closest(xhi_coarse, xhi_guess)
iZ = find_closest(logZ_coarse, logZ_guess)
#xhi_data = xhi_coarse[ixhi]
logZ_data = logZ_coarse[iZ]
xi_data = xi_mock_array[ixhi, iZ, imock, :].flatten()
xi_mask = np.ones_like(xi_data, dtype=bool) # in case you want to mask any xi value, otherwise all True

# Interpolate logZ grid into finer grid
nlogZ = logZ_coarse.size
nlogZ_fine = 1001
logZ_fine_min = logZ_coarse.min()
logZ_fine_max = logZ_coarse.max()
dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine
print("dlogZ_fine is %0.3f with nlogZ_fine %d" % (dlogZ_fine, nlogZ_fine))

ixhi = 0
lnlike_coarse = np.zeros(nlogZ)
for iZ, logZ in enumerate(logZ_coarse):
    lnlike_coarse[iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ixhi, iZ, :], lndet_array[ixhi, iZ], \
                                              icovar_array[ixhi, iZ, :, :])


lnlike_fine = interp_lnlike_1d(logZ_fine, logZ_coarse, lnlike_coarse, interptype=0) # cubic interpolation

x = np.power(10, logZ_fine)
lnprior = 1/x # uninformative prior
lnprob_fine = lnprior + lnlike_fine # ln(posterior) = ln(prior) + ln(likelihood)
                                                      # where ln(prior) is flat in linear scale

lnprob_fine_new = lnprob_fine - lnprob_fine.max()
lnprob_norm = integrate.trapz(np.exp(lnprob_fine_new), x) # summing L
prob = np.exp(lnprob_fine_new) / lnprob_norm

cdf_prob = integrate.cumtrapz(prob, x, initial=0)
cdf_interp = interpolate.interp1d(cdf_prob, x)
cdf95 = cdf_interp(0.95)
print("95-percentile: logZ <", np.log10(cdf95))




