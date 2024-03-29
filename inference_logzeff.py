'''
Functions here:
    - logzeff_coarse
    - interp_logzeff
    - interp_logzeff2
    - do_all
    - plot_corner_hack
    - plot_corrfunc_mcmc_hack
    - interp_fm
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
from astropy.table import Table
import halos_skewers
import inference_enrichment as infen
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl

def logzeff_coarse(outtxt=None):
    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.10, 0.1, 3, 0.1)
    logZ_vec = np.linspace(-4.5, -2.0, 26)  # logZ_vec = np.linspace(-4.5, -2.0, 26)
    nlogM, nR, nlogZ = len(logM), len(R), len(logZ_vec)

    fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
    fm_all = np.reshape(fvfm_master['fm'], (nlogM, nR))

    field = []
    for ilogZ, logZval in enumerate(logZ_vec):
        for iR, Rval in enumerate(R):
            for ilogM, logMval in enumerate(logM):
                fm_i = fm_all[ilogM, iR]
                logZ_eff = np.log10(10 ** (logZval) * fm_i)
                field.append(np.array([np.round(logMval, 2), np.round(Rval, 2), np.round(logZval, 2), logZ_eff]))

    if outtxt != None:
        np.savetxt(outtxt, field, fmt=['%0.2f', '%0.2f', '%0.2f', '%f'], delimiter=',')

    return np.array(field)

# re-use allpts_to_interp.csv
def interp_logzeff(coarse_field, interp_pts):
    from ARBTools.ARBInterp import tricubic

    #fname_pts_to_interp = '/Users/suksientie/Research/CIV_forest/plots/enrichment/inference_enrichment_debug/seed_5382029_9.89_0.98_-3.57/finer_grid/allpts_to_interp.csv'
    #interp_pts = np.genfromtxt(fname_pts_to_interp, delimiter=',')
    #trunc_nlogM, trunc_nR, trunc_nlogZ = 230, 270, 230

    Run = tricubic(coarse_field)

    out_norm, out_grad = Run.Query(interp_pts)  # ~10 min compute time
    #out_norm2 = out_norm[:, 0]
    #out_norm2 = np.reshape(out_norm2, (trunc_nlogM, trunc_nR, trunc_nlogZ))
    #np.save(arbinterp_outnpy, out_norm2)

    return out_norm[:,0]

def interp_logzeff2(param_samples):
    # In progress - 8/16/21
    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.10, 0.1, 3, 0.1)
    logZ_vec = np.linspace(-4.5, -2.0, 26)  # logZ_vec = np.linspace(-4.5, -2.0, 26)
    nlogM, nR, nlogZ = len(logM), len(R), len(logZ_vec)

    fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
    fm_all = np.reshape(fvfm_master['fm'], (nlogM, nR))
    logZ_eff = []

    for ilogZ, logZval in enumerate(logZ_vec):
        logZ_eff.append(np.log10(10 ** (logZval) * fm_all))

    logZ_eff_func = RegularGridInterpolator((logZ_vec, logM, R), logZ_eff)

    pts = np.zeros(np.shape(param_samples))
    pts[:,0] = param_samples[:,-1]
    pts[:,1] = param_samples[:,0]
    pts[:,2] = param_samples[:,1]

    out = logZ_eff_func(pts)
    return out

def do_all(mcmc_chain_fitsfile, new_param_samples_savefilename=None):
    field = logzeff_coarse()
    mcmc_chain = fits.open(mcmc_chain_fitsfile)
    param_samples = mcmc_chain['param_samples'].data
    #param_samples = mcmc_chain['ALL_CHAIN_DISCARD_BURNIN'].data

    interp_zeff_out = interp_logzeff(field, param_samples)

    new_param_samples = np.zeros((np.shape(param_samples)[0], 4))
    new_param_samples[:,0] = param_samples[:,0]
    new_param_samples[:,1] = param_samples[:,1]
    new_param_samples[:,2] = param_samples[:,2]
    new_param_samples[:,3] = interp_zeff_out

    if new_param_samples_savefilename != None:
        np.save(new_param_samples_savefilename, new_param_samples)

    return new_param_samples

def plot_corner_hack(new_param_samples, logM_data, R_data, logZ_data, logZeff_data, savefig=None):
    var_label = ['log(M)', 'R', '[C/H]', r'[C/H]$_\mathrm{eff}$']
    truths = [logM_data, R_data, logZ_data, logZeff_data]

    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.minor.size'] = 4
    label_fontsize = 22
    title_fontsize = 20
    figsize = 13
    tick_fontsize = 18

    fig = corner.corner(new_param_samples, labels=var_label, truths=truths, levels=(0.68,), color='k', \
                        truth_color='darkgreen', \
                        show_titles=True, title_kwargs={"fontsize": title_fontsize}, label_kwargs={'fontsize': label_fontsize}, \
                        data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs={'lw': 1.5}, fig=plt.figure(figsize=(figsize, figsize)))

    for ax in fig.axes:
        ax.tick_params(which='both', labelsize=tick_fontsize)

    if savefig != None:
        fig.savefig(savefig)
    fig.show()

def plot_corrfunc_mcmc_hack(config_file, param_samples):
    init_out, params, ori_logM_fine, ori_R_fine, ori_logZ_fine, ximodel_fine, linear_prior, seed = infen.do_all(config_file, run_mcmc=False)

    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

    fv, fm = halos_skewers.get_fvfm(np.round(logM_data, 2), np.round(R_data, 2))
    logZeff_data = halos_skewers.calc_igm_Zeff(fm, logZ_fid=logZ_data)
    print("logZeff_data", logZeff_data)

    inference.corrfunc_plot_3d(xi_data, param_samples, params, ori_logM_fine, ori_R_fine, ori_logZ_fine, ximodel_fine, logM_coarse,
                               R_coarse, logZ_coarse, covar_array, logM_data, R_data, logZ_data, logZeff_data, nrand=50, seed=seed)

    plt.show()

#############################
def interp_fm(mcmc_fitsfilename):

    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.10, 0.1, 3, 0.1)
    nlogM, nR = len(logM), len(R)

    fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
    fm_all = np.reshape(fvfm_master['fm'], (nlogM, nR))
    fm_func = RegularGridInterpolator((logM, R), fm_all)

    mcmc_chain = fits.open(mcmc_fitsfilename)
    param_samples = mcmc_chain['param_samples'].data
    interp_pts = np.zeros((len(param_samples), 2))
    interp_pts[:,0] = param_samples[:,0]
    interp_pts[:,1] = param_samples[:,1]

    print(np.min(param_samples[:,0]), np.max(param_samples[:,0]))
    print(np.min(param_samples[:,1]), np.max(param_samples[:,1]))

    fm_out = fm_func(interp_pts)



