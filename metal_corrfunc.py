import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from astropy.io import fits
from astropy.table import hstack, vstack
from enigma.reion_forest.compute_model_grid_civ import read_model_grid

def compute_xi_all(params, skewers, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=None, sampling=None, \
                   cgm_dict=None, metal_dndz_func=None, cgm_seed=None):

    # similar as enigma.reion_forest.fig_corrfunc.py
    # if sampling not provided, then default to sampling=3

    vel_lores, (flux_lores_tot, flux_lores_igm, flux_lores_cgm), \
    vel_hires, (flux_hires_tot, flux_hires_igm, flux_hires_cgm), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(params, skewers, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                         cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed)

    # Add noise if snr is provided
    if snr != None:
        print("adding random noise with SNR=%d" % snr)
        noise = np.random.normal(0.0, 1.0 / snr, np.shape(flux_lores_tot))
        flux_lores_tot += noise

    # Compute mean flux and delta_flux
    mean_flux_tot = np.mean(flux_lores_tot)
    delta_f_tot = (flux_lores_tot - mean_flux_tot)/mean_flux_tot
    print('mean flux:', mean_flux_tot)
    print('mean delta_flux:', np.mean(delta_f_tot))

    # xi_tot is an array of 2PCF of each skewer
    (vel_mid, xi_tot, npix_tot, xi_zero_lag_tot) = reion_utils.compute_xi(delta_f_tot, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF averaged from all the skewers, i.e the final quoted 2PCF

    return vel_mid, xi_mean_tot, xi_tot, npix_tot

def compute_xi_all_flexi(vel, flux_tot, vmin_corr, vmax_corr, dv_corr):

    # same as compute_xi_all but takes output of reion_utils.create_metal_forest() as inputs for flexibility

    # Compute mean flux and delta_flux
    mean_flux_tot = np.mean(flux_tot)
    delta_f_tot = (flux_tot - mean_flux_tot)/mean_flux_tot
    print('mean flux:', mean_flux_tot)
    print('mean delta_flux:', np.mean(delta_f_tot))

    # xi_tot is an array of 2PCF of each skewer
    (vel_mid, xi_tot, npix_tot, xi_zero_lag_tot) = reion_utils.compute_xi(delta_f_tot, vel, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF from all the skewers, i.e the final quoted 2PCF

    return vel_mid, xi_mean_tot, xi_tot, npix_tot

def write_corrfunc(vel_mid, xi_tot, npix_tot, outfile):

    # hack because shape(vel_mid) != shape(xi_tot
    vel_mid2 = []
    for i in range(len(xi_tot)):
        vel_mid2.append(vel_mid)

    #xi_mean_tot = np.mean(xi_tot, axis=0)
    #tab0 = Table([xi_mean_tot], names=('xi_mean_tot'))
    tab1 = Table([vel_mid2, xi_tot, npix_tot], names=('vel_mid', 'xi_tot', 'npix_tot'))

    #hdu_param = fits.BinTableHDU(tab0.as_array())
    hdu_table = fits.BinTableHDU(tab1.as_array())
    hdulist = fits.HDUList()
    #hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)

def plot_corrmatrix(params, covar):
    # plotting the correlation matrix, for outputs of enigma.reion_forest.compute_model_grid_civ.py

    nqsos = params['nqsos'][0]
    delta_z = params['delta_z'][0]
    npath = params['npath'][0]
    ncovar = params['ncovar'][0]
    SNR = params['SNR'][0]
    vmin_corr = params['vmin_corr'][0]
    vmax_corr = params['vmax_corr'][0]

    corr = covar / np.sqrt(np.outer(np.diag(covar), np.diag(covar)))  # correlation matrix; see Eqn 14 of Hennawi+ 2020

    plt.figure(figsize=(8, 8))
    plt.imshow(corr, origin='lower', interpolation='nearest', extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr], \
               vmin=0.0, vmax=1.0, cmap='inferno')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.title(r'nqso=%d, $\Delta z$=%0.1f, npath=%d' % (nqsos, delta_z, npath) + '\n' + 'ncovar=%d, SNR=%d' % (ncovar, SNR), fontsize=18)
    plt.colorbar()

    plt.show()

def plot_corrfunc(params, xi_model, label=None):

    vel_mid = params['vel_mid'][0]
    vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)

    plt.plot(vel_mid, xi_model, linewidth=2.0, linestyle='-', label=label)
    plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.6) #label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.legend(fontsize=15)
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)


def plot_all_corrmatrix(modelfile, type, outfig=None, cf_overplot=False):
    # for outputs of enigma.reion_forest.compute_model_grid_civ.py
    # type is either "cov" or "corrfunc"

    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)

    nqsos = params['nqsos'][0]
    delta_z = params['delta_z'][0]
    npath = params['npath'][0]
    ncovar = params['ncovar'][0]
    SNR = params['SNR'][0]
    vmin_corr = params['vmin_corr'][0]
    vmax_corr = params['vmax_corr'][0]
    logZ_vec = params['logZ'][0]
    vel_mid = params['vel_mid'][0]

    # getting rid of redundant axis
    covar_array = covar_array[0]
    xi_model_array = xi_model_array[0]

    plt.figure(figsize=(12,10))
    if type == 'cov':
        for i in range(len(covar_array)):
            covar = covar_array[i]
            corr = covar / np.sqrt(np.outer(np.diag(covar), np.diag(covar)))  # correlation matrix; see Eqn 14 of Hennawi+ 2020
            plt.subplot(3, 3, i + 1)
            plt.imshow(corr, origin='lower', cmap='inferno', interpolation='nearest', \
                       extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr])#, vmin=0.0, vmax=1.0)
            plt.title(r'logZ = $%0.1f$' % logZ_vec[i], fontsize=12)
            plt.colorbar()
            #plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
            #plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)

    elif type == 'corrfunc':
        vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
        #factor = [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1, 1]

        # plotting the mean of all mocks
        for i in range(len(xi_model_array)):
            if cf_overplot:
                plt.plot(vel_mid, xi_model_array[i], linewidth=2.0, linestyle='-',
                         label=r'logZ = $%0.1f$' % logZ_vec[i])
                if i == (len(xi_model_array) - 1):
                    plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2,
                                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
                plt.legend(frameon=False, fontsize=12)
                plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
                plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)
            else:
                plt.subplot(3, 3, i + 1)
                plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2,
                            label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
                plt.plot(vel_mid, xi_model_array[i], linewidth=2.0, linestyle='-')
                plt.title(r'logZ = $%0.1f$' % logZ_vec[i], fontsize=12)
                #plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
                #plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

    plt.suptitle(r'nqso=%d, $\Delta z$=%0.1f, npath=%d, ncovar=%d, SNR=%d' % (nqsos, delta_z, npath, ncovar, SNR), fontsize=18)

    if outfig != None:
        plt.savefig(outfig)
    else:
        plt.show()

def plot_single_cov_elem(modelfile, rand_i=None, rand_j=None):
    # for outputs of enigma.reion_forest.compute_model_grid_civ.py

    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_vec = params['logZ'][0]

    covar_array = covar_array[0] # shape = (logZ, vel_mid, vel_mid)

    # constructing the correlation matrix
    corr_array = []
    for i in range(len(covar_array)):
        corr = covar_array[i] / np.sqrt(np.outer(np.diag(covar_array[i]), np.diag(covar_array[i])))  # correlation matrix; see Eqn 14 of Hennawi+ 2020
        corr_array.append(corr)
    corr_array = np.array(corr_array)

    if rand_i == None and rand_j == None:
        i_size, j_size = np.shape(covar_array)[1:]
        rand_i, rand_j = np.random.randint(i_size), np.random.randint(j_size)
        print(rand_i, rand_j)

    covar_array = corr_array # plot the correlation element instead

    covar_array_d = covar_array[:, rand_i, rand_i]
    covar_array_nd = covar_array[:, rand_i, rand_j]

    plt.plot(logZ_vec, covar_array_d, 'o-', label=r"Diag: ($i,j$)=(%d,%d)" % (rand_i, rand_i))
    plt.plot(logZ_vec, covar_array_nd, 'o-', label=r"Off-diag: ($i,j$)=(%d,%d)" % (rand_i, rand_j))
    plt.xlabel('log(Z)', fontsize=13)
    plt.ylabel(r'Corr$_{ij}$=Cov$_{ij}$/$\sqrt{Cov_{ii}Cov_{jj}}$', fontsize=13)
    #plt.ylabel('Covariance', fontsize=13)
    #plt.yscale('log')
    plt.legend()
    plt.show()

##################### Temporary plotting #####################
def temp1_compare_cf_enrichment():

    indx = np.random.choice(10000, replace=False, size=1000)
    ori_par = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits', hdu=1)
    ori_ske = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits', hdu=2)[indx]

    mask1_par = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r0.34375_logM9.00.fits', hdu=1)
    mask1_ske = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r0.34375_logM9.00.fits', hdu=2)[indx]

    mask2_par = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r1.37500_logM10.00.fits', hdu=1)
    mask2_ske = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r1.37500_logM10.00.fits', hdu=2)[indx]

    mask3_par = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r2.75000_logM9.00.fits', hdu=1)
    mask3_ske = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r2.75000_logM9.00.fits', hdu=2)[indx]

    return ori_par, ori_ske, mask1_par, mask1_ske, mask2_par, mask2_ske, mask3_par, mask3_ske

def temp2_compare_cf_enrichment(ori_par, ori_ske, mask1_par, mask1_ske, mask2_par, mask2_ske, mask3_par, mask3_ske, logZ, fwhm=10, wantlores=True):

    vel_lores_ori, (flux_lores_tot_ori, flux_lores_igm, flux_lores_cgm), \
    vel_hires_ori, (flux_hires_tot_ori, flux_hires_igm, flux_hires_cgm), _, _ = \
        reion_utils.create_metal_forest(ori_par, ori_ske, logZ, fwhm, 'C IV', sampling=3)

    vel_lores_mask1, (flux_lores_tot_mask1, flux_lores_igm, flux_lores_cgm), \
    vel_hires_mask1, (flux_hires_tot_mask1, flux_hires_igm, flux_hires_cgm), _, _ = \
        reion_utils.create_metal_forest(mask1_par, mask1_ske, logZ, fwhm, 'C IV', sampling=3)

    vel_lores_mask2, (flux_lores_tot_mask2, flux_lores_igm, flux_lores_cgm), \
    vel_hires_mask2, (flux_hires_tot_mask2, flux_hires_igm, flux_hires_cgm), _, _ = \
        reion_utils.create_metal_forest(mask2_par, mask2_ske, logZ, fwhm, 'C IV', sampling=3)

    vel_lores_mask3, (flux_lores_tot_mask3, flux_lores_igm, flux_lores_cgm), \
    vel_hires_mask3, (flux_hires_tot_mask3, flux_hires_igm, flux_hires_cgm), _, _ = \
        reion_utils.create_metal_forest(mask3_par, mask3_ske, logZ, fwhm, 'C IV', sampling=3)

    if wantlores:
        vel_mid_ori, xi_mean_tot_ori, _, _ = compute_xi_all2(vel_lores_ori, flux_lores_tot_ori, 10, 2000, 5)
        vel_mid_mask1, xi_mean_tot_mask1, _, _ = compute_xi_all2(vel_lores_mask1, flux_lores_tot_mask1, 10, 2000, 5)
        vel_mid_mask2, xi_mean_tot_mask2, _, _ = compute_xi_all2(vel_lores_mask2, flux_lores_tot_mask2, 10, 2000, 5)
        vel_mid_mask3, xi_mean_tot_mask3, _, _ = compute_xi_all2(vel_lores_mask3, flux_lores_tot_mask3, 10, 2000, 5)
    else:
        vel_mid_ori, xi_mean_tot_ori, _, _ = compute_xi_all2(vel_hires_ori, flux_hires_tot_ori, 10, 2000, 5)
        vel_mid_mask1, xi_mean_tot_mask1, _, _ = compute_xi_all2(vel_hires_mask1, flux_hires_tot_mask1, 10, 2000, 5)
        vel_mid_mask2, xi_mean_tot_mask2, _, _ = compute_xi_all2(vel_hires_mask2, flux_hires_tot_mask2, 10, 2000, 5)
        vel_mid_mask3, xi_mean_tot_mask3, _, _ = compute_xi_all2(vel_hires_mask3, flux_hires_tot_mask3, 10, 2000, 5)

    return vel_mid_ori, xi_mean_tot_ori, vel_mid_mask1, xi_mean_tot_mask1, \
           vel_mid_mask2, xi_mean_tot_mask2, vel_mid_mask3, xi_mean_tot_mask3

def temp3_plot_cf_enrichment(vel_mid_hires_ori, xi_mean_tot_hires_ori, vel_mid_hires_mask1, xi_mean_tot_hires_mask1, vel_mid_hires_mask2, xi_mean_tot_hires_mask2, vel_mid_hires_mask3, xi_mean_tot_hires_mask3):

    masklabel = ['r0.34375_logM9.00', 'r1.37500_logM10.00', 'r2.75000_logM9.00']
    plt.plot(vel_mid_hires_ori, xi_mean_tot_hires_ori, label='uniform')
    plt.plot(vel_mid_hires_mask1, xi_mean_tot_hires_mask1, label = masklabel[0])
    plt.plot(vel_mid_hires_mask2, xi_mean_tot_hires_mask2, label = masklabel[1])
    plt.plot(vel_mid_hires_mask3, xi_mean_tot_hires_mask3, label=masklabel[2])
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=13)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=13)
    plt.legend()
    plt.show()


##################### Inferencing section #####################
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from enigma.reion_forest.inference import lnprior_1d, lnlike_1d, interp_lnlike_1d, lnprob_1d
from scipy import integrate
from scipy.special import logsumexp

def init_inference_1d_civ(logZ_guess):

    seed = 125913
    rand = np.random.RandomState(seed)

    # Read in the model grid
    path = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/'
    modelfile = path + 'igm_cluster/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'

    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr'][0]
    vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]

    nmock = xi_mock_array.shape[2]  # np.shape(xi_mock_array) = nhi, nlogZ, nmock, ncorr
    imock = rand.choice(np.arange(nmock), size=1)
    #logZ_guess = -4.33  # the desired value, to be interpolated to the nearest grid value
    linearZprior = False

    # find the closest model values to guesses
    ixhi = 0  # xhi = find_closest(xhi_coarse, xhi_guess)
    iZ = find_closest(logZ_coarse, logZ_guess)
    # xhi_data = xhi_coarse[ixhi]
    logZ_data = logZ_coarse[iZ]
    xi_data = xi_mock_array[ixhi, iZ, imock, :].flatten()
    xi_mask = np.ones_like(xi_data, dtype=bool)  # in case you want to mask any xi value, otherwise all True

    # Interpolate logZ grid into finer grid
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

    lnlike_fine = interp_lnlike_1d(logZ_fine, logZ_coarse, lnlike_coarse)
    xi_model_fine = inference.interp_model_1d(logZ_fine, logZ_coarse, xi_model_array)

    return lnlike_fine, logZ_fine, dlogZ_fine

def normalize_like(lnlike_fine, logZ_fine, dlogZ_fine):

    lnlike_fine_new = lnlike_fine - lnlike_fine.max()

    # normalize using trapz
    norm = integrate.trapz(np.exp(lnlike_fine_new), logZ_fine)  # summing L
    plogZ = np.exp(lnlike_fine_new) / norm # P = np.exp(lnL - lnL.max())

    # normalize using logsumexp = np.log(np.sum(np.exp(input)))
    ln_norm = logsumexp(lnlike_fine_new) + np.log(dlogZ_fine) # np.log(dlogZ_fine)
    plogZ2 = np.exp(lnlike_fine_new - ln_norm)

    #print(integrate.trapz(plogZ, logZ_fine))
    #print(integrate.trapz(plogZ2, logZ_fine))

    #plt.subplot(121)
    #plt.plot(logZ_fine, plogZ)
    #plt.subplot(122)
    #plt.plot(logZ_fine, plogZ2)
    #plt.show()
    return plogZ

def calc_plogZ(logZ_guess_list):

    plogZ_ls = []
    for i in range(len(logZ_guess_list)):
        lnlike_fine, logZ_fine, dlogZ_fine = init_inference_1d_civ(logZ_guess_list[i])
        plogZ = normalize_like(lnlike_fine, logZ_fine, dlogZ_fine)
        plogZ_ls.append(plogZ)
        #plt.plot(logZ_fine, plogZ, '.-', ms=4)

    #plt.xlabel('logZ', fontsize=15)
    #plt.ylabel('P(logZ)', fontsize=15)

    return plogZ_ls, logZ_fine

from random import choices
from scipy.stats import norm

def calc_prec(logZ_fine, plogZ, n_sample):
    samples = choices(logZ_fine, plogZ, k=n_sample)
    median = np.median(samples)

    lower_bound = norm.cdf(-1.0) #(1 - 0.6827) / 2.
    upper_bound = norm.cdf(1.0) #1.0 - lower_bound
    lower = median - np.percentile(samples, 100*lower_bound)
    upper = np.percentile(samples, 100 * upper_bound) - median

    plt.plot(logZ_fine, plogZ)
    plt.hist(samples, bins=40, density=True)
    plt.title('Median=%0.3f, lower=%0.3f, upper=%0.3f' % (median, lower, upper), fontsize=15)
    plt.show()