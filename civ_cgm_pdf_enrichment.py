import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from linetools.lists.linelist import LineList
from astropy import constants as const
from astropy import units as u
import civ_cgm
import halos_skewers
import time

# to run on IGM machine where production files are
taupath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/tau/'
fvfm_igm_path = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/fvfm_all.fits'

# random seeds for drawing CGM absorbers
seed = 102938
rand = np.random.RandomState(seed)

def init(R_want, logM_want, logZ=-3.5, metal_ion='C IV', fwhm=10, sampling=3.0):
    # creates flux skewers for the input enrichment model, using production files stored on IGM machine

    skewerfile = taupath + 'rand_skewers_z45_ovt_xciv_tau_R_%0.2f' % R_want + '_logM_%0.2f' % logM_want + '.fits'
    par = Table.read(skewerfile, hdu=1)
    ske = Table.read(skewerfile, hdu=2)
    z = par['z'][0]

    # cgm model
    in_cgm_dict_old = civ_cgm.init_metal_cgm_dict(alpha=-0.50, W_star=0.45, n_star=28.0, W_min=0.001, W_max=5.0, \
                                           b_weak=10.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0,
                                           logN_strong=14.5, logN_trans=0.35)

    in_cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-0.75, n_star=5.0) # using Sarah's data

    metal_dndz_func = civ_cgm.civ_dndz_sch

    start = time.time()
    v_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
    v_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ, fwhm, metal_ion, z=z, \
                                                                         sampling=sampling, cgm_dict=in_cgm_dict, \
                                                                         metal_dndz_func=metal_dndz_func, seed=seed)
    end = time.time()
    print("creating metal forest done in", (end-start)/60)

    return v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, \
           v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand

def plot_pdf_simple(flux, label=None, noise=False):
    """
    Simple plotting of the flux pdf.
    Inputs:
        flux: flux skewers from reion_utils.create_metal_forest
        label: plot label
        noise: if the input 'flux' is random noise, then set to True.
    """

    npix = flux.size
    nbins = 101
    oneminf_max = 1.0
    oneminf_min = 1e-5

    #dbinf = (np.log10(oneminf_max) - np.log10(oneminf_min)) / (nbins - 1)
    #log_flux_bins = np.log10(oneminf_min) + np.arange(nbins) * dbinf
    #flux_bins = np.power(10.0, log_flux_bins)

    if noise:
        flux = 1 + flux
        flux_bins, pdf_out = reion_utils.pdf_calc(1.0 - flux, oneminf_min, oneminf_max, nbins)
        #flux_bins, pdf_out = reion_utils.pdf_calc(-flux, oneminf_min, oneminf_max, nbins)
    else:
        flux_bins, pdf_out = reion_utils.pdf_calc(1.0 - flux, oneminf_min, oneminf_max, nbins)

    if label != None:
        plt.plot(flux_bins, pdf_out, drawstyle='steps-mid', label=label)
    else:
        plt.plot(flux_bins, pdf_out, drawstyle='steps-mid')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)

    plt.tight_layout()
    #plt.show()

def varying_fv(outfig, snr=50):
    """
    Plot the flux PDFs for varying volume filling factors (by varying logM and R values).
    """

    # looking at just a subset of all models
    #logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.50, 0.1, 3, 0.4)
    #R_want = R[:-1]  # the last element is outside model boundaries

    logM_want = 8.5
    R_want = [0.1, 0.7, 1.3, 1.9, 2.5]

    for iR, Rval in enumerate(R_want):
        print("Rval", Rval)
        v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, \
        v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand = init(Rval, logM_want)

        noise = rand.normal(0.0, 1.0 / snr, flux_cgm_lores.shape)
        #flux_noise_igm_lores = flux_igm_lores + noise
        #flux_noise_cgm_lores = flux_cgm_lores + noise

        fv_want, fm_want = halos_skewers.get_fvfm(logM_want, Rval, fvfm_file=fvfm_igm_path)
        logZ_eff_want = halos_skewers.calc_igm_Zeff(fm_want, logZ_fid=-3.5)

        if iR == 0: # plot with noise
            plot_pdf_simple(flux_cgm_lores, label='CGM')
            plot_pdf_simple(noise, label='noise (SNR=%d)' % snr, noise=True)

        plot_pdf_simple(flux_igm_lores, label=r'IGM (log$Z_{eff}=%0.3f$)' % logZ_eff_want)

    plt.legend(fontsize=10)
    plt.savefig(outfig)