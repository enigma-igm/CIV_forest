import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import civ_cgm

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'
skewerfile = data_path + 'rand_skewers_z45_ovt_tau_xciv_flux.fits'
skewerfile = data_path + 'subset100/subset100_civ_forest.fits' # subset100 for debugging
par = Table.read(skewerfile, hdu=1)
ske = Table.read(skewerfile, hdu=2)

logZ = -3.5
metal_ion = 'C IV'
fwhm = 10 # km/s
snr = 50
sampling = 3.0 # number of pixels per resolution element
z = par['z'][0]

# CGM model
cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star = 0.45, n_star = 28.0, W_min=0.01, W_max=5.0, \
                                       b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0, logN_strong=14.5, logN_trans=0.25)
metal_dndz_func = civ_cgm.civ_dndz_sch

# random seeds for drawing CGM absorbers
seed = 102938
rand = np.random.RandomState(seed)

def init():
    # Generating the forest with CGM absorbers
    v_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
    v_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ, fwhm, metal_ion, z=z, sampling=sampling, \
                                                                         cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=seed)

    return v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand

def plot_pdf(v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand):

    noise = rand.normal(0.0, 1.0/snr, flux_cgm_lores.shape)
    flux_noise_igm_lores = flux_igm_lores + noise
    flux_noise_cgm_lores = flux_cgm_lores + noise
    flux_noise_tot_lores = flux_tot_lores + noise

    npix = flux_igm_lores.size
    nbins = 151
    oneminf_max = 1.0
    oneminf_min = 1e-5
    #nbins = 201
    #oneminf_max = 1.0
    #oneminf_min = 1e-6

    # no noise
    flux_bins, pdf_igm, = reion_utils.pdf_calc(1.0 - flux_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm, = reion_utils.pdf_calc(1.0 - flux_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot, = reion_utils.pdf_calc(1.0 - flux_tot_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_noise, = reion_utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)
    # with noise
    _, pdf_igm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot_noise, = reion_utils.pdf_calc(1.0 - flux_noise_tot_lores, oneminf_min, oneminf_max, nbins)

    plt.subplot(121)
    plt.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label='IGM')
    plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM')
    #plt.plot(flux_bins, pdf_tot, drawstyle='steps-mid', label='IGM + CGM')
    plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.legend(fontsize=13)

    plt.subplot(122)
    plt.plot(flux_bins, pdf_igm_noise, drawstyle='steps-mid', label='IGM + noise')
    #plt.plot(flux_bins, pdf_cgm_noise, drawstyle='steps-mid', label='CGM + noise')
    plt.plot(flux_bins, pdf_tot_noise, drawstyle='steps-mid', label='IGM + CGM + noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.legend(fontsize=13)
    plt.show()