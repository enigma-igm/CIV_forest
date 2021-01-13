import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from linetools.lists.linelist import LineList
from astropy import constants as const
from astropy import units as u
import civ_cgm

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'
skewerfile = data_path + 'rand_skewers_z45_ovt_tau_xciv_flux.fits'
# skewerfile = data_path + 'subset100/subset100_civ_forest.fits' # subset100 for debugging
par = Table.read(skewerfile, hdu=1)
ske = Table.read(skewerfile, hdu=2)

logZ = -3.5
metal_ion = 'C IV'
fwhm = 10 # km/s
snr = 50
sampling = 3.0 # number of pixels per resolution element
z = par['z'][0]
# snr, fwhm = 100, 10

# CGM model
cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star = 0.45, n_star = 28.0, W_min=0.01, W_max=5.0, \
                                       b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0, logN_strong=14.5, logN_trans=0.25)

metal_dndz_func = civ_cgm.civ_dndz_sch

# random seeds for drawing CGM absorbers
seed = 102938
rand = np.random.RandomState(seed)

def init(in_cgm_dict=cgm_dict, logZ=logZ):
    # Generating the forest with CGM absorbers
    v_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
    v_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ, fwhm, metal_ion, z=z, sampling=sampling, \
                                                                         cgm_dict=in_cgm_dict, metal_dndz_func=metal_dndz_func, seed=seed)

    return v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand

def plot_pdf(v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand):

    noise = rand.normal(0.0, 1.0/snr, flux_cgm_lores.shape)
    flux_noise_igm_lores = flux_igm_lores + noise
    flux_noise_cgm_lores = flux_cgm_lores + noise
    flux_noise_tot_lores = flux_tot_lores + noise

    npix = flux_igm_lores.size
    nbins = 101
    oneminf_max = 1.0
    oneminf_min = 1e-5

    # no noise
    flux_bins, pdf_igm, = reion_utils.pdf_calc(1.0 - flux_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm, = reion_utils.pdf_calc(1.0 - flux_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot, = reion_utils.pdf_calc(1.0 - flux_tot_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_noise, = reion_utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)
    # with noise
    _, pdf_igm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot_noise, = reion_utils.pdf_calc(1.0 - flux_noise_tot_lores, oneminf_min, oneminf_max, nbins)

    strong_lines = LineList('Strong', verbose=False)
    wave_1548 = strong_lines['CIV 1548']['wrest']
    Wfactor = ((fwhm / sampling) * u.km / u.s / const.c).decompose() * wave_1548.value
    Wmin, Wmax = Wfactor * oneminf_min, Wfactor * oneminf_max
    ymin, ymax = 1e-3, 3.0

    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label='IGM')
    plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM')
    #plt.plot(flux_bins, pdf_tot, drawstyle='steps-mid', label='IGM + CGM')
    plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.ylim([ymin, ymax])
    plt.legend(fontsize=13)
    atwin = plt.twiny()
    atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}  (\mathrm{{\AA}})$', fontsize=13)  # , labelpad=8)
    atwin.xaxis.tick_top()
    atwin.set_xscale('log')
    atwin.axis([Wmin, Wmax, ymin, ymax])
    atwin.tick_params(top=True)

    plt.subplot(122)
    plt.plot(flux_bins, pdf_igm_noise, drawstyle='steps-mid', label='IGM + noise')
    plt.plot(flux_bins, pdf_cgm_noise, drawstyle='steps-mid', label='CGM + noise')
    plt.plot(flux_bins, pdf_tot_noise, drawstyle='steps-mid', color='r', label='IGM + CGM + noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.ylim([ymin, ymax])
    plt.legend(fontsize=13)

    atwin = plt.twiny()
    atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}  (\mathrm{{\AA}})$', fontsize=13) #, labelpad=8)
    atwin.xaxis.tick_top()
    atwin.set_xscale('log')
    atwin.axis([Wmin, Wmax, ymin, ymax])
    atwin.tick_params(top=True)
    #atwin.tick_params(axis="x", labelsize=16)

    plt.suptitle('logZ = %0.1f, SNR = %d, fwhm = %d km/s' % (logZ, snr, fwhm), fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_wrange():
    cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star=0.45, n_star=28.0, W_min=0.01, W_max=5.0, \
                                           b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0,
                                           logN_strong=14.5, logN_trans=0.25)

    # Wmin, Wmax = 0.01, 0.1
    cgm_dict_sub1 = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star=0.45, n_star=28.0, W_min=0.01, W_max=0.1, \
                                                b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0,
                                                logN_strong=14.5, logN_trans=0.25)

    # Wmin, Wmax = 0.1, 0.5
    cgm_dict_sub2 = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star=0.45, n_star=28.0, W_min=0.1, W_max=0.5, \
                                                b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0,
                                                logN_strong=14.5, logN_trans=0.25)

    # Wmin, Wmax = 0.5, 5.0
    cgm_dict_sub3 = civ_cgm.init_metal_cgm_dict(alpha=-0.20, W_star=0.45, n_star=28.0, W_min=0.5, W_max=5.0, \
                                                b_weak=20.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0,
                                                logN_strong=14.5, logN_trans=0.25)

    v_lores, flux_tot_lores, flux_igm_lores, flux_cgm_lores, \
    v_hires, flux_tot_hires, flux_igm_hires, flux_cgm_hires, cgm_tup, rand = init(cgm_dict)

    v_lores_sub1, flux_tot_lores_sub1, flux_igm_lores_sub1, flux_cgm_lores_sub1, \
    v_hires_sub1, flux_tot_hires_sub1, flux_igm_hires_sub1, flux_cgm_hires_sub1, cgm_tup_sub1, rand_sub1 = \
        init(cgm_dict_sub1)

    v_lores_sub2, flux_tot_lores_sub2, flux_igm_lores_sub2, flux_cgm_lores_sub2, \
    v_hires_sub2, flux_tot_hires_sub2, flux_igm_hires_sub2, flux_cgm_hires_sub2, cgm_tup_sub2, rand_sub2 = \
        init(cgm_dict_sub2)

    v_lores_sub3, flux_tot_lores_sub3, flux_igm_lores_sub3, flux_cgm_lores_sub3, \
    v_hires_sub3, flux_tot_hires_sub3, flux_igm_hires_sub3, flux_cgm_hires_sub3, cgm_tup_sub3, rand_sub3 = \
        init(cgm_dict_sub3)

    noise = rand.normal(0.0, 1.0 / snr, flux_cgm_lores.shape)

    nbins = 101
    oneminf_max = 1.0
    oneminf_min = 1e-5

    # no noise
    flux_bins, pdf_igm = reion_utils.pdf_calc(1.0 - flux_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm = reion_utils.pdf_calc(1.0 - flux_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot = reion_utils.pdf_calc(1.0 - flux_tot_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_noise = reion_utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_sub1 = reion_utils.pdf_calc(1.0 - flux_cgm_lores_sub1, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_sub2 = reion_utils.pdf_calc(1.0 - flux_cgm_lores_sub2, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_sub3 = reion_utils.pdf_calc(1.0 - flux_cgm_lores_sub3, oneminf_min, oneminf_max, nbins)

    strong_lines = LineList('Strong', verbose=False)
    wave_1548 = strong_lines['CIV 1548']['wrest']
    Wfactor = ((fwhm / sampling) * u.km / u.s / const.c).decompose() * wave_1548.value
    Wmin, Wmax = Wfactor * oneminf_min, Wfactor * oneminf_max
    ymin, ymax = 1e-3, 3.0

    plt.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label='IGM')
    plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM')
    plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise')
    plt.plot(flux_bins, pdf_cgm_sub1, drawstyle='steps-mid', label=r'CGM ($W_{min} = 0.01, W_{max} = 0.1)$')
    plt.plot(flux_bins, pdf_cgm_sub2, drawstyle='steps-mid', label=r'CGM ($W_{min} = 0.1, W_{max} = 0.5)$')
    plt.plot(flux_bins, pdf_cgm_sub3, drawstyle='steps-mid', label=r'CGM ($W_{min} = 0.5, W_{max} = 5.0)$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.ylim([ymin, ymax])
    plt.legend(fontsize=13)

    atwin = plt.twiny()
    atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}  (\mathrm{{\AA}})$', fontsize=13)  # , labelpad=8)
    atwin.xaxis.tick_top()
    atwin.set_xscale('log')
    atwin.axis([Wmin, Wmax, ymin, ymax])
    atwin.tick_params(top=True)

    plt.suptitle('logZ = %0.1f, SNR = %d, fwhm = %d km/s' % (logZ, snr, fwhm), fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_logZ():

    # logZ values don't affect the CGM pdf
    logZ0 = -4.5
    logZ1 = -3.5
    logZ2 = -3.0

    v_lores_0, flux_tot_lores_0, flux_igm_lores_0, flux_cgm_lores_0, \
    v_hires_0, flux_tot_hires_0, flux_igm_hires_0, flux_cgm_hires_0, cgm_tup_0, rand_0 = init(cgm_dict, logZ0)

    v_lores_1, flux_tot_lores_1, flux_igm_lores_1, flux_cgm_lores_1, \
    v_hires_1, flux_tot_hires_1, flux_igm_hires_1, flux_cgm_hires_1, cgm_tup_1, rand_1 = init(cgm_dict, logZ1)

    v_lores_2, flux_tot_lores_2, flux_igm_lores_2, flux_cgm_lores_2, \
    v_hires_2, flux_tot_hires_2, flux_igm_hires_2, flux_cgm_hires_2, cgm_tup_2, rand_2 = init(cgm_dict, logZ2)

    noise = rand.normal(0.0, 1.0 / snr, flux_cgm_lores_0.shape)

    nbins = 101
    oneminf_max = 1.0
    oneminf_min = 1e-5

    # no noise
    flux_bins, pdf_igm0 = reion_utils.pdf_calc(1.0 - flux_igm_lores_0, oneminf_min, oneminf_max, nbins)
    _, pdf_igm1 = reion_utils.pdf_calc(1.0 - flux_igm_lores_1, oneminf_min, oneminf_max, nbins)
    _, pdf_igm2 = reion_utils.pdf_calc(1.0 - flux_igm_lores_2, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm = reion_utils.pdf_calc(1.0 - flux_cgm_lores_0, oneminf_min, oneminf_max, nbins)
    _, pdf_noise = reion_utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)

    strong_lines = LineList('Strong', verbose=False)
    wave_1548 = strong_lines['CIV 1548']['wrest']
    Wfactor = ((fwhm / sampling) * u.km / u.s / const.c).decompose() * wave_1548.value
    Wmin, Wmax = Wfactor * oneminf_min, Wfactor * oneminf_max
    ymin, ymax = 1e-3, 3.0

    plt.plot(flux_bins, pdf_igm0, drawstyle='steps-mid', label='IGM, logZ = %0.1f' % logZ0)
    plt.plot(flux_bins, pdf_igm1, drawstyle='steps-mid', label='IGM, logZ = %0.1f' % logZ1)
    plt.plot(flux_bins, pdf_igm2, drawstyle='steps-mid', label='IGM, logZ = %0.1f' % logZ2)
    plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM')
    plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1-F', fontsize=13)
    plt.ylabel('PDF', fontsize=13)
    plt.ylim([ymin, ymax])
    plt.legend(fontsize=13)

    atwin = plt.twiny()
    atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}  (\mathrm{{\AA}})$', fontsize=13)  # , labelpad=8)
    atwin.xaxis.tick_top()
    atwin.set_xscale('log')
    atwin.axis([Wmin, Wmax, ymin, ymax])
    atwin.tick_params(top=True)

    plt.suptitle('SNR = %d, fwhm = %d km/s' % (snr, fwhm), fontsize=16)
    plt.tight_layout()
    plt.show()