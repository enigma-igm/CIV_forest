import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import civ_cgm
import time

# parameters to set
metal_ion = 'C IV'
logZ = -3.5
fwhm = 10 # for creating the metal forest
sampling = 3 # for creating the metal forest
vmin_corr = fwhm
vmax_corr = 2000
dv_corr = 5 # has to be >= than fwhm/sampling
dv_corr_tot = 10 # for IGM + CGM
snr = 50 # or None for noiseless data
npath = 100 # number of skewers; ~ 8 min for 100 random skewers
seed = 1199876 # only used if npath < len(skewers)
rand = np.random.RandomState(seed)

# CGM model
cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-0.50, W_star = 0.45, n_star = 28.0, W_min=0.001, W_max=5.0, \
                                       b_weak=10.0, b_strong=150.0, logN_metal_min=10.0, logN_metal_max=22.0, \
                                       logN_strong=14.5, logN_trans=0.35)

metal_dndz_func = civ_cgm.civ_dndz_sch
cgm_seed = 102938  # same seed as civ_cgm_pdf.py
flux_decr_cutoff = 0.1 # mask where pixels with 1 - F > cutoff will be masked

# input and output files
tau_metal_file = 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits' # 10,000 skewers total
#tau_metal_file = 'nyx_sim_data/subset100/subset100_civ_forest.fits'

suffix = 'fcut_{:3.2f}_fwhm_{:5.3f}_samp_{:5.3f}_SNR_{:5.3f}_npath_{:d}.fits'.format(flux_decr_cutoff, fwhm, sampling, snr, npath)
corr_outfile_igm = 'nyx_sim_data/corrfunc_igm_' + suffix # pure IGM 2pcf
corr_outfile_igm_mask = 'nyx_sim_data/corrfunc_igmmask_' + suffix # pure IGM + flux mask
corr_outfile_igm_cgm = 'nyx_sim_data/corrfunc2_tot_' + suffix # total IGM+CGM 2pcf
corr_outfile_igm_cgm_mask = 'nyx_sim_data/corrfunc_totmask_' + suffix # total + flux mask 2pcf

params = Table.read(tau_metal_file, hdu=1)
skewers = Table.read(tau_metal_file, hdu=2)

# controls
compute_corr = True
compute_corr = False

if compute_corr:
    if npath < len(skewers):
        print('randomly selecting %d skewers from a total of %d ...' % (npath, len(skewers)))
        indx = rand.choice(len(skewers), replace=False, size=npath)
        skewers = skewers[indx]

    vel_lores, (flux_lores_tot, flux_lores_igm, flux_lores_cgm), \
    vel_hires, (flux_hires_tot, flux_hires_igm, flux_hires_cgm), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(params, skewers, logZ, fwhm, metal_ion,
                                                                         sampling=sampling, \
                                                                         cgm_dict=cgm_dict,
                                                                         metal_dndz_func=metal_dndz_func, seed=cgm_seed)
    if snr != None:
        print("adding random noise with SNR=%d" % snr)
        noise = np.random.normal(0.0, 1.0 / snr, np.shape(flux_lores_tot))
        flux_lores_tot += noise # adding random noise to the total flux
        flux_lores_igm += noise # adding random noise to igm flux

    # IGM only
    start = time.time()
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all_flexi(vel_lores, flux_lores_igm, vmin_corr, vmax_corr, dv_corr)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile_igm)
    end = time.time()
    print("Done computing 2PCF in %0.2f min" % ((end - start) / 60.))

    # IGM + flux cutoff
    start = time.time()
    mask_want = (1 - flux_lores_igm) < flux_decr_cutoff
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all_flexi(vel_lores, flux_lores_igm[mask_want], vmin_corr, vmax_corr, dv_corr)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile_igm_mask)
    end = time.time()
    print("Done computing 2PCF in %0.2f min" % ((end - start) / 60.))

    # IGM + CGM (no flux mask); using slightly larger dv bin
    start = time.time()
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all_flexi(vel_lores, flux_lores_tot, vmin_corr, vmax_corr, dv_corr_tot)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile_igm_cgm)
    end = time.time()
    print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

    # IGM + CGM + flux cutoff
    start = time.time()
    mask_want = (1 - flux_lores_tot) < flux_decr_cutoff
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all_flexi(vel_lores, flux_lores_tot[mask_want], vmin_corr, vmax_corr, dv_corr)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile_igm_cgm_mask)
    end = time.time()
    print("Done computing 2PCF in %0.2f min" % ((end - start) / 60.))

else:
    outcorr_igm = Table.read(corr_outfile_igm)
    vel_mid_igm = outcorr_igm['vel_mid'][0]
    xi_tot_igm = outcorr_igm['xi_tot'] # 2PCF for each skewer
    xi_mean_tot_igm = np.mean(xi_tot_igm, axis=0) # 2PCF averaging over all skewers

    outcorr_igm_mask = Table.read(corr_outfile_igm_mask)
    vel_mid_igm_mask = outcorr_igm_mask['vel_mid'][0]
    xi_tot_igm_mask = outcorr_igm_mask['xi_tot']
    xi_mean_tot_igm_mask = np.mean(xi_tot_igm_mask, axis=0)

    outcorr_igm_cgm = Table.read(corr_outfile_igm_cgm)
    vel_mid_igm_cgm = outcorr_igm_cgm['vel_mid'][0]
    xi_tot_igm_cgm = outcorr_igm_cgm['xi_tot']
    xi_mean_tot_igm_cgm = np.mean(xi_tot_igm_cgm, axis=0)

    outcorr_mask = Table.read(corr_outfile_igm_cgm_mask)
    vel_mid_mask = outcorr_mask['vel_mid'][0]
    xi_tot_mask = outcorr_mask['xi_tot']
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)

    plt.figure(figsize=(12, 8))
    scaling_factor = 500
    #plt.subplot(411)
    plt.plot(vel_mid_igm_cgm, xi_mean_tot_igm_cgm/scaling_factor, linewidth=2.0, linestyle='-', color='b', label='(IGM + CGM)/%d' % scaling_factor)
    #plt.legend()
    #plt.subplot(412)
    plt.plot(vel_mid_mask, xi_mean_tot_mask, linewidth=2.0, linestyle='-', color='r', label='IGM + CGM + mask')
    #plt.legend()
    #plt.subplot(413)
    plt.plot(vel_mid_igm, xi_mean_tot_igm, linewidth=2.0, linestyle='-', color='k', label='IGM')
    #plt.legend()
    #plt.subplot(414)
    plt.plot(vel_mid_igm_mask, xi_mean_tot_igm_mask, linewidth=2.0, linestyle='-', color='k', alpha=0.5, label='IGM + mask')

    # labeling double and observational setup
    vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
    plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.title('%d skewers, fwhm=%d km/s, SNR=%s, sampling=%d, logZ = %0.1f' % (len(xi_tot_igm), fwhm, str(snr), sampling, logZ) + \
              '\n' + r'vmin = %0.1f, vmax=%0.1f, dv=%0.1f, (1-F)$_{cut}$=%0.2f' % (vmin_corr, vmax_corr, dv_corr, flux_decr_cutoff), fontsize=15)

    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)
    plt.legend()
    plt.show()
