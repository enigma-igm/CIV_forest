import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import time

# parameters to set
metal_ion = 'C IV'
logZ = -3.5
fwhm = 10    # for creating the metal forest
sampling = 3 # for creating the metal forest
vmin_corr = fwhm
vmax_corr = 3000.
#dv_corr = fwhm/sampling
dv_corr = 5
snr = None # or None for noiseless data
npath = 100
seed = 1199 # only used if npath < len(skewers)
rand = np.random.RandomState(seed)

# input and output files
tau_metal_file = 'nyx_sim_data/subset100_civ_forest.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
corr_outfile = 'nyx_sim_data/subset100_civ_forest_corrfunc_dv5_npath9.fits' # saving output correlation function
compute_corr = False

if compute_corr:

    params = Table.read(tau_metal_file, hdu=1)
    skewers = Table.read(tau_metal_file, hdu=2)

    if npath < len(skewers):
        print('randomly selecting %d skewers...' % npath)
        indx = rand.choice(len(skewers), replace=False, size=9)
        skewers = skewers[indx]

    start = time.time()
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all(params, skewers, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=snr, sampling=sampling)
    mcf.write_corr(vel_mid, xi_tot, npix_tot, corr_outfile)
    end = time.time()

    print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

else:
    noiseless_corr = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_dv3p3_npath9.fits')
    vel_mid_noiseless = noiseless_corr['vel_mid'][0]
    xi_tot_noiseless = noiseless_corr['xi_tot']
    xi_mean_tot_noiseless = np.mean(xi_tot_noiseless, axis=0)

    outcorr = Table.read(corr_outfile)
    vel_mid = outcorr['vel_mid'][0]
    xi_tot = outcorr['xi_tot']
    xi_mean_tot = np.mean(xi_tot, axis=0)

    factor = 1.0
    plt.figure(figsize=(12,8))
    plt.plot(vel_mid, factor*xi_mean_tot, linewidth=2.0, linestyle='-')#, label='SNR=%d' % snr)
    #plt.plot(vel_mid_noiseless, factor*xi_mean_tot_noiseless, linewidth=2.0, linestyle='-', color='k', alpha=0.6, label='Noiseless')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

    ymin, ymax = (factor*xi_mean_tot).min(), 1.07*((factor*xi_mean_tot).max())
    vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
    plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
              '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
    plt.legend(frameon=False)
    plt.xlim([-50, 2000])
    plt.ylim([ymin, ymax])
    plt.show()