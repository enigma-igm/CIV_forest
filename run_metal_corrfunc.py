import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import time

metal_ion = 'C IV'
logZ = -3.5
fwhm = 10 #10.
sampling = 3
vmin_corr = fwhm
vmax_corr = 3000.
dv_corr = fwhm/sampling
corr_outfile = 'nyx_sim_data/subset100_civ_forest_corrfunc_fwhm10.fits' # saving output correlation functions
tau_metal_file = 'nyx_sim_data/subset100_civ_forest.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
compute_corr = False

if compute_corr:

    params = Table.read(tau_metal_file, hdu=1)
    skewers = Table.read(tau_metal_file, hdu=2)

    start = time.time()
    vel_mid, xi_mean_tot, xi_tot, npix_tot = mcf.compute_xi_all(params, skewers, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, sampling=sampling)
    mcf.write_corr(vel_mid, xi_tot, npix_tot, corr_outfile)
    end = time.time()

    print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

else:
    outcorr = Table.read(corr_outfile)
    vel_mid = outcorr['vel_mid'][0]
    xi_tot = outcorr['xi_tot']
    xi_mean_tot = np.mean(xi_tot, axis=0)

    factor = 1.0
    plt.figure(figsize=(8,8))
    plt.plot(vel_mid, factor*xi_mean_tot, linewidth=2.0, linestyle='-')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

    ymin, ymax = (factor*xi_mean_tot).min(), 1.07*((factor*xi_mean_tot).max())
    vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
    plt.vlines(vel_doublet.value, ymin=ymin, ymax=ymax, color='red', linestyle='--', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    #plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f, dv=%0.1f' % (len(xi_tot), fwhm, sampling, logZ, dv_corr), fontsize=15)
    plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
              '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
    plt.legend(frameon=False)
    plt.ylim([ymin, ymax])
    plt.show()