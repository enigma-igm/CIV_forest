import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from astropy.io import fits
from astropy.table import hstack, vstack

def compute_xi_all(params, skewers, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=None, sampling=None, cgm_dict=None):

    # similar as enigma.reion_forest.fig_corrfunc.py

    vel_lores, (flux_lores_tot, flux_lores_igm, flux_lores_cgm), \
    vel_hires, (flux_hires_tot, flux_hires_igm, flux_hires_cgm), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(params, skewers, logZ, fwhm, metal_ion, sampling=sampling, cgm_dict=cgm_dict)

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
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF from all the skewers, i.e the final quoted 2PCF

    return vel_mid, xi_mean_tot, xi_tot, npix_tot

def write_corr(vel_mid, xi_tot, npix_tot, outfile):

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