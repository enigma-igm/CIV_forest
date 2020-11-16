import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import enigma.reion_forest.utils as reion_utils
import enigma.reion_forest.compute_model_grid_civ as compute
from astropy.io import fits

rantaufile = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/subset100_civ_forest.fits'
par100 = Table.read(rantaufile, hdu=1)
ske100 = Table.read(rantaufile, hdu=2)

logZ = -3.5
fwhm = 10
sampling = 3
nqsos = 10
delta_z = 0.2
covar_outfile = 'nyx_sim_data/covar.fits'
compute_corr = False

if compute_corr:
    npath = compute.get_npath(par100, ske100, logZ, fwhm, sampling, nqsos, delta_z)

    SNR = 20
    vmin_corr = 20
    vmax_corr = 2000
    dv_corr = 5
    ncovar = int(1e5)
    nmock = 50 # number of mock datasets to store -- irrelevant for now
    seed = 1199

    vel_mid, xi_mock, xi_mean, covar, icovar, logdet = compute.compute_model_test(logZ, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock, seed)

    hdu_vel_mid = fits.PrimaryHDU(data=vel_mid)
    hdu_ximock = fits.ImageHDU(data=xi_mock, name='xi_mock')
    hdu_ximean = fits.ImageHDU(data=xi_mean, name='xi_mean')
    hdu_covar = fits.ImageHDU(data=covar, name='covar')
    hdu_icovar = fits.ImageHDU(data=icovar, name='icovar')
    hdu_logdet = fits.ImageHDU(data=np.array([logdet]), name='logdet')

    hdulist = fits.HDUList([hdu_vel_mid, hdu_ximean, hdu_ximock, hdu_covar, hdu_icovar, hdu_logdet])
    hdulist.writeto(covar_outfile, overwrite=True)

else:
    data_out = fits.open(covar_outfile)
    covar = data_out['COVAR'].data
    corr = covar/np.sqrt(np.outer(np.diag(covar),np.diag(covar))) # correlation matrix; see Eqn 14 of Hennawi+ 2020
    data_out.close()

    plt.figure(figsize=(8,8))
    plt.imshow(corr, origin='lower', cmap='inferno', interpolation='nearest', extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr], vmin=0.0, vmax=1.0)
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.colorbar()
    plt.title(r'nqso=%d, $\Delta z$=%0.1f, npath=%d, ncovar=%d' % (nqsos, delta_z, npath, ncovar) + '\n'+ \
              'SNR=%d, dv_corr=%d' % (SNR, dv_corr), fontsize=18)
    plt.tight_layout()
    plt.show()
