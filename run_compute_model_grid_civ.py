import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import enigma.reion_forest.utils as reion_utils
import enigma.reion_forest.compute_model_grid_civ as compute

rantaufile = ('/Users/suksientie/Research/CIV_forest/nyx_sim_data/subset100_civ_forest.fits')
par100 = Table.read(rantaufile, hdu=1)
ske100 = Table.read(rantaufile, hdu=2)

logZ = -3.5
fwhm = 10
sampling = 3
nqsos = 10
delta_z = 0.2

npath = compute.get_npath(par100, ske100, logZ, fwhm, sampling, nqsos, delta_z)

SNR = 20
vmin_corr = 20
vmax_corr = 2000
dv_corr = 5
ncovar = 10000
nmock = 50 # number of mock datasets to store -- irrelevant for now
seed = 1199
"""
vel_mid, xi_mock, xi_mean, covar, icovar, logdet = compute.compute_model_test(logZ, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock, seed)

np.save('nyx_sim_data/vel_mid.npy', vel_mid)
np.save('nyx_sim_data/xi_mock.npy', xi_mock)
np.save('nyx_sim_data/xi_mean.npy', xi_mean)
np.save('nyx_sim_data/covar.npy', covar)

"""

# plot CF as well ??

vel_mid = np.load('nyx_sim_data/vel_mid.npy')
xi_mock = np.load('nyx_sim_data/xi_mock.npy')
xi_mean = np.load('nyx_sim_data/xi_mean.npy')
covar = np.load('nyx_sim_data/covar.npy')

corr = covar/np.sqrt(np.outer(np.diag(covar),np.diag(covar))) # correlation matrix; see Eqn 14 of Hennawi+ 2020
plt.imshow(corr, origin='lower', cmap='inferno', interpolation='nearest', extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr], vmin=0.0, vmax=1.0)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)
plt.colorbar()
plt.title(r'nqso=%d, $\Delta z$=%0.1f, npath=%d, ncovar=%d' % (nqsos, delta_z, npath, ncovar), fontsize=18)
plt.show()
