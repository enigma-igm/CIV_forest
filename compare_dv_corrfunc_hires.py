import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils
import metal_corrfunc

##### comparing the CF at varying dv bins, using hires and lores spectra #####
metal_ion = 'C IV'
logZ = -3.5
fwhm = 10    # for creating the metal forest
sampling = 3 # for creating the metal forest
vmin_corr = 10
vmax_corr = 2000

snr = None # or None for noiseless data
npath = 5000
seed = 319450 # only used if npath < len(skewers)
rand = np.random.RandomState(seed)
dv_corr_ls = [4, 5, 10, 20, 40]

tau_metal_file = 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
#tau_metal_file = 'nyx_sim_data/subset100/subset100_civ_forest.fits'

params = Table.read(tau_metal_file, hdu=1)
skewers = Table.read(tau_metal_file, hdu=2)

if npath < len(skewers):
    print('randomly selecting %d skewers...' % npath)
    indx = rand.choice(len(skewers), replace=False, size=9)
    skewers = skewers[indx]

vel_lores, (flux_lores_tot, flux_lores_igm, flux_lores_cgm), vel_hires, (flux_hires_tot, flux_hires_igm, flux_hires_cgm), \
(oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(params, skewers, logZ, fwhm, metal_ion, sampling=sampling)

out_vel_mid = []
out_ximean_tot = []
plt.figure(figsize=(10,5))

# CF using hires spectra
plt.subplot(121)
plt.title('HIRES', fontsize=16)
for dv_corr in dv_corr_ls:
    vel_mid, xi_mean_tot, xi_tot, npix_tot = metal_corrfunc.compute_xi_all_flexi(vel_hires, flux_hires_igm, vmin_corr, vmax_corr, dv_corr)
    #out_vel_mid.append(vel_mid)
    #out_ximean_tot.append(xi_mean_tot)

    plt.plot(vel_mid, xi_mean_tot, label='dv=%0.1f km/s' % dv_corr)

vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation \n (%0.1f km/s)' % vel_doublet.value)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)
plt.xlim([-50, 1000])
plt.legend()

# CF using lores spectra
plt.subplot(122)
plt.title('LORES (fwhm=%d km/s)' % fwhm, fontsize=16)

for dv_corr in dv_corr_ls:
    vel_mid, xi_mean_tot, xi_tot, npix_tot = metal_corrfunc.compute_xi_all_flexi(vel_lores, flux_lores_igm, vmin_corr, vmax_corr, dv_corr)
    #out_vel_mid.append(vel_mid)
    #out_ximean_tot.append(xi_mean_tot)
    plt.plot(vel_mid, xi_mean_tot, label='dv=%0.1f km/s' % dv_corr)

plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation \n (%0.1f km/s)' % vel_doublet.value)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)
plt.xlim([0, 1000])
plt.legend()

plt.show()




