import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils

dv2 = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_dv2.fits')
dv3p3 = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_fwhm10.fits')
dv10 = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_dv10.fits')
dv50 = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_dv50.fits')
dv100 = Table.read('nyx_sim_data/subset100_civ_forest_corrfunc_dv100.fits')

dv_file = [dv3p3, dv10, dv50, dv100]
dv_ls = [3.3, 10, 50, 100]

plt.figure(figsize=(8,6))
for i in range(len(dv_file)):
    vel_mid = dv_file[i]['vel_mid'][0]
    xi_mean = np.mean(dv_file[i]['xi_tot'], axis=0)
    plt.plot(vel_mid, xi_mean, label='dv=%0.1f km/s' % dv_ls[i])

vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

plt.title('dv_Nyx = 3.15 km/s', fontsize=15)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)
plt.xlim([-50, 1000])
plt.legend()
plt.show()






