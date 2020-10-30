import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'
skewerfile = data_path + 'subset100_civ_forest.fits' # generated by run_reion_skewers_metal.py

metal_par = Table.read(skewerfile, hdu=1)
metal_ske = Table.read(skewerfile, hdu=2)

logZ = -3.5
fwhm = 10 # km/s; 100 for JWST
metal_ion = 'C IV'

i = np.random.randint(0, len(metal_ske))
i = 68
print('random index', i)

# creating the metal forest for random skewer 'i'
v_lores, (ftot_lores, figm_lores, fcgm_lores), \
v_hires, (ftot_hires, figm_hires, fcgm_hires), \
(oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(metal_par, metal_ske[[i]], logZ, fwhm, metal_ion)

tau = metal_ske['TAU'][i]

vmin, vmax = v_hires.min(), v_hires.max()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12,9), sharex=True)

# oden plot
ax1.plot(v_hires, oden[0])
ax1.set_ylabel('Overdensity', fontsize=13)
ax1.set_xlim([vmin, vmax])
# temp plot
ax2.plot(v_hires, T[0])
ax2.set_ylabel('Temperature', fontsize=13)
ax2.set_xlim([vmin, vmax])
# x_metal plot
ax3.plot(v_hires, x_metal[0])
ax3.set_ylabel('X_metal', fontsize=13)
ax3.set_xlim([vmin, vmax])
# tau plot
#ax4.plot(tau)
#ax4.set_ylabel('Tau', fontsize=13)
#ax4.set_xlim([vmin, vmax])
# flux plot
ax4.plot(v_hires, ftot_hires[0], label='hires')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
ax4.plot(v_lores, ftot_lores[0], label='FWHM=%0.1f km/s' % fwhm, c='r', alpha=0.6)#, drawstyle='steps-mid', color='k', zorder=1, linewidth=1.5)
ax4.set_xlabel('v (km/s)', fontsize=14)
ax4.set_ylabel('F_metal', fontsize=13)
ax4.legend()
ax4.set_xlim([vmin, vmax])

plt.tight_layout()
plt.show()