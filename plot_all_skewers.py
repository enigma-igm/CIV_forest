"""
Plot oden, T, x_metal, and flux skewers for a randomly selected skewer.
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'
#skewerfile = data_path + 'enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r0.34375_logM9.00.fits'
skewerfile = data_path + 'enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r1.37500_logM10.00.fits'
#skewerfile = data_path + 'enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r0.34375_logM9.00.fits'
metal_par = Table.read(skewerfile, hdu=1)
metal_ske = Table.read(skewerfile, hdu=2)

logZ = -3.5
metal_ion = 'C IV'
fwhm = 10 # km/s
snr = 50

i = np.random.randint(0, len(metal_ske))
print('random index', i)

# creating the metal forest for random skewer 'i'
v_lores, (ftot_lores, figm_lores, fcgm_lores), \
v_hires, (ftot_hires, figm_hires, fcgm_hires), \
(oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(metal_par, metal_ske[[i]], logZ, fwhm, metal_ion)
# # ~0.00014 sec to generate one skewer

#### uniformly enriched ####
ori_skewerfile = data_path + 'rand_skewers_z45_ovt_tau_xciv_flux.fits' # uniformly enriched
ori_metal_par = Table.read(ori_skewerfile, hdu=1)
ori_metal_ske = Table.read(ori_skewerfile, hdu=2)

ori_v_lores, (ori_ftot_lores, ori_figm_lores, ori_fcgm_lores), \
ori_v_hires, (ori_ftot_hires, ori_figm_hires, ori_fcgm_hires), \
(ori_oden, ori_v_los, ori_T, ori_x_metal), ori_cgm_tup = reion_utils.create_metal_forest(ori_metal_par, ori_metal_ske[[i]], logZ, fwhm, metal_ion)
###########################

tau = metal_ske['TAU'][i]
vmin, vmax = v_hires.min(), v_hires.max()

# Add noise
noise = np.random.normal(0.0, 1.0/snr, ftot_lores[0].flatten().shape)
ftot_lores_noise = ftot_lores[0] + noise

#fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12,8.5), sharex=True)
fig, (ax1, ax2) = plt.subplots(2, figsize=(12,7), sharex=True)

"""
#### oden plot ####
ax1.plot(v_hires, oden[0], c='k')
ax1.set_ylabel('Overdensity', fontsize=13)
ax1.set_xlim([vmin, vmax])

#### temp plot ####
ax2.plot(v_hires, T[0], c='k')
ax2.set_ylabel('Temperature', fontsize=13)
ax2.set_xlim([vmin, vmax])
"""
#### x_metal plot ####
ax1.plot(ori_v_hires, ori_x_metal[0], alpha=0.7)
ax1.plot(v_hires, x_metal[0], c='k')
ax1.set_ylabel('X_CIV', fontsize=13)
ax1.set_xlim([vmin, vmax])

#### flux plot ####
ax2.plot(ori_v_hires, ori_ftot_hires[0], alpha=0.7, label='hires (uniform Z)')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
ax2.plot(v_hires, ftot_hires[0], c='k', label='hires')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
#ax4.plot(v_lores, ftot_lores_noise, label='FWHM=%0.1f km/s; SNR=%0.1f' % (fwhm, snr), c='r', alpha=0.6, zorder=1)#, drawstyle='steps-mid')
ax2.set_xlabel('v (km/s)', fontsize=14)
ax2.set_ylabel('F_CIV', fontsize=13)
ax2.legend()
ax2.set_xlim([vmin, vmax])
ax2.set_ylim([0.85, 1.1])

plt.tight_layout()
plt.show()
