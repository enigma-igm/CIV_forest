"""
Plot oden, T, x_metal, and flux skewers for a randomly selected skewer.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# TODO: Plot enrichment topology (i.e. mask skewer in the xciv panel) and Skewer of N_CIV.

font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'

#skewerfile = data_path + 'enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r1.37500_logM10.00.fits'
#skewerfile = 'nyx_sim_data/tmp_igm_cluster/rand_skewers_z45_ovt_xciv_R_1.10_logM_11.00_tau.fits'

#skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.30_logM_9.50.fits'
#skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.30_logM_10.50.fits'
skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
#skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_10.50.fits'
#skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_2.00_logM_9.50.fits'
#skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_2.00_logM_10.50.fits'

metal_par = Table.read(skewerfile, hdu=1)
metal_ske = Table.read(skewerfile, hdu=2)

logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
logZ = -3.5
savefig = 'paper_plots/skewers_R_%0.2f_logM_%0.2f.pdf' % (logM, R_Mpc)

metal_ion = 'C IV'
fwhm = 10 # km/s
snr = 50
sampling = 3.0

#i = np.random.randint(0, len(metal_ske))
i = 2500 # other good los: 4197, 7504, 1061
print('random index', i)

# creating the metal forest for random skewer 'i'
v_lores, (ftot_lores, figm_lores, fcgm_lores), \
v_hires, (ftot_hires, figm_hires, fcgm_hires), \
(oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(metal_par, metal_ske[[i]], logZ, fwhm, metal_ion, sampling=sampling)
# # ~0.00014 sec to generate one skewer

#### uniformly enriched ####

ori_skewerfile = data_path + 'rand_skewers_z45_ovt_tau_xciv_flux.fits' # uniformly enriched
#ori_skewerfile = 'nyx_sim_data/tmp_igm_cluster/rand_skewers_z45_ovt_xciv_R_1.35_logM_11.00_tau.fits'
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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12,8.5), sharex=True)
fig.subplots_adjust(hspace=0)

#### oden plot ####
ax1.plot(v_hires, oden[0], c='k')
ax1.set_ylabel('Overdensity', fontsize=13)
ax1.tick_params(top=True, which='both', labelsize=11)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
oden_min, oden_max = -2, np.round(2 + oden[0].max())
ax1.set_xlim([vmin, vmax])
ax1.set_xlim([oden_min, oden_max])

#### temp plot ####
#ax2.plot(v_hires, T[0], c='k')
#ax2.set_ylabel('T (K)', fontsize=13)
#ax2.set_xlim([vmin, vmax])
#ax1.tick_params(axis="y", labelsize=11)

#### enrichment mask plot ####
# block below is hack from reion_utils.create_metal_forest()
vside, Ng = metal_par['VSIDE'][0], metal_par['Ng'][0]
v_min, v_max = 0.0, vside # not to be confused with vmin and vmax set above
dvpix_hires = vside/Ng
v_metal = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
npad = int(np.ceil((7.0*fwhm + v_metal.value)/dvpix_hires))
v_pad = npad*dvpix_hires
pad_tuple = ((0,0), (npad, npad))
vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)

mask_metal_pad = np.pad(metal_ske[[i]]['MASK'].data, pad_tuple, 'wrap')
mask_metal = mask_metal_pad[:,iobs_hires]
ax2.plot(v_hires, mask_metal[0], 'k')
ax2.set_ylabel('Topology', fontsize=13)
ax2.set_xlim([vmin, vmax])
ax2.tick_params(top=True, which='both', labelsize=11)
ax2.xaxis.set_minor_locator(AutoMinorLocator())

#### x_metal plot ####
ax3.plot(ori_v_hires, ori_x_metal[0], alpha=0.7)
ax3.plot(v_hires, x_metal[0], 'k')
#ax3.annotate('logM = {:5.2f}, '.format(logM) + 'R = {:5.2f} Mpc, '.format(R_Mpc) + '[C/H] = ${:5.2f}$'.format(logZ), xy=(50,0.5), xytext=(50,0.5), textcoords='data', xycoords='data', annotation_clip=False, fontsize=12)
ax3.set_ylabel(r'X$_{CIV}$', fontsize=13)
ax3.set_xlim([vmin, vmax])
ax3.tick_params(top=True, which='both', labelsize=11)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.set_ylim([-0.05, 0.49])

#### flux plot ####
#ax3.plot(ori_v_hires, ori_ftot_hires[0], alpha=0.7, label='hires (uniform Z)')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
ax4.plot(v_hires, ftot_hires[0], 'k', label='Perfect spectrum', drawstyle='steps-mid')#, alpha=0.6, zorder=10, color='red')
ax4.plot(v_lores, ftot_lores_noise, label='FWHM=%0.1f km/s; SNR=%0.1f' % (fwhm, snr), c='r', alpha=0.6, zorder=1, drawstyle='steps-mid')
ax4.annotate('logM = {:5.2f}, '.format(logM) + 'R = {:5.2f} Mpc, '.format(R_Mpc) + '[C/H] = ${:5.2f}$'.format(logZ), xy=(300,0.925), xytext=(300,0.925), textcoords='data', xycoords='data', annotation_clip=False, fontsize=13)
ax4.set_xlabel('v (km/s)', fontsize=14)
ax4.set_ylabel(r'F$_{CIV}$', fontsize=13)
ax4.legend()
ax4.set_xlim([vmin, vmax])
ax4.set_ylim([0.9, 1.12])
#ax3.tick_params(axis="x", labelsize=11)
#ax3.tick_params(axis="y", labelsize=11)
ax4.tick_params(top=True, which='both', labelsize=11)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())

# plot upper axis
z = metal_par['z'][0]
cosmo = FlatLambdaCDM(H0=100.0 * metal_par['lit_h'][0], Om0=metal_par['Om0'][0], Ob0=metal_par['Ob0'][0])

Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
atwin = ax1.twiny()
atwin.set_xlabel('R (cMpc)', fontsize=14, labelpad=8)
atwin.axis([rmin, rmax, oden_min, oden_max])
atwin.tick_params(top=True, axis="x", labelsize=11)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

#plt.tight_layout()
#plt.savefig(savefig)
plt.show()
plt.close()
