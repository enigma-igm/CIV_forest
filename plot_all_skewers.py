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
import halos_skewers

# TODO: Plot enrichment topology (i.e. mask skewer in the xciv panel) and Skewer of N_CIV.

### Figure settings
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(16, 13), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.98, top=0.93, wspace=0, hspace=0.)

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2.5
alpha_uniform = 0.6
alpha_data = 0.5

### Files
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
sampling = 3

# cosmology
z = metal_par['z'][0]
cosmo = FlatLambdaCDM(H0=100.0 * metal_par['lit_h'][0], Om0=metal_par['Om0'][0], Ob0=metal_par['Ob0'][0])
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)

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

#### oden plot ####
ax1.plot(v_hires, oden[0], c='k')
#ax1.set_ylabel('Overdensity', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\Delta$ [$\rho/\bar{\rho}$]', fontsize=xylabel_fontsize)
ax1.tick_params(top=True, which='both', labelsize=xytick_size)
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
# block below is hack from reion_utils.create_metal_forest() to get the vel-axis
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
ax2.set_ylabel('Enrichment \ntopology', fontsize=xylabel_fontsize)
ax2.set_xlim([vmin, vmax])
ax2.tick_params(top=True, which='both', labelsize=xytick_size)
ax2.xaxis.set_minor_locator(AutoMinorLocator())

#### x_metal plot ####
ax3.plot(ori_v_hires, ori_x_metal[0], alpha=alpha_uniform, label='uniform \nenrichment')
ax3.plot(v_hires, x_metal[0], 'k')
#ax3.annotate('logM = {:5.2f}, '.format(logM) + 'R = {:5.2f} Mpc, '.format(R_Mpc) + '[C/H] = ${:5.2f}$'.format(logZ), xy=(50,0.5), xytext=(50,0.5), textcoords='data', xycoords='data', annotation_clip=False, fontsize=12)
ax3.legend(fontsize=legend_fontsize, loc='lower center', bbox_to_anchor=(0.61, 0.12))
ax3.set_ylabel(r'X$_{\mathrm{CIV}}$', fontsize=xylabel_fontsize)
ax3.set_xlim([vmin, vmax])
ax3.tick_params(top=True, which='both', labelsize=xytick_size)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.set_ylim([-0.05, 0.5])

#### N_CIV plot
nH_bar = 3.1315263992114194e-05 # from other skewerfile
vscale = np.ediff1d(v_hires)[0]
pixscale = (vscale*u.km/u.s/a/Hz).to('cm').value # equals 35.6 ckpc
#pixscale = ((100/4096)*u.Mpc).to(u.cm) # equals 24 h^-1 ckpc (9/8/21)
pixscale *= a # proper distance
N_civ = halos_skewers.get_Nciv(oden[0], x_metal[0], logZ, nH_bar, pixscale)

ax4.plot(v_hires, N_civ/1e10, c='k')
ax4.set_ylabel(r'N$_{\mathrm{CIV}}$' + '\n' + r'[10$^{10}$ cm$^{-2}$]', fontsize=xylabel_fontsize)
ax4.tick_params(top=True, which='both', labelsize=xytick_size)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.set_xlim([vmin, vmax])

#### flux plot ####
#ax3.plot(ori_v_hires, ori_ftot_hires[0], alpha=0.7, label='hires (uniform Z)')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
ax5.plot(v_hires, ftot_hires[0], 'k', label='Perfect spectrum', drawstyle='steps-mid')#, alpha=0.6, zorder=10, color='red')
ax5.plot(v_lores, ftot_lores_noise, label='FWHM=%0.1f km/s; SNR=%0.1f' % (fwhm, snr), c='r', alpha=alpha_data, zorder=1, drawstyle='steps-mid')
ax5.annotate('log(M)={:5.2f} '.format(logM) + r'M$_{\odot}$, ' + 'R={:5.2f} cMpc, '.format(R_Mpc) + '[C/H]=${:5.2f}$'.format(logZ), \
             xy=(500, 1.085), xytext=(500, 1.085), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize)
ax5.set_xlabel('v [km/s]', fontsize=xylabel_fontsize)
ax5.set_ylabel(r'F$_{\mathrm{CIV}}$', fontsize=xylabel_fontsize)
ax5.legend(fontsize=legend_fontsize, ncol=2, loc=1)
ax5.set_xlim([vmin, vmax])
ax5.set_ylim([0.9, 1.12])
ax5.tick_params(top=True, which='both', labelsize=xytick_size)
ax5.xaxis.set_minor_locator(AutoMinorLocator())
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# plot upper axis
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
atwin = ax1.twiny()
atwin.set_xlabel('R [cMpc]', fontsize=xylabel_fontsize, labelpad=8)
atwin.axis([rmin, rmax, oden_min, oden_max])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(savefig)
plt.show()
plt.close()
