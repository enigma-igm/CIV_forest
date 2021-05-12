import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import civ_cgm
import time
from enigma.reion_forest import inference
import inference_enrichment
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from matplotlib.ticker import AutoMinorLocator

# setting the figure
font = {'family' : 'serif', 'weight' : 'normal'}
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

plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.89)

xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2
scalefactor = 1e-5

################
savefig = 'paper_plots/cf_masking_cgm.pdf'
skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
par = Table.read(skewerfile, hdu=1)
ske = Table.read(skewerfile, hdu=2)
z = par['z'][0]
logZ = -3.5
metal_ion = 'C IV'
fwhm = 10
#snr = 50
sampling = 3.0
seed = 3429381 # random seeds for drawing CGM absorbers
rand = np.random.RandomState(seed)

metal_dndz_func = civ_cgm.civ_dndz_sch
cgm_model = civ_cgm.init_metal_cgm_dict(alpha=-1.1, n_star=5) # rest are default
flux_decr_cutoff = 0.10 # all pix with 1 - F > cutoff will be masked

################
# TODO: Add mock data points
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
logM_guess = 9.50 # set this as fiducial model
R_guess = 0.80 # fiducial model
logZ_guess = -3.50 # fiducial model

"""
init_out = inference_enrichment.init(modelfile, logM_guess, R_guess, logZ_guess, seed=None) #don't care which mock dataset is picked
logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out

theta = np.array([logM_guess, R_guess, logZ_guess])
covar_mean = inference.covar_model_3d(theta, logM_coarse, R_coarse, logZ_coarse, covar_array)
xi_err = np.sqrt(np.diag(covar_mean))

plt.errorbar(vel_corr, xi_data/scalefactor, yerr=xi_err/scalefactor, marker='o', ms=3, color='black', ecolor='black', capthick=2, capsize=4, \
             alpha=0.8, mec='none', ls='none', label='mock data', zorder=20)
"""
################
vmin_corr = 10
vmax_corr = 2000
dv_corr = 10

vel_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), vel_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ, fwhm, metal_ion, z=z, \
                                                                             sampling=sampling, cgm_dict=cgm_model, \
                                                                             metal_dndz_func=metal_dndz_func, seed=seed)
# igm only
start = time.time()
meanflux_igm = np.mean(flux_igm_lores)
deltaf_igm = (flux_igm_lores - meanflux_igm) / meanflux_igm
vel_mid, xi_igm, npix_igm, _ = reion_utils.compute_xi(deltaf_igm, vel_lores, vmin_corr, vmax_corr, dv_corr)
xi_mean_igm = np.mean(xi_igm, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")

# igm + cgm
start = time.time()
meanflux_tot = np.mean(flux_tot_lores)
deltaf_tot = (flux_tot_lores - meanflux_tot) / meanflux_tot
vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel_lores, vmin_corr, vmax_corr, dv_corr)
xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")

# igm + cgm (flux mask)
start = time.time()
mask_want = (1 - flux_tot_lores) < flux_decr_cutoff
meanflux_tot_fluxmask = np.mean(flux_tot_lores[mask_want])
deltaf_tot_fluxmask = (flux_tot_lores - meanflux_tot_fluxmask) / meanflux_tot_fluxmask
vel_mid, xi_tot_fluxmask, npix_tot_fluxmask, _ = reion_utils.compute_xi(deltaf_tot_fluxmask, vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask_want)
xi_mean_tot_fluxmask = np.mean(xi_tot_fluxmask, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")

vmin, vmax = 0, 1250
ymin, ymax = -0.1, 10
plt.plot(vel_mid, xi_mean_igm/scalefactor, linewidth=linewidth, linestyle='-', c='tab:orange', label='IGM')
plt.plot(vel_mid, xi_mean_tot/(scalefactor*10), linewidth=linewidth, linestyle='-', c='tab:gray', label='IGM + CGM, unmasked/10')
plt.plot(vel_mid, xi_mean_tot_fluxmask/scalefactor, linewidth=linewidth, linestyle='-', c='tab:blue', label='IGM + CGM, masked')
plt.legend(fontsize=legend_fontsize)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)
plt.xlim([vmin, vmax])
plt.ylim([ymin, ymax])
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
text = 'log(M) = {:4.2f} '.format(logM) + r'M$_{\odot}$' + '\nR = {:4.2f} Mpc'.format(R_Mpc) + '\n[C/H] = ${:5.2f}$'.format(logZ)
plt.text(825, 6, text, fontsize=xytick_size, linespacing=1.8)

# Create upper axis in cMpc
cosmo = FlatLambdaCDM(H0=100.0 * par['lit_h'][0], Om0=par['Om0'][0], Ob0=par['Ob0'][0])
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
# Make the new upper x-axes
atwin = plt.gca().twiny()
atwin.set_xlabel('R (cMpc)', fontsize=xylabel_fontsize, labelpad=8)
atwin.xaxis.tick_top()
# atwin.yaxis.tick_right()
atwin.axis([rmin, rmax, ymin, ymax])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=xytick_size)

plt.savefig(savefig)
plt.show()