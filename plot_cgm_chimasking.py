import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from linetools.lists.linelist import LineList
from astropy import units as u
from astropy import constants as const
import civ_cgm # new version
import time
from scipy.stats import norm
from enigma.reion_forest.compute_model_grid import read_model_grid
import civ_find
from enigma.reion_forest import inference
import inference_enrichment

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
plt.subplots_adjust(left=0.11, bottom=0.09, right=0.98, top=0.89)

xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2

################
skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
par = Table.read(skewerfile, hdu=1)
ske = Table.read(skewerfile, hdu=2)
z = par['z'][0]
logZ = -3.5
metal_ion = 'C IV'
fwhm = 10
snr = 50
sampling = 3.0
seed = 3429381 # random seeds for drawing CGM absorbers
rand = np.random.RandomState(seed)

cgm_alpha = -1.1
cgm_n_star = 5
metal_dndz_func = civ_cgm.civ_dndz_sch
cgm_model = civ_cgm.init_metal_cgm_dict(alpha=cgm_alpha, n_star=cgm_n_star) # rest are default

nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0 # gives d(oneminf) = 0.01
sig_min, sig_max = 1e-2, 100.0
signif_mask_nsigma = 10
one_minF_thresh = 0.05
signif_thresh = 4.0
#signif_mask_dv = 300.0

signif_mask_dv = 200.0 # b-strong=150 km/s; CF after masking practically same as dv=300
W_2796_igm = 1.0 # default
bval_igm = 150 # default

################
start = time.time()

ske = ske[0:1000]
v_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), v_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ, fwhm, metal_ion, z=z, \
                                                                             sampling=sampling, cgm_dict=cgm_model, \
                                                                          metal_dndz_func=metal_dndz_func, seed=seed)
end = time.time()
print("............ creating metal forest done in", (end-start)/60, "min") # 2 min

noise = rand.normal(0.0, 1.0 / snr, flux_cgm_lores.shape)
flux_noise_igm_lores = flux_igm_lores + noise
flux_noise_cgm_lores = flux_cgm_lores + noise
flux_noise_tot_lores = flux_tot_lores + noise
ivar = np.full_like(noise, snr**2) # returns an array with same shape as ‘noise’ and filled with values ‘snr**2’

################
# PDF
civ_tot = civ_find.MgiiFinder(v_lores, flux_noise_tot_lores, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, \
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh, W_2796_igm=W_2796_igm, bval_igm=bval_igm)
civ_igm = civ_find.MgiiFinder(v_lores, flux_noise_igm_lores, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, \
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh, W_2796_igm=W_2796_igm, bval_igm=bval_igm)
civ_cgm = civ_find.MgiiFinder(v_lores, flux_noise_cgm_lores, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, \
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh, W_2796_igm=W_2796_igm, bval_igm=bval_igm)
civ_noise = civ_find.MgiiFinder(v_lores, 1.0 + noise, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, \
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh, W_2796_igm=W_2796_igm, bval_igm=bval_igm)

sig_bins, sig_pdf_igm = reion_utils.pdf_calc(civ_igm.signif, sig_min, sig_max, nbins)
_, sig_pdf_cgm = reion_utils.pdf_calc(civ_cgm.signif, sig_min, sig_max, nbins)
_, sig_pdf_tot = reion_utils.pdf_calc(civ_tot.signif, sig_min, sig_max, nbins)
_, sig_pdf_noise = reion_utils.pdf_calc(civ_noise.signif, sig_min, sig_max, nbins)

"""
# Compute PDFs of masked arrays
_, sig_pdf_flu_mask = reion_utils.pdf_calc(civ_tot.signif[civ_tot.flux_gpm], sig_min, sig_max, nbins)
_, sig_pdf_fit_mask = reion_utils.pdf_calc(civ_tot.signif[civ_tot.fit_gpm], sig_min, sig_max, nbins)

# Compute the PDF for mock dataset
nmocks = 1000
sig_bins_mc, sig_pdf_mc, sig_pdf_tot_mock = utils.pdf_calc_mc(mgii_tot.signif, sig_min, sig_max, nbins,
                                                                 npath, nmocks, rand=rand)
# Upper and lower limits on PDf from percentiles
sig_pdf_tot_mock_lo = np.percentile(sig_pdf_mc, 100.0*norm.cdf(-1.0), axis=0)
sig_pdf_tot_mock_hi = np.percentile(sig_pdf_mc, 100.0*norm.cdf(1.0), axis=0)
"""

################
# 2PCF
vmin_corr = 10
vmax_corr = 2000
dv_corr = 10

# igm only
start = time.time()
meanflux_igm = np.mean(flux_igm_lores)
deltaf_igm = (flux_igm_lores - meanflux_igm) / meanflux_igm
vel_mid, xi_igm, npix_igm, _ = reion_utils.compute_xi(deltaf_igm, v_lores, vmin_corr, vmax_corr, dv_corr)
xi_mean_igm = np.mean(xi_igm, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")

# igm + cgm
start = time.time()
meanflux_tot = np.mean(flux_tot_lores)
deltaf_tot = (flux_tot_lores - meanflux_tot) / meanflux_tot
vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, v_lores, vmin_corr, vmax_corr, dv_corr)
xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")

# igm + cgm (flux mask)
start = time.time()
mask_want = (1 - flux_tot_lores) < one_minF_thresh
meanflux_tot_fluxmask = np.mean(flux_tot_lores[mask_want])
deltaf_tot_fluxmask = (flux_tot_lores - meanflux_tot_fluxmask) / meanflux_tot_fluxmask
vel_mid, xi_tot_fluxmask, npix_tot_fluxmask, _ = reion_utils.compute_xi(deltaf_tot_fluxmask, v_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask_want)
xi_mean_tot_fluxmask = np.mean(xi_tot_fluxmask, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")
n_gpm = np.count_nonzero(mask_want)
ntot = len(flux_tot_lores.flatten())
print(n_gpm, ntot, n_gpm/ntot)

# igm + cgm (flux + chi mask)
meanflux_tot_chimask = np.mean(flux_tot_lores[civ_tot.fit_gpm])
deltaf_tot_chimask = (flux_tot_lores - meanflux_tot_chimask) / meanflux_tot_chimask
vel_mid, xi_tot_chimask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_chimask, v_lores, vmin_corr, vmax_corr, dv_corr, gpm=civ_tot.fit_gpm)
xi_mean_tot_chimask = np.mean(xi_tot_chimask, axis=0) # 2PCF from all the skewers
end = time.time()
print("............ compute xi done in", (end-start)/60, "min")
n_gpm = np.count_nonzero(civ_tot.fit_gpm)
ntot = len(flux_tot_lores.flatten())
print(n_gpm, ntot, n_gpm/ntot)

# getting errors
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
logM_guess = 9.50 # set this as fiducial model
R_guess = 0.80 # fiducial model
logZ_guess = -3.50 # fiducial model

init_out = inference_enrichment.init(modelfile, logM_guess, R_guess, logZ_guess, seed=None) #don't care which mock dataset is picked
logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out

theta = np.array([logM_guess, R_guess, logZ_guess])
covar_mean = inference.covar_model_3d(theta, logM_coarse, R_coarse, logZ_coarse, covar_array)
xi_err = np.sqrt(np.diag(covar_mean))

################ plotting
plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', lw=linewidth, c='tab:gray', alpha=0.8, label='noise')
plt.plot(sig_bins, sig_pdf_igm, drawstyle='steps-mid', lw=linewidth, c='tab:orange', label='IGM + noise')
plt.plot(sig_bins, sig_pdf_cgm, drawstyle='steps-mid', lw=linewidth, c='tab:blue', label='CGM + noise')
plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid',  lw=linewidth, c='tab:green', label='IGM + CGM + noise')
plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=linewidth)

#xlim = 1e-4
#ymin, ymax = 1e-3, 3.0
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\chi$', fontsize=xylabel_fontsize)
plt.ylabel('PDF', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)
#plt.gca().set_xlim(left=xlim)
#plt.ylim([ymin, ymax])
plt.legend(fontsize=legend_fontsize, loc=2)

plt.figure()
scalefactor = 1e-5
plt.plot(vel_mid, xi_mean_igm/scalefactor, linewidth=linewidth, linestyle='-', c='tab:orange', label='IGM')
plt.plot(vel_mid, xi_mean_tot/(scalefactor*50), linewidth=linewidth, linestyle='-', c='tab:gray', label='IGM + CGM, unmasked/50')
plt.plot(vel_mid, xi_mean_tot_fluxmask/scalefactor, linewidth=linewidth, linestyle='--', c='tab:blue', label='IGM + CGM, flux masked')
plt.plot(vel_mid, xi_mean_tot_chimask/scalefactor, linewidth=linewidth, linestyle='-', c='tab:blue', label='IGM + CGM, flux+chi masked')
plt.fill_between(vel_mid, (xi_mean_tot_chimask - xi_err)/scalefactor, (xi_mean_tot_chimask + xi_err)/scalefactor, facecolor='tab:blue', step='mid', alpha=0.5, zorder=1)

vmin, vmax = 0, 1250
ymin, ymax = -0.1, 2.5
plt.legend(fontsize=legend_fontsize)
plt.xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)
plt.xlim([vmin, vmax])
plt.ylim([ymin, ymax])


plt.show()