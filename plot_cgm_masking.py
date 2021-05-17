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
flux_decr_cutoff = 0.07
savefig = 'paper_plots/flux_pdf_masking_007.pdf'

################
start = time.time()

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

# with noise
flux_bins, pdf_igm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_igm_lores, oneminf_min, oneminf_max, nbins)
_, pdf_cgm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_cgm_lores, oneminf_min, oneminf_max, nbins)
_, pdf_tot_noise, = reion_utils.pdf_calc(1.0 - flux_noise_tot_lores, oneminf_min, oneminf_max, nbins)
_, pdf_noise = reion_utils.pdf_calc(-noise, oneminf_min, oneminf_max, nbins)

# with noise and flux cutoff
mask_want = (1 - flux_noise_tot_lores) < flux_decr_cutoff  # checked
_, pdf_tot_noise_mask = reion_utils.pdf_calc(1.0 - flux_noise_tot_lores[mask_want], oneminf_min, oneminf_max, nbins)

print('...masked igm pixel fraction', len((flux_noise_igm_lores[mask_want]).flatten()), len(flux_noise_igm_lores.flatten()))

# mc realizations to get errors on PDF
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
params_xi, _, _, _, _, _ = read_model_grid(modelfile)
npath = params_xi['npath'][0]
nmocks = params_xi['nmock'][0]
nmocks = 2*nmocks
print('npath', npath, 'nmocks', nmocks)
flux_bins_mc, flux_pdf_mc, flux_pdf_tot_mock = reion_utils.pdf_calc_mc(1.0 - flux_noise_tot_lores, oneminf_min, oneminf_max, nbins,
                                                                 npath, nmocks, rand=rand)
# Upper and lower limits on PDf from percentiles
flux_pdf_tot_mock_lo = np.percentile(flux_pdf_mc, 100.0*norm.cdf(-1.0), axis=0)
flux_pdf_tot_mock_hi = np.percentile(flux_pdf_mc, 100.0*norm.cdf(1.0), axis=0)

################ plotting
plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', lw=linewidth, c='tab:gray', alpha=0.8, label='noise')
plt.plot(flux_bins, pdf_igm_noise, drawstyle='steps-mid', lw=linewidth, c='tab:orange', label='IGM + noise')
plt.plot(flux_bins, pdf_cgm_noise, drawstyle='steps-mid', lw=linewidth, c='tab:blue', label='CGM + noise')
plt.plot(flux_bins, pdf_tot_noise, drawstyle='steps-mid',  lw=linewidth, c='tab:green', label='IGM + CGM + noise')
plt.fill_between(flux_bins, flux_pdf_tot_mock_lo, flux_pdf_tot_mock_hi, facecolor='gray', step='mid', alpha=0.5, zorder=1)
plt.plot(flux_bins, pdf_tot_noise_mask, drawstyle='steps-mid', lw=linewidth, c='k', label='IGM + CGM + noise + mask')
plt.axvline(flux_decr_cutoff, color='k', ls='--', lw=linewidth)

xlim = 1e-4
ymin, ymax = 1e-3, 3.0
plt.xscale('log')
plt.yscale('log')
plt.xlabel('1-F', fontsize=xylabel_fontsize)
plt.ylabel('PDF', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)
plt.gca().set_xlim(left=xlim)
plt.ylim([ymin, ymax])
plt.legend(fontsize=legend_fontsize, loc=2)

strong_lines = LineList('Strong', verbose=False)
wave_1548 = strong_lines['CIV 1548']['wrest']
Wfactor = ((fwhm / sampling) * u.km / u.s / const.c).decompose() * wave_1548.value
Wmin_top, Wmax_top = Wfactor * oneminf_min, Wfactor * oneminf_max  # top axis

atwin = plt.twiny()
atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}$ [$\mathrm{{\AA}}]$', fontsize=xylabel_fontsize, labelpad=10)
atwin.xaxis.tick_top()
atwin.set_xscale('log')
atwin.axis([Wmin_top, Wmax_top, ymin, ymax])
atwin.tick_params(top=True)
atwin.tick_params(axis="both", labelsize=xytick_size)

plt.savefig(savefig)
plt.show()

