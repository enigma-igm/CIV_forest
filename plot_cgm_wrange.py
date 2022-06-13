# breaking down CGM models into wranges, ala plots/cgm_pdf_wrange_alpha_neg0.65.png
# fixed CGM model
# fixed IGM model (choose a fiducial model)

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
from linetools.lists.linelist import LineList
from astropy import units as u
from astropy import constants as const
import civ_cgm # new version
import halos_skewers
import time

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

plt.figure(figsize=(11, 9))
plt.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.89)

xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 23 #20
legend_fontsize = 18 #14
linewidth = 2
alpha = 0.75

savefig = 'paper_plots/flux_pdf_wrange_refreport.pdf'
logZ = -3.5 # (9/13/21) input metallicity for non-uniform model; logZ for uniform model defined below
metal_ion = 'C IV'
fwhm = 10
snr = 50
sampling = 3.0
seed = 3429381 # random seeds for drawing CGM absorbers
rand = np.random.RandomState(seed)

cgm_alpha = -1.1
cgm_n_star = 5
metal_dndz_func = civ_cgm.civ_dndz_sch
nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0 # gives d(oneminf) = 0.01

def init_one_model(skewerfile, Wmin, Wmax, logZ):
    par = Table.read(skewerfile, hdu=1)
    ske = Table.read(skewerfile, hdu=2)
    z = par['z'][0]

    cgm_model = civ_cgm.init_metal_cgm_dict(alpha=cgm_alpha, n_star=cgm_n_star, W_min=Wmin, W_max=Wmax) # rest are default

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

    # no noise
    flux_bins, pdf_igm = reion_utils.pdf_calc(1.0 - flux_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm = reion_utils.pdf_calc(1.0 - flux_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot = reion_utils.pdf_calc(1.0 - flux_tot_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_noise = reion_utils.pdf_calc(-noise, oneminf_min, oneminf_max, nbins) # flux_noise  = 1 + noise
                                                                                 # 1 - flux_noise = 1 - (1 + noise) = -noise

    # with noise
    _, pdf_igm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_igm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_cgm_noise, = reion_utils.pdf_calc(1.0 - flux_noise_cgm_lores, oneminf_min, oneminf_max, nbins)
    _, pdf_tot_noise, = reion_utils.pdf_calc(1.0 - flux_noise_tot_lores, oneminf_min, oneminf_max, nbins)

    out_pdf_no_noise = (pdf_igm, pdf_cgm, pdf_tot, pdf_noise)
    out_pdf_with_noise = (pdf_igm_noise, pdf_cgm_noise, pdf_tot_noise)

    return flux_bins, out_pdf_no_noise, out_pdf_with_noise


##### plot uniform IGM as reference
Wmin, Wmax = 0.001, 0.05 # placeholder values, since we just want the uniform IGM pdf
skewerfile = 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits' # uniformly enriched
uniform_logZ = -3.97 # (9/13/21) uniform logZ = effective logZ for non-uniform model
flux_bins, out_pdf_no_noise, out_pdf_with_noise = init_one_model(skewerfile, Wmin, Wmax, uniform_logZ)
pdf_igm, pdf_cgm, pdf_tot, pdf_noise = out_pdf_no_noise

plt.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label='IGM, uniform, [C/H]=${:5.2f}$'.format(uniform_logZ), lw=linewidth + 0.5, c='tab:red', alpha=alpha)

##### plot fiducial IGM model
skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
#text = 'log(M)={:4.2f} '.format(logM) + r'M$_{\odot}$, ' + '\nR={:4.2f} Mpc, '.format(R_Mpc) + '\n[C/H]=${:5.2f}$'.format(logZ)
text = 'log(M)={:4.2f}'.format(logM) + r' $M_{\odot}$, ' + 'R={:4.2f} cMpc, '.format(R_Mpc) + '[C/H]=${:5.2f}$'.format(logZ) + '; [C/H]$_{\mathrm{eff}}$' + '=${:5.2f}$'.format(uniform_logZ)

W_range_ls = [[0.001, 5.0], [0.001, 0.05], [0.05, 0.5], [0.5, 5.0]]
#W_range_ls = [[0.001, 5.0], [0.5, 5.0],  [0.1, 0.5], [0.01, 0.1], [0.001, 0.01]]
alpha_ls = np.linspace(1, 0.4, len(W_range_ls))

for iW, W_range in enumerate(W_range_ls):
    Wmin, Wmax = W_range[0], W_range[1]
    flux_bins, out_pdf_no_noise, out_pdf_with_noise = init_one_model(skewerfile, Wmin, Wmax, logZ)
    pdf_igm, pdf_cgm, pdf_tot, pdf_noise = out_pdf_no_noise

    if iW == 0: # entire W-range
        plt.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label='IGM, %s' % text, lw=linewidth + 0.5, c='tab:orange')
        plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM', lw=linewidth + 0.5, c='tab:blue')
        #plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise', lw=linewidth + 0.5, c='tab:gray', alpha=alpha)

    else: # sub W-range
        if iW == 1:
            #label = r'CGM, $W$ = ' + '[{:5.3f}'.format(Wmin) + r'$-$' + '{:4.2f}]'.format(Wmax) + r' $\mathrm{{\AA}}$'
            label = r'$W$ = ' + '[{:5.3f}'.format(Wmin) + r'$-$' + '{:4.2f}]'.format(Wmax) + r' $\mathrm{{\AA}}$'
        else:
            label = r'$W$ = ' + '[{:4.2f}'.format(Wmin) + r'$-$' + '{:4.2f}]'.format(Wmax) + r' $\mathrm{{\AA}}$'
        plt.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label=label, lw=linewidth, alpha=alpha_ls[iW], c='tab:blue')

plt.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise', lw=linewidth + 0.5, c='tab:gray', alpha=alpha)

#plt.gca().annotate('log(M)={:5.2f} '.format(logM) + r'M$_{\odot}$, ' + '\nR={:5.2f} Mpc, '.format(R_Mpc) + '\n[C/H]=${:5.2f}$'.format(logZ), \
#             xy=(2.5e-3, 1.0), xytext=(2.5e-3, 1.0), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize)
#plt.text(0.07, 0.83, text, fontsize=legend_fontsize, linespacing=1.5)

ymin, ymax = 1e-3, 4.0 #3.0
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'1$-$F', fontsize=xylabel_fontsize)
plt.ylabel('PDF', fontsize=xylabel_fontsize)
plt.ylim([ymin, ymax])
plt.legend(loc=2, fontsize=legend_fontsize, frameon=False)
plt.gca().tick_params(axis="both", labelsize=xytick_size)

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





