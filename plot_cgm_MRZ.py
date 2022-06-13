# fix CGM model and vary IGM logM and R (at fixed logZ), which is effectively varying the filling factor

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
import matplotlib.style as style
style.use('tableau-colorblind10')

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7.5), sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.1, right=0.98, top=0.89, wspace=0)

xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 23 #20
legend_fontsize = 18 #14
linewidth = 2
alpha = 0.6

savefig = 'paper_plots/flux_pdf_MRZ_refreport.pdf'
metal_ion = 'C IV'
fwhm = 10
snr = 50
sampling = 3.0
seed = 3429381 # random seeds for drawing CGM absorbers
rand = np.random.RandomState(seed)

cgm_alpha = -1.1
cgm_n_star = 5
cgm_model = civ_cgm.init_metal_cgm_dict(alpha=cgm_alpha, n_star=cgm_n_star) # rest are default (W_min=0.001, W_max=5.0, W_star=0.45)
metal_dndz_func = civ_cgm.civ_dndz_sch
nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0 # gives d(oneminf) = 0.01

def init_one_model(skewerfile, logZ_in):

    par = Table.read(skewerfile, hdu=1)
    ske = Table.read(skewerfile, hdu=2)
    z = par['z'][0]

    start = time.time()
    v_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), v_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), \
    (oden, v_los, T, x_metal), cgm_tup = reion_utils.create_metal_forest(par, ske, logZ_in, fwhm, metal_ion, z=z, \
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


fvfm_master = 'nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits'
strong_lines = LineList('Strong', verbose=False)
wave_1548 = strong_lines['CIV 1548']['wrest']
Wfactor = ((fwhm / sampling) * u.km / u.s / const.c).decompose() * wave_1548.value
Wmin_top, Wmax_top = Wfactor * oneminf_min, Wfactor * oneminf_max  # top axis
ymin, ymax = 1e-3, 4.0

############### varying logM plot ###############
logZ = -3.5
file1 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_8.50.fits' # (9/13/21) added
file2 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
file3 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_10.50.fits'
skewerfile_ls = [file1, file2, file3]
igm_color_ls = ['tab:red', 'tab:orange', 'gold']

for ifile, skewerfile in enumerate(skewerfile_ls):
    flux_bins, out_pdf_no_noise, out_pdf_with_noise = init_one_model(skewerfile, logZ)
    pdf_igm, pdf_cgm, pdf_tot, pdf_noise = out_pdf_no_noise

    logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
    R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
    fv_want, fm_want = halos_skewers.get_fvfm(logM, R_Mpc, fvfm_file=fvfm_master)
    logZ_eff_want = halos_skewers.calc_igm_Zeff(fm_want, logZ_fid=logZ)

    igm_label = 'IGM (logM={:5.2f}'.format(logM) + 'M$_{\odot}$, ' + r'[C/H]$_{\mathrm{eff}}$=' + '${:5.2f}$)'.format(logZ_eff_want)
    ax1.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label=igm_label, lw=linewidth, color=igm_color_ls[ifile])

    if ifile == 2:
        ax1.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM', lw=linewidth, c='tab:blue', alpha=alpha)
        ax1.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise', lw=linewidth, c='tab:gray', alpha=alpha)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('1$-$F', fontsize=xylabel_fontsize)
ax1.set_ylabel('PDF', fontsize=xylabel_fontsize)
ax1.set_ylim([ymin, ymax])
ax1.legend(loc=2, fontsize=legend_fontsize, frameon=False)
ax1.tick_params(axis="both", labelsize=xytick_size)

atwin = ax1.twiny()
atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}$ [$\mathrm{{\AA}}]$', fontsize=xylabel_fontsize, labelpad=10)
atwin.xaxis.tick_top()
atwin.set_xscale('log')
atwin.axis([Wmin_top, Wmax_top, ymin, ymax])
atwin.tick_params(top=True)
atwin.tick_params(axis="both", labelsize=xytick_size)

############### varying R plot ###############
logZ = -3.5
file1 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.30_logM_9.50.fits'
file2 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
file3 = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_2.00_logM_9.50.fits'
skewerfile_ls = [file1, file2, file3]
igm_color_ls = ['gold', 'tab:orange', 'tab:red']

for ifile, skewerfile in enumerate(skewerfile_ls):
    flux_bins, out_pdf_no_noise, out_pdf_with_noise = init_one_model(skewerfile, logZ)
    pdf_igm, pdf_cgm, pdf_tot, pdf_noise = out_pdf_no_noise

    logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
    R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
    fv_want, fm_want = halos_skewers.get_fvfm(logM, R_Mpc, fvfm_file=fvfm_master)
    logZ_eff_want = halos_skewers.calc_igm_Zeff(fm_want, logZ_fid=logZ)

    igm_label = 'IGM (R={:4.2f} cMpc, '.format(R_Mpc) + r'[C/H]$_{\mathrm{eff}}$=' + '${:5.2f}$)'.format(logZ_eff_want)
    ax2.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label=igm_label, lw=linewidth, color=igm_color_ls[ifile])

    if ifile == 2:
        ax2.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM', lw=linewidth, c='tab:blue', alpha=alpha)
        ax2.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise', lw=linewidth, c='tab:gray', alpha=alpha)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('1$-$F', fontsize=xylabel_fontsize)
ax2.set_ylim([ymin, ymax])
ax2.legend(loc=2, fontsize=legend_fontsize, frameon=False)
ax2.tick_params(axis="both", labelsize=xytick_size)

atwin = ax2.twiny()
atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}$ [$\mathrm{{\AA}}]$', fontsize=xylabel_fontsize, labelpad=10)
atwin.xaxis.tick_top()
atwin.set_xscale('log')
atwin.axis([Wmin_top, Wmax_top, ymin, ymax])
atwin.tick_params(top=True)
atwin.tick_params(axis="both", labelsize=xytick_size)

############### varying logZ plot ###############
logZ_ls = [-4.0, -3.5, -3.0]
skewerfile = 'nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.80_logM_9.50.fits'
igm_color_ls = ['gold', 'tab:orange', 'tab:red']

for ilogZ, logZ in enumerate(logZ_ls):
    flux_bins, out_pdf_no_noise, out_pdf_with_noise = init_one_model(skewerfile, logZ)
    pdf_igm, pdf_cgm, pdf_tot, pdf_noise = out_pdf_no_noise

    logM = float(skewerfile.split('logM_')[-1].split('.fits')[0])
    R_Mpc = float(skewerfile.split('R_')[-1].split('_logM')[0])
    fv_want, fm_want = halos_skewers.get_fvfm(logM, R_Mpc, fvfm_file=fvfm_master)
    logZ_eff_want = halos_skewers.calc_igm_Zeff(fm_want, logZ_fid=logZ)

    igm_label = 'IGM ([C/H]$_{\mathrm{in}}$=' + '${:5.2f}$)'.format(logZ) + r', [C/H]$_{\mathrm{eff}}$=' + '${:5.2f}$)'.format(logZ_eff_want)
    ax3.plot(flux_bins, pdf_igm, drawstyle='steps-mid', label=igm_label, lw=linewidth, color=igm_color_ls[ilogZ])

    if ilogZ == 2:
        ax3.plot(flux_bins, pdf_cgm, drawstyle='steps-mid', label='CGM', lw=linewidth, c='tab:blue', alpha=alpha)
        ax3.plot(flux_bins, pdf_noise, drawstyle='steps-mid', label='noise', lw=linewidth, c='tab:gray', alpha=alpha)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('1$-$F', fontsize=xylabel_fontsize)
ax3.set_ylim([ymin, ymax])
ax3.legend(loc=2, fontsize=legend_fontsize, frameon=False)
ax3.tick_params(axis="both", labelsize=xytick_size)

atwin = ax3.twiny()
atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}$ [$\mathrm{{\AA}}]$', fontsize=xylabel_fontsize, labelpad=10)
atwin.xaxis.tick_top()
atwin.set_xscale('log')
atwin.axis([Wmin_top, Wmax_top, ymin, ymax])
atwin.tick_params(top=True)
atwin.tick_params(axis="both", labelsize=xytick_size)

"""
#text = 'log(M)={:5.2f} '.format(logM)
#plt.text(0.07, 1.7, text, fontsize=legend_fontsize, linespacing=1.5)

ymin, ymax = 1e-3, 3.0
plt.xscale('log')
plt.yscale('log')
plt.xlabel('1-F', fontsize=xylabel_fontsize)
plt.ylabel('PDF', fontsize=xylabel_fontsize)
plt.ylim([ymin, ymax])
plt.legend(loc=2, fontsize=legend_fontsize)
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
"""

plt.savefig(savefig)
plt.show()
