import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
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

fig = plt.figure(figsize=(9, 7.5))
fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.88)

xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2
################
skewerfile = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.30_logM_9.50.fits'
params = Table.read(skewerfile, hdu=1)
skewers = Table.read(skewerfile, hdu=2)
skewers = skewers[0:10]
print(len(skewers))

logZ = -3.50
metal_ion = 'C IV'
vmin_corr = 10
vmax_corr = 2000
sampling = 3.0
dv_corr = 10

fwhm_ls = [60]#, 30, 60]
vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
divbyfactor = 1e-6

vel_mid_out = []
xi_mean_tot_out = []
label_ls = []
colorls = ['tab:black', '#7570b3', '#d95f02', '#1b9e77']

for ifwhm, fwhm_val in enumerate(fwhm_ls):
    print(len(skewers))

    if ifwhm == 0:
        # compute perfect resolution
        dv_corr = fwhm_val
        vel_mid, xi_mean_tot, _, _ = mcf.compute_xi_all(params, skewers, logZ, fwhm_val, metal_ion, vmin_corr,
                                                        vmax_corr, dv_corr, sampling=sampling, want_hires=True)
        vel_mid_out.append(vel_mid_out)
        xi_mean_tot_out.append(xi_mean_tot)
        label_ls.append('Perfect resolution')

    #dv_corr = fwhm_val
    vel_mid, xi_mean_tot, _, _ = mcf.compute_xi_all(params, skewers, logZ, fwhm_val, metal_ion, vmin_corr, vmax_corr, dv_corr, sampling=sampling)
    vel_mid_out.append(vel_mid_out)
    xi_mean_tot_out.append(xi_mean_tot)
    label_ls.append('FWHM = %d' % fwhm_val)

print("Done")

for i in range(len(vel_mid_out)):
    plt.plot(vel_mid_out[i], xi_mean_tot_out[i]/ divbyfactor, '-', linewidth=linewidth, label=label_ls[i])

vmin, vmax = 0, 1250
ymin, ymax = -0.1, 1.0

print("Done")
exit()

plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=linewidth, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
plt.text(140, 0.75*ymax, r'log(M) = %0.2f M$_{\odot}$' % 9.50 + '\n' + 'R = $%0.2f$ Mpc' % 0.30 + \
         '\n' + '[C/H] = $%0.2f$' % -3.50, fontsize=xytick_size, linespacing=1.8)
plt.legend(fontsize=legend_fontsize)
plt.xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
plt.ylabel(r'$\xi(\Delta v)$ $[10^{-6}]$', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="x", labelsize=xytick_size)
plt.gca().tick_params(axis="y", labelsize=xytick_size)
plt.xlim([vmin, vmax])
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

# Create upper axis in cMpc
cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
z = params['z'][0]
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
# Make the new upper x-axes
atwin = plt.gca().twiny()
atwin.set_xlabel('R [cMpc]', fontsize=xylabel_fontsize, labelpad=8)
atwin.xaxis.tick_top()
# atwin.yaxis.tick_right()
atwin.axis([rmin, rmax, ymin, ymax])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=xytick_size)
print("Done...")


plt.show()
