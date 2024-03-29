from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import metal_corrfunc
import halos_skewers
from enigma.reion_forest.compute_model_grid_civ import read_model_grid
import enigma.reion_forest.utils as reion_utils
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from matplotlib.ticker import AutoMinorLocator

### Figure settings
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

fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.9)

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2

outfig = 'paper_plots/cf_varying_MR.pdf'
modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)

logM = [9.5, 10.5]
R = [0.5, 1.5, 2.5]
logZ_fid = [-3.50]
vel_mid = params['vel_mid'][0]

colorls = ['#1b9e77','#d95f02']
linestyle_ls = [':', '--', '-']
vel_doublet = reion_utils.vel_metal_doublet('C IV', returnVerbose=False)
divbyfactor = 1e-5

for ilogZ, logZval in enumerate(logZ_fid):
    for ilogM, logMval in enumerate(logM):
        for iR, Rval in enumerate(R):
            ximodel_ilogM = np.where(params['logM'][0] == logMval)[0][0]
            ximodel_iR = np.where(np.round(params['R_Mpc'][0],2) == Rval)[0][0]
            ximodel_ilogZ = np.where(params['logZ'][0] == logZval)[0][0]
            xi_model = xi_model_array[ximodel_ilogM][ximodel_iR][ximodel_ilogZ]

            fv, fm = halos_skewers.get_fvfm(logMval, Rval)
            logZ_eff = halos_skewers.calc_igm_Zeff(fm, logZ_fid=logZval)

            label = r'log(M)=%0.2f, R=%0.2f (f$_\mathrm{m}$=%0.2f, f$_\mathrm{v}$=%0.2f)' % (logMval, Rval, fm, fv)
            plt.plot(vel_mid, xi_model/divbyfactor, linewidth=linewidth, color=colorls[ilogM], linestyle=linestyle_ls[iR], label=label)

vmin, vmax = 0, 1250
ymin, ymax = -0.1, 2.1

plt.axvline(vel_doublet.value, color='black', linestyle=':', linewidth=linewidth, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
plt.text(150, 0.9*ymax, '[C/H] = $%0.2f$' % logZ_fid[0], fontsize=xytick_size)
plt.legend(fontsize=legend_fontsize)
plt.xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)
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

#plt.savefig(outfig)
plt.show()
plt.close()