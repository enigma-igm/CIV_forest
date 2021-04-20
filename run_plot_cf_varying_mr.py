from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers
import metal_corrfunc
from enigma.reion_forest.compute_model_grid_civ import read_model_grid
import enigma.reion_forest.utils as reion_utils
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from matplotlib.ticker import AutoMinorLocator

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

plt.figure(figsize=(8.5, 6.5))
for ilogZ, logZval in enumerate(logZ_fid):
    for ilogM, logMval in enumerate(logM):
        for iR, Rval in enumerate(R):
            ximodel_ilogM = np.where(params['logM'][0] == logMval)[0][0]
            ximodel_iR = np.where(np.round(params['R_Mpc'][0],2) == Rval)[0][0]
            ximodel_ilogZ = np.where(params['logZ'][0] == logZval)[0][0]
            xi_model = xi_model_array[ximodel_ilogM][ximodel_iR][ximodel_ilogZ]

            fv, fm = halos_skewers.get_fvfm(logMval, Rval)
            logZ_eff = halos_skewers.calc_igm_Zeff(fm, logZ_fid=logZval)

            label = r'log$M$=%0.2f, $R$=%0.2f ($f_m$=%0.2f, $f_v$=%0.2f)' % (logMval, Rval, fm, fv)
            plt.plot(vel_mid, xi_model/divbyfactor, linewidth=2.0, color=colorls[ilogM], linestyle=linestyle_ls[iR], label=label)

vmin, vmax = 0, 1250
ymin, ymax = -0.1, 3.0

plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.8) #label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
plt.text(150, 0.9*ymax, r'log$Z = %0.2f$' % logZ_fid[0], fontsize='x-large')
plt.legend(fontsize=12)
plt.xlabel(r'$\Delta v$ (km/s)', fontsize=18)
plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)
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
atwin.set_xlabel('R (cMpc)', fontsize=18, labelpad=8)
atwin.xaxis.tick_top()
# atwin.yaxis.tick_right()
atwin.axis([rmin, rmax, ymin, ymax])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=13)

plt.tight_layout()
#plt.show()
plt.savefig(outfig)