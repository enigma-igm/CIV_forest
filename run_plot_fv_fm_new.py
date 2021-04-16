from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers

##### to run on IGM cluster where all production files are #####
maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
outfig = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/fvfm_all.png'

# looking at just a subset of all models
logM = np.array([8.50, 9.00, 9.50, 10.00, 10.50, 11.00])
R = np.array([0.1, 0.50, 1.50, 3.00])

fm_vs_R = []
fv_vs_R = []
fm_vs_logM = []
fv_vs_logM = []

# assemble things for plotting vs R
for i_logM, logMval in enumerate(logM):
    temp_fm = []
    temp_fv = []
    for i_R, Rval in enumerate(R):
        mask_outfile = os.path.join(maskpath,
                                    'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(
                                        logMval) + '.fits')
        ske = Table.read(mask_outfile, hdu=2)
        mask_arr = ske['MASK'].astype(bool)

        fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)
        temp_fm.append(fm)
        temp_fv.append(fv)
        ske = None
        mask_arr = None

    fm_vs_R.append(temp_fm)
    fv_vs_R.append(temp_fv)

# assemble things for plotting vs logM
for i_R, Rval in enumerate(R):
    temp_fm = []
    temp_fv = []
    for i_logM, logMval in enumerate(logM):
        mask_outfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')
        ske = Table.read(mask_outfile, hdu=2)
        mask_arr = ske['MASK'].astype(bool)

        fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)
        temp_fm.append(fm)
        temp_fv.append(fv)
        ske = None
        mask_arr = None

    fm_vs_logM.append(temp_fm)
    fv_vs_logM.append(temp_fv)

# now let's plot things
plt.figure(figsize=(12,5))
plt.subplot(121)
for i_logM, logMval in enumerate(logM):
    plt.plot(R, fv_vs_R[i_logM], 'o-', label=r'log$_{10}M$ = %0.2f' % logMval)
for i_logM, logMval in enumerate(logM):
    plt.plot(R, fm_vs_R[i_logM], 'o--')
plt.legend(fontsize=13)
plt.xlabel('R (Mpc)', fontsize=18)
plt.ylabel('Filling factor', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=13)
plt.gca().tick_params(axis="y", labelsize=13)

plt.subplot(122)
for i_R, Rval in enumerate(R):
    plt.plot(logM, fv_vs_logM[i_R], 'o-', label='R = %0.2f Mpc' % Rval)
for i_R, Rval in enumerate(R):
    plt.plot(logM, fm_vs_logM[i_R], 'o--')
plt.legend(fontsize=13)
plt.xlabel(r'log$_{10}M$', fontsize=18)
plt.gca().axes.yaxis.set_visible(False)
plt.gca().tick_params(axis="x", labelsize=13)

plt.tight_layout()
plt.show()
#plt.savefig(outfig)