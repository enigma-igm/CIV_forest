from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers

##### to run on IGM cluster where all production files are #####

maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
outfig = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/fv_fm.png'

# looking at just a subset of all models
logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.50, 0.1, 3, 0.4)

fm_all = []
fv_all = []
for i_logM, logMval in enumerate(logM):
    temp_fm = []
    temp_fv = []
    for i_R, Rval in enumerate(R):
        mask_outfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(
                                        logMval) + '.fits')
        ske = Table.read(mask_outfile, hdu=2)
        mask_arr = ske['MASK'].astype(bool)

        fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)
        temp_fm.append(fm)
        temp_fv.append(fv)
        ske = None
        mask_arr = None

    fm_all.append(temp_fm)
    fv_all.append(temp_fv)

plt.figure(figsize=(10,4))
plt.subplot(121)
for i_logM, logMval in enumerate(logM):
    plt.plot(R, fv_all[i_logM], 'o-', label='logM = %0.2f' % logMval)
    plt.xlabel('R (Mpc)', fontsize=15)
    plt.ylabel(r'$f_v$', fontsize=15)

plt.subplot(122)
for i_logM, logMval in enumerate(logM):
    plt.plot(R, fm_all[i_logM], 'o-', label='logM = %0.2f' % logMval)
    plt.xlabel('R (Mpc)', fontsize=15)
    plt.ylabel(r'$f_m$', fontsize=15)

plt.tight_layout()
plt.savefig(outfig)