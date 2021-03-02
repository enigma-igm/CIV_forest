from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers

# to run on IGM cluster where all production files are

slice_thickness = 1.0
Zc = 50
logM_min_ls = [8.5, 9.0, 10.0, 11.0]
R_ls = [0.1, 0.5, 1.7, 2.3]

logM_min_ls = [8.5, 9.0]
R_ls = [0.1, 0.5]

halofile = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/z45_halo_logMmin_8.fits'
halos = Table.read(halofile)
maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
outfig = '/home/sstie/CIV_forest/outfig.png'

Zmin = Zc - slice_thickness/2.
Zmax = Zc + slice_thickness/2.
slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax)
halo_slice = halos[slice]

plt.figure(figsize=(12,12))
nrow = len(logM_min_ls)
ncol = len(R_ls)
subplot_counter = 0

for ilogM, logMval in enumerate(logM_min_ls):
    for iR, Rval in enumerate(R_ls):
        skewerfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')
        ske = Table.read(skewerfile, hdu=2)
        mask_arr = ske['MASK'].astype(bool)
        fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)

        subplot_counter +=1
        plt.subplot(nrow, ncol, subplot_counter)
        plt.plot(halo_slice['XHALO'], halo_slice['YHALO'], 'k.', ms=3, alpha=0.7)

        for ihalo in halo_slice:
            if np.log10(ihalo['MASS']) > logMval:
                circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), Rval, color='r', fill=True)
                plt.gca().add_patch(circ)

        text = r'$f_v = %0.3f, f_m = %0.3f$' % (fv, fm)
        plt.text(10, 120, text, fontsize=12)

plt.tight_layout()
plt.savefig(outfig)