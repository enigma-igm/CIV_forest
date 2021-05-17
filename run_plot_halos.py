from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers

##### to run on IGM cluster where all production files are #####

slice_thickness = 1.0 # Mpc
Zc = 50
logM_min_ls = [9.0, 10.0, 11.0]
R_ls = [0.1, 0.5, 1.5, 3.0] # [0.5, 1.5, 2.7]

halofile = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/z45_halo_logMmin_8.fits'
halos = Table.read(halofile)
maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
outfig = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/paper_plots/enriched_halos_paper.pdf'
baryon_slice = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/bslice_Z_50.npy'
bs = np.load(baryon_slice)
Lbox = 145.985401459854 # Mpc

Zmin = Zc - slice_thickness/2.
Zmax = Zc + slice_thickness/2.
slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax)
halo_slice = halos[slice]

nrow = len(logM_min_ls)
ncol = len(R_ls)
tot_subplot = nrow*ncol
subplot_counter = 0

plt.figure(figsize=(13,9.5))
#plt.subplots_adjust(left=0.067, bottom=0.067, right=1-0.067, top=1-0.067, wspace=0.05, hspace=0.05)
for ilogM, logMval in enumerate(logM_min_ls):
    for iR, Rval in enumerate(R_ls):
        skewerfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')
        ske = Table.read(skewerfile, hdu=2)
        mask_arr = ske['MASK'].astype(bool)
        fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)

        subplot_counter +=1
        plt.subplot(nrow, ncol, subplot_counter)
        """
        plt.plot(halo_slice['XHALO'], halo_slice['YHALO'], 'k.', ms=3, alpha=0.7)
        for ihalo in halo_slice:
            if np.log10(ihalo['MASS']) > logMval:
                circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), Rval, color='r', fill=True)
                plt.gca().add_patch(circ)
        """

        slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax) * (np.log10(halos['MASS']) >= logMval)
        halo_slice = halos[slice]
        for ihalo in halo_slice:
            circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), Rval, color='r', fill=True, alpha=0.65)
            plt.gca().add_patch(circ)
        plt.imshow(np.transpose(bs), vmin=0, vmax=10, cmap='gray_r', origin='lower', extent=[0, Lbox, 0, Lbox])

        text = r'$f_v = %0.3f$' % fv + '\n' + r'$f_m = %0.3f$' % fm
        plt.text(22, 52, text, fontsize=13, bbox=dict(facecolor='white'))
        plt.xlim([20, 60])
        plt.ylim([20, 60])
        plt.gca().set_aspect('equal')

        if subplot_counter <= ncol:
            plt.title('R = %0.2f Mpc' % Rval, fontsize=18)
        if subplot_counter > (tot_subplot - ncol):
            plt.xlabel('X (Mpc)', fontsize=18)
            plt.gca().tick_params(axis="x", labelsize=13)
        else:
            plt.gca().axes.xaxis.set_visible(False)

        if subplot_counter % ncol == 1:
            plt.ylabel(r'log$_{10}(M/M_{\odot})$ = %0.2f' % logMval + '\n\n Y (Mpc)', fontsize=18)
            plt.gca().tick_params(axis="y", labelsize=13)
        else:
            plt.gca().axes.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig(outfig)
#plt.show()