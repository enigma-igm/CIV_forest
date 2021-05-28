from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import os
import halos_skewers
from matplotlib.collections import PatchCollection
import matplotlib as mpl

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

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(13, 9.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.12, bottom=0.08, right=0.98, top=0.95, wspace=0.02, hspace=0.02)

xytick_size = 16
xylabel_fontsize = 20
annotate_text_size = 13
outfig = 'paper_plots/enriched_halos_paper_new.pdf'

##########################

slice_thickness = 1.0 # Mpc
Zc = 50
logM_min_ls = [9.0, 10.0, 11.0]
R_ls = [0.1, 0.5, 1.5, 3.0] # [0.5, 1.5, 2.7]

halofile = 'nyx_sim_data/z45_halo_logMmin_8.fits'
halos = Table.read(halofile)
fvfm_master = Table.read('nyx_sim_data/igm_cluster/enrichment_models/fvfm_all.fits')
baryon_slice = 'bslice_Z_50.npy'
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
ax_ls = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

for ilogM, logMval in enumerate(logM_min_ls):
    for iR, Rval in enumerate(R_ls):
        #skewerfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')
        #ske = Table.read(skewerfile, hdu=2)
        #mask_arr = ske['MASK'].astype(bool)
        #fm, fv = halos_skewers.calc_fm_fv(mask_arr, ske)
        fv, fm = halos_skewers.get_fvfm(np.round(logMval, 2), np.round(Rval, 2))

        subplot_counter +=1
        #plt.subplot(nrow, ncol, subplot_counter)

        if Rval == 0.1:
            alpha = 1.0
            ec = 'r'
            vmax = 7

        elif Rval == 0.5:
            alpha = 0.5
            ec = None
            vmax = 7
        elif logMval == 9.0:
            ec = None
            if Rval == 1.5:
                alpha = 0.09
                vmax = 7
            elif Rval == 3.0:
                alpha = 0.07
                vmax = 5
        elif logMval == 10.0:
            ec = None
            if Rval == 1.5:
                alpha = 0.15
                vmax = 7
            elif Rval == 3.0:
                alpha = 0.08
                vmax = 7
        else:
            alpha = 0.35
            ec = None
            vmax = 7

        slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax) * (np.log10(halos['MASS']) >= logMval)
        halo_slice = halos[slice]
        allcirc = []
        for ihalo in halo_slice:
            circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), Rval, color='r', fill=True, alpha=alpha, ec=ec)
            allcirc.append(circ)
        coll = PatchCollection(allcirc, match_original=True)
        #plt.gca().add_collection(coll)
        #plt.imshow(np.transpose(bs), vmin=0, vmax=7, cmap='gray_r', origin='lower', extent=[0, Lbox, 0, Lbox])
        ax_want = ax_ls[subplot_counter-1]
        ax_want.add_collection(coll)
        ax_want.imshow(np.transpose(bs), vmin=0, vmax=vmax, cmap='gray_r', origin='lower', extent=[0, Lbox, 0, Lbox])

        text = r'$f_v = %0.3f$' % fv + '\n' + r'$f_m = %0.3f$' % fm
        ax_want.text(20, 54, text, fontsize=13, bbox=dict(facecolor='white'))
        #ax_want.set_xlim([20, 60])
        #ax_want.set_ylim([20, 60])
        ax_want.set_xlim([18, 62])
        ax_want.set_ylim([18, 62])
        ax_want.set_aspect('equal')

        if subplot_counter <= ncol:
            ax_want.set_title('R = %0.2f Mpc' % Rval, fontsize=xylabel_fontsize)
        if subplot_counter > (tot_subplot - ncol):
            ax_want.set_xlabel('X [Mpc]', fontsize=xylabel_fontsize)
            ax_want.tick_params(axis="x", labelsize=xytick_size)
        else:
            ax_want.axes.xaxis.set_visible(False)

        if subplot_counter % ncol == 1:
            ylabel = r'log(M/M$_{\odot}$)' + '={:4.2f}'.format(logMval) + '\n\nY [Mpc]'
            ax_want.set_ylabel(ylabel, fontsize=xylabel_fontsize)
            ax_want.tick_params(axis="y", labelsize=xytick_size)
        else:
            ax_want.axes.yaxis.set_visible(False)

plt.savefig(outfig)
#plt.show()