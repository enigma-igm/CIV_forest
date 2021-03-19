'''
Functions here:
    - init_all
    - init_halo_grids
    - plot_halos
    - calc_distance_one_skewer
    - calc_distance_all_skewers
    - write_iz_mask
    - plot_halos_with_skewers
    - calc_fm_fv
    - calc_igm_Zeff
    - calc_fvfm_all
    - check_halo_xyz
    - check_skewers_xyz
    - common_cell
    - plot_common_cell
'''

import time
import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils
from scipy.spatial import distance # v14.0 syntax
from linetools.abund import solar as labsol
import os

def init_all(halofile='nyx_sim_data/z45_halo_logMmin_8.fits', skewerfile='nyx_sim_data/rand_skewers_z45_ovt_tau.fits'):
    par = Table.read(skewerfile, hdu=1)
    ske = Table.read(skewerfile, hdu=2)
    halos = Table.read(halofile)

    return par, ske, halos

def init_halo_grids(logMmin=8.5, logMmax=11.0, dlogM=0.5, Rmin=0.1, Rmax=3.0, dR=0.5):

    # Booth et al (2012): logM=8 to logM=11.0 in 0.5 dex and R=31.25 proper kpc to R=500 proper kpc, in factor of 2
    # r_pmpc_booth = np.array([0.03125, 0.0625 , 0.125  , 0.25   , 0.5    ])
    # r_cmpc_booth = (1+4.5) * r_pmpc_booth
    # r_cmpc_booth = array([0.171875, 0.34375 , 0.6875  , 1.375   , 2.75    ])

    # default logMmin=8.5 since that is the minimum mass of the halo catalog

    logM_grid = np.arange(logMmin, logMmax + dlogM, dlogM)
    R_grid = np.arange(Rmin, Rmax + dR, dR) # cMpc physical

    return logM_grid, R_grid

def plot_halos(halos, slice_thickness, Zc, logM_min=8.0):
    # halos: Astropy table
    # slice thickness: in Mpc

    imass = np.where(np.log10(halos['MASS']) >= logM_min)[0]
    halos = halos[imass]
    print("after mass cut, N(halos): ", len(halos))

    Zmin = Zc - slice_thickness/2.
    Zmax = Zc + slice_thickness/2.
    slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax)

    plt.plot(halos['XHALO'][slice], halos['YHALO'][slice], '.', ms=5, alpha = 0.5, label=logM_min)
    plt.axis('equal')
    plt.xlabel('XHALO', fontsize=15)
    plt.ylabel('YHALO', fontsize=15)
    plt.axis('equal')
    plt.legend()

def calc_distance_one_skewer(one_skewer, params, halos, Rmax, logM_min):
    # halos: list of halos; see init_all()
    # distance computation includes periodic BC

    start = time.time()
    mass_mask = np.log10(halos['MASS']) >= logM_min
    halos = halos[mass_mask]

    lit_h = params['lit_h'][0]
    Lbox = params['Lbox'][0] / lit_h # Mpc unit
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng # Mpc unit

    # Mpc units
    xskew = one_skewer['XSKEW']
    yskew = one_skewer['YSKEW']
    zskew = np.arange(Ng) * cellsize

    # Mpc units
    xhalos = np.array(halos['XHALO'])
    yhalos = np.array(halos['YHALO'])
    zhalos = np.array(halos['ZHALO'])

    dx = xskew - xhalos
    dy = yskew - yhalos

    # periodic BC
    mask_x = np.abs(dx) > (0.5 * Lbox)
    mask_y = np.abs(dy) > (0.5 * Lbox)
    dx[mask_x] = Lbox - np.abs(dx[mask_x])
    dy[mask_y] = Lbox - np.abs(dy[mask_y])

    # first cut to trim down the number of halos, based on 2D position
    dx2dy2 = dx ** 2 + dy ** 2
    want_halos = np.where(dx2dy2 <= Rmax ** 2)[0]
    dx2dy2 = dx2dy2[want_halos]

    # this is slower...
    #mask_halo_2d = dx2dy2 <= Rmax**2
    #dx2dy2 = dx2dy2[mask_halo_2d]

    zpix_near_halo = []
    for zpix in zskew: # looping through each pixel along the z-direction
        dz = zpix - zhalos[want_halos]
        mask_z = np.abs(dz) > (0.5 * Lbox)
        dz[mask_z] = Lbox - np.abs(dz[mask_z])
        r2 = dx2dy2 + dz**2 # =len(halos)

        if np.sum(r2 < Rmax**2):
            zpix_near_halo.append(True)
        else:
            zpix_near_halo.append(False)

    end = time.time()
    #print((end-start)/60.)

    return zpix_near_halo # mask array

def calc_distance_all_skewers(params, skewers, halos, Rmax, logM_min):
    # 0.17 min (0.3 min) for 100 skewers at Rmax=0.2 Mpc (2.5 Mpc)
    print("Doing (R, logM)", Rmax, logM_min)
    start = time.time()
    all_iz_near_halo = []
    for iskew in skewers:
        iz_near_halo = calc_distance_one_skewer(iskew, params, halos, Rmax, logM_min)
        all_iz_near_halo.append(iz_near_halo)

    all_iz_near_halo = np.array(all_iz_near_halo)
    end = time.time()
    print((end - start) / 60.)

    return all_iz_near_halo

def write_iz_mask(params, skewers, all_iz_near_halo, outfile):
    # testing only, not used for production run

    skewers['ZPIX_NEAR_HALO'] = all_iz_near_halo

    hdu_param = fits.BinTableHDU(params.as_array())
    hdu_table = fits.BinTableHDU(skewers.as_array())
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)

def plot_halos_with_skewers(params, skewers, halos, slice_thickness, Zc, logM_min):

    Lbox = params['Lbox'][0]
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng
    zskew = np.arange(Ng) * cellsize

    if logM_min != None:
        imass = np.where(np.log10(halos['MASS']) >= logM_min)[0]
        halos = halos[imass]
        print("after mass cut, N(halos): ", len(halos))

    Zmin = Zc - slice_thickness / 2.
    Zmax = Zc + slice_thickness / 2.
    halo_slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax)
    skewer_slice = (zskew >= Zmin) * (zskew < Zmax)

    plt.plot(halos['XHALO'][halo_slice], halos['YHALO'][halo_slice], '.', ms=5, alpha=0.5, label=logM_min)
    for iskew in skewers:
        if np.sum(iskew['ZPIX_NEAR_HALO'][skewer_slice]):
            plt.plot(iskew['XSKEW'], iskew['YSKEW'], 'y*', ms=5)
        #else:
            #plt.plot(iskew['XSKEW'], iskew['YSKEW'], 'r*', ms=5)
    plt.legend()
    plt.show()

def calc_fm_fv(mask_arr, skewers):
    # calculates the mass- and volume-filling fraction of the enriched regions

    mask1d = mask_arr.flatten()
    fv = np.sum(mask1d)/len(mask1d)
    fm = np.sum(skewers['ODEN'][mask_arr]) / np.sum(skewers['ODEN'])

    return fm, fv

def calc_igm_Zeff(fm, logZ_fid=-3.5):
    # calculates effective metallicity

    sol = labsol.SolarAbund()
    logZ_sol = sol.get_ratio('C/H') # same as sol['C'] - 12.0
    nC_nH_sol = 10**(logZ_sol)

    nH_bar = 3.1315263992114194e-05 # from skewerfile
    Z_fid = 10 ** (logZ_fid)
    nC_nH_fid = Z_fid * nC_nH_sol
    nC = nH_bar * nC_nH_fid * fm

    logZ_eff = np.log10(nC / nH_bar) - logZ_sol
    logZ_jfh = np.log10(10**(logZ_fid) * fm)

    return logZ_eff

def calc_fvfm_all():
    # calculates the volume- and mass-filling fraction
    # requires enrichment models stored on IGM machine.
    # outputs the table of fv and fm as fits file.

    maskpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/xciv_mask/'
    outfile = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/fvfm_all.fits'

    logM, R = init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

    fm_all = []
    fv_all = []
    logM_all = []
    R_all = []

    for i_logM, logMval in enumerate(logM):
        for i_R, Rval in enumerate(R):
            mask_outfile = os.path.join(maskpath, 'rand_skewers_z45_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(
                logMval) + '.fits')
            ske = Table.read(mask_outfile, hdu=2)
            mask_arr = ske['MASK'].astype(bool)

            fm, fv = calc_fm_fv(mask_arr, ske)
            fm_all.append(fm)
            fv_all.append(fv)
            logM_all.append(logMval)
            R_all.append(Rval)

    outtable = Table([logM_all, R_all, fv_all, fm_all], names=('logM', 'R_Mpc', 'fv', 'fm'))
    hdu_table = fits.table_to_hdu(outtable)
    hdulist = fits.HDUList()
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)

def get_fvfm(logM_want, R_want, fvfm_file='nyx_sim_data/igm_cluster/enrichment/fvfm_all.fits'):
    fvfm = Table.read(fvfm_file)
    logM_all = np.array(fvfm['logM'])
    R_all = np.round(np.array(fvfm['R_Mpc']), 2)
    #i = np.where(logM_all == logM_want)[0]
    #j = np.where(R_all == R_want)[0]
    k = np.where((logM_all == logM_want) & (R_all == R_want))[0]
    fv_want = (fvfm['fv'][k])[0] # the [0] is just to extract the value from astropy column
    fm_want = (fvfm['fm'][k])[0]

    return fv_want, fm_want

def get_logM_R(fm_lower, fm_upper, logZ_fid, logM_grid, R_grid, fvfm_file='nyx_sim_data/igm_cluster/enrichment/fvfm_all.fits'):

    logZ_eff_lower = calc_igm_Zeff(fm_lower, logZ_fid)
    logZ_eff_upper = calc_igm_Zeff(fm_upper, logZ_fid)
    print('fm_lower %0.2f corresponds to logZ_eff_lower %0.3f' % (fm_lower, logZ_eff_lower))
    print('fm_upper %0.2f corresponds to logZ_eff_upper %0.3f' % (fm_upper, logZ_eff_upper))

    fvfm = Table.read(fvfm_file)
    logM_all = np.array(fvfm['logM'])
    R_all = np.round(np.array(fvfm['R_Mpc']), 2)

    i = np.where((fvfm['fm'] <= fm_upper) & (fvfm['fm'] >= fm_lower))[0]
    logM_want = logM_all[i]
    R_want = R_all[i]

    ilogM_want = np.array([np.where(logM_grid == m)[0][0] for m in logM_want])
    iR_want = np.array([np.where(np.round(R_grid,2) == r)[0][0] for r in R_want])

    return logM_want, R_want, ilogM_want, iR_want

##### checking coordinate consistency of halo and skewer files #####
def check_halo_xyz(halos, Lbox, Ng, lit_h):
    xhalos = halos['ZHALO'] # likely in Mpc unit (rather than Mpc/h)
    ixhalos = halos['IHALOZ']

    Lbox = Lbox / lit_h # converting from Mpc/h to Mpc
    xhalos_pred = (ixhalos + 0.)* Lbox/Ng

    return xhalos_pred, xhalos

def check_skewers_xyz(param, skewers):
    # skewers XYZ seem to be grid edges
    xskew = skewers['XSKEW'] # likely in Mpc unit (rather than Mpc/h)
    iskew = skewers['ISKEWX']

    Lbox = param['Lbox'][0] / param['lit_h'][0] # converting from Mpc/h to Mpc
    Ng = param['Ng'][0]
    xskew_pred = iskew * Lbox/Ng

    return xskew_pred, xskew

def common_cell(halos, ske):
    # plt.plot(ske['ISKEWX'], ske['ISKEWY'], 'k.')
    # plt.plot(halos['IHALOX'][10000:15000], halos['IHALOY'][10000:15000], 'r+')

    ix = [2121, 599, 600, 3150, 3441, 3654, 2798, 2789, 1224]
    iy = [2838, 1164, 1063, 2125, 1498, 3740, 4056, 3879, 411]
    # nc, c, c?, c, c, c?, c, nc, c

    halo_ind = []
    ske_ind = []
    for i in range(len(ix)):
        halo_ind.append(np.where((halos['IHALOX'] == ix[i]) & (halos['IHALOY'] == iy[i]))[0][0])
        ske_ind.append(np.where((ske['ISKEWX'] == ix[i]) & (ske['ISKEWY'] == iy[i]))[0][0])

    return halo_ind, ske_ind

def plot_common_cell(halos, par, ske, halo_ind, ske_ind, index):

    lit_h = par['lit_h'][0]
    Lbox = par['Lbox'][0] / lit_h  # Mpc unit
    Ng = par['Ng'][0]
    cellsize = Lbox / Ng

    oden = ske['ODEN']
    zskew = np.arange(Ng) * cellsize
    zskew_offset = (0.5 + np.arange(Ng)) * cellsize

    zhalo = halos['ZHALO'][halo_ind[index]]
    plt.plot(zhalo, 30, 'r*', ms=10)
    plt.plot(zskew_offset, oden[ske_ind[index]], 'r')
    plt.plot(zskew, oden[ske_ind[index]], 'k-')
    plt.xlim(zhalo - 0.25, zhalo + 0.25)
    plt.grid()
    plt.show()

