import time
import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import enigma.reion_forest.utils as reion_utils
from scipy.spatial import distance # v14.0 syntax

def init_all():
    halofile = 'nyx_sim_data/z45_halo_logMmin_8.fits'
    skewerfile = 'nyx_sim_data/rand_skewers_z45_ovt_tau.fits'
    skewerfile = 'nyx_sim_data/subset100/subset100_rand_skewers_z45_ovt_tau_xciv.fits'
    par = Table.read(skewerfile, hdu=1)
    ske = Table.read(skewerfile, hdu=2)
    halos = Table.read(halofile)

    return par, ske, halos

def init_halo_grids(logMmin=8.0, logMmax=11.0, dlogM=0.5, Rmin=0.20, Rmax=2.5, nR=5):

    # Booth et al (2012): logM=8 to logM=11.0 in 0.5 dex and R=31.25 proper kpc to R=500 proper kpc, in factor of 2
    # r_pmpc_booth = np.array([0.03125, 0.0625 , 0.125  , 0.25   , 0.5    ])
    # r_cmpc_booth = (1+4.5) * r_pmpc_booth
    # r_cmpc_booth = array([0.171875, 0.34375 , 0.6875  , 1.375   , 2.75    ])

    logM_grid = np.arange(logMmin, logMmax + dlogM, dlogM)
    R_grid = np.linspace(Rmin, Rmax, nR) # cMpc

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

###### testing ######
def calc_distance_one_skewer(one_skewer, params, halos, Rmax):
    # including periodic BC

    start = time.time()
    Lbox = params['Lbox'][0]
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng

    xskew = one_skewer['XSKEW']
    yskew = one_skewer['YSKEW']
    zskew = np.arange(Ng) * cellsize

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

    # first cut to trim down the number of halos
    dx2dy2 = dx ** 2 + dy ** 2
    want_halos = np.where(dx2dy2 <= Rmax ** 2)[0]
    dx2dy2 = dx2dy2[want_halos]

    # this is slower...
    #mask_halo_2d = dx2dy2 <= Rmax**2
    #dx2dy2 = dx2dy2[mask_halo_2d]

    zpix_near_halo = []
    for zpix in zskew:
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

    return zpix_near_halo

def calc_distance_all_skewers(params, skewers, halos, Rmax):
    # 0.17 min (0.3 min) for 100 skewers at Rmax=0.2 Mpc (2.5 Mpc)
    start = time.time()
    all_iz_near_halo = []
    for iskew in skewers:
        iz_near_halo = calc_distance_one_skewer(iskew, params, halos, Rmax)
        all_iz_near_halo.append(iz_near_halo)
    end = time.time()
    print((end - start) / 60.)

    return all_iz_near_halo

def write_iz_mask(params, skewers, all_iz_near_halo, outfile):

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

    Zmin = Zc - slice_thickness
    Zmax = Zc + slice_thickness
    iz_halos = np.where((halos['ZHALO'] >= Zmin) & (halos['ZHALO'] < Zmax))[0]
    iz_skewers = np.where((zskew >= Zmin) & (zskew < Zmax))[0]

    plt.plot(halos['XHALO'][iz_halos], halos['YHALO'][iz_halos], '.', ms=5, alpha=0.5, label=logM_min)

    for iskew in skewers:
        if np.sum(iskew['ZPIX_NEAR_HALO'][iz_skewers]):
            plt.plot(iskew['XSKEW'], iskew['YSKEW'], 'y*', ms=5)
        #else:
            #plt.plot(iskew['XSKEW'], iskew['YSKEW'], 'r*', ms=5)
    plt.legend()
    plt.show()

def make_3darr(params, skewers, halos):

    halos_xyz = [[halos['XHALO'][i], halos['YHALO'][i], halos['ZHALO'][i]] for i in range(len(halos))]

    Lbox = params['Lbox'][0]
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng

    skew_xyz = []
    for i in range(10):
        xskew = skewers['XSKEW'][i]
        yskew = skewers['YSKEW'][i]
        zskew = np.arange(Ng) * cellsize
        one_skew_xyz = [[xskew, yskew, zskew[j]] for j in range(len(zskew))]
        skew_xyz.append(one_skew_xyz)

    return halos_xyz, skew_xyz

def calc_distance_old(skew_xyz, halos_xyz):
    start = time.time()
    out = distance.cdist(skew_xyz[0], halos_xyz)
    end = time.time()
    print((end-start)/60.)

    start = time.time()
    out = distance.cdist(skew_xyz[0], halos_xyz, 'sqeuclidean')
    end = time.time()
    print((end - start) / 60.)
