import time
import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
import enigma.reion_forest.utils as reion_utils
from scipy.spatial import distance # v14.0 syntax

def init_halo_grids(logMmin=8.0, logMmax=11.0, dlogM=0.5, Rmin=0.20, Rmax=2.5, nR=5):

    # Booth et al: logM=8 to logM=11.0 in 0.5 dex and R=31.25 proper kpc to R=500 proper kpc, in factor of 2
    # r_pmpc_booth = np.array([0.03125, 0.0625 , 0.125  , 0.25   , 0.5    ])
    # r_cmpc_booth = (1+4.5) * r_pmpc_booth
    # r_cmpc_booth = array([0.171875, 0.34375 , 0.6875  , 1.375   , 2.75    ])

    logM_grid = np.arange(logMmin, logMmax + dlogM, dlogM)
    R_grid = np.linspace(Rmin, Rmax, nR) # cMpc

    return logM_grid, R_grid

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

def calc_dz2(params, halos):

    Lbox = params['Lbox'][0]
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng
    zskew = np.arange(Ng) * cellsize
    zhalos = np.array(halos['ZHALO'])

    dz2 = []
    for izskew in zskew:
        dz2.append((izskew - zhalos)**2)

    return dz2

def calc_distance_one_skewer2(one_skewer, params, halos, Rmax):
    start = time.time()
    xskew = one_skewer['XSKEW']
    yskew = one_skewer['YSKEW']
    # print(xskew, yskew)
    dx2 = (xskew - halos['XHALO']) ** 2
    dy2 = (yskew - halos['YHALO']) ** 2
    dx2dy2 = dx2 + dy2

    want_halos = np.where(dx2dy2 <= Rmax ** 2)[0]
    dx2dy2 = dx2dy2[want_halos]

    Lbox = params['Lbox'][0]
    Ng = params['Ng'][0]
    cellsize = Lbox / Ng
    zskew = np.arange(Ng) * cellsize
    zhalos = np.array(halos['ZHALO'])

    # setting the pixel masks
    iz_near_halo = []
    for iz in range(Ng):
        r2 = (zskew[iz] - zhalos[want_halos])**2 + dx2dy2
        ir2 = np.where(r2 < Rmax ** 2)[0]
        if len(ir2) > 0:
            iz_near_halo.append(True)
        else:
            iz_near_halo.append(False)
    end = time.time()
    print((end-start)/60.)

    return iz_near_halo

def calc_distance_one_skewer(one_skewer, halos, Rmax, dz2_all):

    start = time.time()
    xskew = one_skewer['XSKEW']
    yskew = one_skewer['YSKEW']
    #print(xskew, yskew)
    dx2 = (xskew - halos['XHALO'])**2
    dy2 = (yskew - halos['YHALO'])**2
    dx2dy2 = dx2 + dy2

    want_halos = np.where(dx2dy2 <= Rmax**2)[0]
    dx2dy2 = dx2dy2[want_halos]

    # setting the pixel masks
    iz_near_halo = []
    for iz in range(len(dz2_all)): # 4096 pixels
        r2 = dz2_all[iz][want_halos] + dx2dy2
        ir2 = np.where(r2 < Rmax ** 2)[0]
        if len(ir2) > 0:
            iz_near_halo.append(True)
        else:
            iz_near_halo.append(False)

    end = time.time()
    print((end-start)/60.)
    #print(len(want_halos), len(want_iz))

    return iz_near_halo

def calc_distance_all_skewers(params, skewers, halos, Rmax, dz2_all):
    start = time.time()
    all_iz_near_halo = []
    for iskew in skewers:
        iz_near_halo = calc_distance_one_skewer(iskew, halos, Rmax, dz2_all)
        all_iz_near_halo.append(iz_near_halo)
    end = time.time()
    print((end - start) / 60.)

    return all_iz_near_halo
    #skewers['X_HI'] = x_HI.reshape(Nskew, Ng)

def plot_halos(halos, slice_thickness, Zc=50, logM_min=None):

    if logM_min != None:
        imass = np.where(np.log10(halos['MASS']) >= logM_min)[0]
        halos = halos[imass]
        print("after mass cut, N(halos): ", len(halos))

    Zmin = Zc - slice_thickness
    Zmax = Zc + slice_thickness
    iz_halos = np.where((halos['ZHALO'] >= Zmin) & (halos['ZHALO'] < Zmax))[0]

    plt.plot(halos['XHALO'][iz_halos], halos['YHALO'][iz_halos], '.', ms=5, alpha = 0.5, label=logM_min)
    plt.axis('equal')
    plt.xlabel('XHALO', fontsize=15)
    plt.ylabel('YHALO', fontsize=15)
    plt.axis('equal')
    plt.legend()
    #plt.show()

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
            plt.plot(skewers['XSKEW'], skewers['YSKEW'], 'y*', ms=5)
        else:
            plt.plot(skewers['XSKEW'], skewers['YSKEW'], 'r*', ms=5)
    plt.legend()
    plt.show()
