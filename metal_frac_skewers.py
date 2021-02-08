"""
Functions in this module:
    - make_xciv_skewer
    - make_xmetal_skewer
    - plot_skewers
    - get_tau0_frac
"""

import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
import enigma.reion_forest.utils as reion_utils

def make_xciv_skewer(params, skewers, cloudy_lookup, outfile):
    # modeled after enigma.tpe.utils.make_tau_skewers
    # old version ---> see make_xmetal_skewer

    nh_bar = params['nH_bar']
    nh_skewers = np.log10(nh_bar*skewers['ODEN']) # since the interpolation takes log10 units
    temp_skewers = np.log10(skewers['T']) # since the interpolation takes log10 units

    Nskew = len(skewers)
    xciv_arr = []
    for i in range(Nskew): # ~1 min to process 10,000 skewers
        xciv_out = cloudy_utils.get_ion_frac(cloudy_lookup, 'IONI CARB 4 1', -3.5, nh_skewers[i], temp_skewers[i])
        xciv_arr.append(xciv_out)
    skewers['X_CIV'] = xciv_arr

    if outfile is not None:
        hdu_param = fits.BinTableHDU(params.as_array())
        hdu_table = fits.BinTableHDU(skewers.as_array())
        hdulist = fits.HDUList()
        hdulist.append(hdu_param)
        hdulist.append(hdu_table)
        hdulist.writeto(outfile, overwrite=True)
        return True
    else:
        return params.as_array(), skewers.as_array()

def make_xmetal_skewer(params, skewers, cloudy_lookup, cloudy_lookup_str, out_str, outfile):
    """
    Generate skewers of x_metal fraction.
        - modeled after enigma.tpe.utils.make_tau_skewers.
        - clipping on skewers falling outside of cloudy grid limits
        - note: grid limits are hardcoded... please change as appropriate.

    Args:
        params (astropy table)
        skewers (astropy table)
        cloudy_lookup (pandas dataframe):
            obtained by calling cloudy_utils.read_cloudy_koki
        cloudy_lookup_str (str):
            column name from cloudy files for the ionization stage of the desired metal ion. E.g. 'IONI CARB 4 1'
        out_str (str):
            name to assign to the ion fraction to be written out in 'outfile'. E.g., 'X_CIV' for CIV ion fraction
        outfile (str, optional):
            name of output file, .fits format.
    Returns:
        If outfile == None, then return the changed params and skewers arrays.
        If outfile is given, then write out params and skewers.
    """

    nh_bar = params['nH_bar']
    nh_skewers = np.log10(nh_bar*skewers['ODEN']) # since the interpolation takes log10 units
    temp_skewers = np.log10(skewers['T']) # since the interpolation takes log10 units

    # Clipping for skewers outside of cloudy grid limits (hardcoded grid limit...)
    clip_nh_skewers = np.clip(nh_skewers, -7, 0)
    clip_temp_skewers = np.clip(temp_skewers, 2, 7)

    Nskew = len(skewers)
    xmetal_arr = [] # metal ion fraction array to store the skewers and to be written out
    for i in range(Nskew): # ~1 min to process 10,000 skewers
        xmetal_out = cloudy_utils.get_ion_frac(cloudy_lookup, cloudy_lookup_str, -3.5, clip_nh_skewers[i], clip_temp_skewers[i])
        xmetal_arr.append(xmetal_out)

    skewers[out_str] = xmetal_arr # appending a new x_metal column to the skewers array

    if outfile is not None:
        hdu_param = fits.BinTableHDU(params.as_array())
        hdu_table = fits.BinTableHDU(skewers.as_array())
        hdulist = fits.HDUList()
        hdulist.append(hdu_param)
        hdulist.append(hdu_table)
        hdulist.writeto(outfile, overwrite=True)
        return True
    else:
        return params.as_array(), skewers.as_array()

def plot_skewers(params, skewers, i):
    # plot oden, T, and x_metal skewers for index 'i'

    print(i)
    #nh_bar = params['nH_bar']
    #hden_skewers = nh_bar*skewers['ODEN']
    oden_skewers = skewers['ODEN']
    temp_skewers = skewers['T']
    xciv_skewers = skewers['X_CIV']

    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.plot(oden_skewers[i])
    plt.ylabel('Overdensity', fontsize=13)
    plt.subplot(312)
    plt.plot(temp_skewers[i])
    plt.ylabel('Temperature', fontsize=13)
    plt.subplot(313)
    plt.plot(xciv_skewers[i])
    plt.ylabel('X_CIV', fontsize=13)

    plt.tight_layout()
    plt.show()

def get_tau0_frac(lookup, cldy_metal_ion, metal_ion, nh_log10, temp_log10, z, logZ=-3.5):
    """
    Compute tau0 and metal ion fraction.

    Args:
        lookup (pandas dataframe):
            obtained by calling cloudy_utils.read_cloudy_koki
        cldy_metal_ion (str):
            column name from cloudy files for the ionization stage of the desired metal ion. E.g. 'IONI CARB 4 1'
        metal_ion (str):
            name of metal ion, e.g. 'Mg II' (must include whitespace between name and ionization stage)
        nh_log10 (float or None):
            value of HI density in log10 unit at which to extract the metal ionic fraction;
            if set to None, then use the mean density at that redshift
        temp_log10 (float):
            value of temperature in log10 unit at which to extract the metal ionic fraction
        z (float):
            redshift of the cloudy models for the ionic fraction
        logZ (float, optional):
            log10 of metallicity relative to solar at which to compute tau0

    Returns:
        tau0 and metal ion fraction
    """

    tau0, f_ratio, v_metal, nH_bar = reion_utils.metal_tau0(metal_ion, z, logZ)

    # using the mean density at z if the input density is not specified
    if nh_log10 == None:
        nh_log10 = np.log10(nH_bar.value)
        print('nh_log10 not provided... using log10(mean density)', nh_log10)

    # because some Cloudy models are run without varying metallicity
    try:
        ion_frac = cloudy_utils.get_ion_frac(lookup, cldy_metal_ion, logZ, nh_log10, temp_log10)[0][0]
    except KeyError:
        ion_frac = cloudy_utils.get_ion_frac(lookup, cldy_metal_ion, None, nh_log10, temp_log10)[0][0]

    return tau0, ion_frac

####################### halos stuffs #######################
import time
from scipy.spatial import distance # v14.0 syntax

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

    dz2 = []
    for izskew in zskew:
        dz2.append((izskew - halos['ZHALO'])**2)

    return dz2

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

    iz_near_halo = []
    for iz in range(len(dz2_all)): # 4096 pixels
        r2 = dz2_all[iz][want_halos] + dx2dy2
        ir2 = np.where(r2 < Rmax ** 2)[0]
        if len(ir2) > 0:
            iz_near_halo.append(True)
        else:
            iz_near_halo.append(False)

    end = time.time()
    #print((end-start)/60.)
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
    iz_halos = np.where((halos['ZHALO'] >= Zmin) & (halos['ZHALO'] <= Zmax))[0]

    plt.plot(halos['XHALO'][iz_halos], halos['YHALO'][iz_halos], '.', ms=5, alpha = 0.5, label=logM_min)
    plt.axis('equal')
    plt.xlabel('XHALO', fontsize=15)
    plt.ylabel('YHALO', fontsize=15)
    plt.axis('equal')
    plt.legend()
    #plt.show()








