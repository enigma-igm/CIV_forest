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
from astropy.table import Table
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

def plot_masked_xciv(xciv_ori, xciv_mask1, xciv_mask2, xciv_mask3):
    #xciv_ori = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau_xciv.fits', hdu=2)['X_CIV']
    #xciv_mask1 = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_r0.34375_logM9.00.fits', hdu=2)['X_CIV']
    #xciv_mask2 = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_r1.37500_logM10.00.fits', hdu=2)['X_CIV']
    #xciv_mask3 = Table.read('nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_r2.75000_logM9.00.fits', hdu=2)['X_CIV']

    ind = np.random.randint(0, len(xciv_ori))
    print(ind)

    plt.figure(figsize=(12,6))
    plt.subplot(311)
    plt.plot(xciv_ori[ind], alpha=0.7)
    plt.plot(xciv_mask1[ind])
    plt.ylabel('X_CIV', fontsize=12)

    plt.subplot(312)
    plt.plot(xciv_ori[ind], alpha=0.7)
    plt.plot(xciv_mask2[ind])
    plt.ylabel('X_CIV', fontsize=12)

    plt.subplot(313)
    plt.plot(xciv_ori[ind], alpha=0.7)
    plt.plot(xciv_mask3[ind])
    plt.ylabel('X_CIV', fontsize=12)

    plt.show()

