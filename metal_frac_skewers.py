"""
Functions in this module:
    - make_xciv_skewer
    - make_xmetal_skewer
    - plot_skewers
"""

import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
import enigma.reion_forest.utils as reion_utils

def make_xciv_skewer(params, skewers, cloudy_lookup, outfile):
    # modeled after enigma.tpe.utils.make_tau_skewers

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
    # modeled after enigma.tpe.utils.make_tau_skewers

    nh_bar = params['nH_bar']
    nh_skewers = np.log10(nh_bar*skewers['ODEN']) # since the interpolation takes log10 units
    temp_skewers = np.log10(skewers['T']) # since the interpolation takes log10 units

    # hardcoded grid limit...
    clip_nh_skewers = np.clip(nh_skewers, -7, 0)
    clip_temp_skewers = np.clip(temp_skewers, 2, 7)

    Nskew = len(skewers)
    xmetal_arr = []
    for i in range(Nskew): # ~1 min to process 10,000 skewers
        xmetal_out = cloudy_utils.get_ion_frac(cloudy_lookup, cloudy_lookup_str, -3.5, clip_nh_skewers[i], clip_temp_skewers[i])
        xmetal_arr.append(xmetal_out)
    skewers[out_str] = xmetal_arr

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
    # 9488, 1704, 5684
    # 8604, 2706, 7339 - pix > grid limits?

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

def get_tau0_frac(lookup, nh_log10, temp_log10, z, logZ=-3.5):

    cldy_metal_ion_ls = ['IONI CARB 4 1', 'IONI SILI 4 1', 'IONI NITR 5 1', 'IONI OXYG 6 1']
    metal_ion_ls = ['C IV', 'Si IV', 'N V', 'O VI']

    t0f = []
    for i, cldy_metal_ion in enumerate(cldy_metal_ion_ls):
        ion_frac = cloudy_utils.get_ion_frac(lookup, cldy_metal_ion, logZ, nh_log10, temp_log10)[0][0]
        tau0, f_ratio, v_metal, nH_bar = reion_utils.metal_tau0(metal_ion_ls[i], z, logZ)
        print(metal_ion_ls[i], tau0, ion_frac)
        t0f.append(tau0*ion_frac)

    return t0f

