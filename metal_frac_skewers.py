import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits

"""
data_path = '/Users/suksientie/research/CIV_forest/sim_data/'
tau_params = Table.read(data_path + 'rand_skewers_z45_ovt_tau.fits', hdu=1)
tau_skewers = Table.read(data_path + 'rand_skewers_z45_ovt_tau.fits', hdu=2)
"""
# LOOKUP_TABLE = cloudy_utils.read_cloudy_koki('')

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

def plot_skewers(params, skewers, i):
    # 9488, 1704, 5684
    # 8604, 2706, 7339 - pix > grid limits?

    print(i)
    nh_bar = params['nH_bar']
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
