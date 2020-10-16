import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def read_cloudy_koki(filename):
    # modified from Koki (https://github.com/enigma-igm/xcorr_QSOfield/blob/master/cloudy/plot_cloudy.py)

    """
        read CLOUDY outputs (*.avr and *.ovr) and make a lookup table
        of various chemical elements as function of density and temperature
        usage: e.g. lookup=read_cloudy(filename='hm12_z6_lookup_table') # note without the ".in" extension
                    output: lookup [pandas table]
    """

    print('reading ... : ', filename)

    avr=pd.read_csv(filename+'.avr',sep='\t') # load average ionization fraction of various speices
    avr=avr.drop(avr.columns[0],axis=1)       # drop the first column as it is nan
    ovr=pd.read_csv(filename+'.ovr',sep='\t') # load overview file, which inlcude hydrogen density (hden [1/cm3]) and temperature (Te [K])
    grd=pd.read_csv(filename+'.grd',sep='\t') # grid file containing the values of the grid points

    lookup = pd.concat([ovr, avr, grd], axis=1, sort=False) # combine .ovr and .avr outputs into a summary CLOUDY lookup table

    # remove leading and trailing white spaces in the column names
    for i, col in enumerate(lookup):
        lookup.columns.values[i] = col.strip()

    print('----- Available column names ----- ')
    print(lookup.columns.values)

    return lookup

def create_metal_lookup(filename, metal_name='CARB'):

    # metal_name options: 'CARB', 'OXYG', 'SILI'

    ##### this block similar to read_cloudy_koki #####
    print('reading ... : ', filename)
    avr = pd.read_csv(filename + '.avr', sep='\t')  # load average ionization fraction of various speices
    avr = avr.drop(avr.columns[0], axis=1)  # drop the first column as it is nan
    ovr = pd.read_csv(filename + '.ovr',
                      sep='\t')  # load overview file, which inlcude hydrogen density (hden [1/cm3]) and temperature (Te [K])
    grd = pd.read_csv(filename + '.grd', sep='\t')  # grid file containing the values of the grid points
    lookup = pd.concat([ovr, avr, grd], axis=1,
                       sort=False)  # combine .ovr and .avr outputs into a summary CLOUDY lookup table

    # remove leading and trailing white spaces in the column names
    for i, col in enumerate(lookup):
        lookup.columns.values[i] = col.strip()

    temp = lookup['Te']
    hden = lookup['hden']
    Z = lookup['METALS= %']

    # TODO: below currently only works for metals with ions up to 6
    met1_frac = lookup['%s' % ('IONI ' + metal_name + ' 1 1')]
    met2_frac = lookup['%s' % ('IONI ' + metal_name + ' 2 1')]
    met3_frac = lookup['%s' % ('IONI ' + metal_name + ' 3 1')]
    met4_frac = lookup['%s' % ('IONI ' + metal_name + ' 4 1')]
    met5_frac = lookup['%s' % ('IONI ' + metal_name + ' 5 1')]
    met6_frac = lookup['%s' % ('IONI ' + metal_name + ' 6 1')]

    # use log10 values instead...?! Should maybe be consistent with skewers values
    lookup_new = pd.concat([np.log10(hden), np.log10(temp), Z, \
                            met1_frac, met2_frac, met3_frac, met4_frac, met5_frac, met6_frac], axis=1)

    lookup_new.columns.values[2] = 'Z'
    lookup_new.columns.values[3] = metal_name + '1'
    lookup_new.columns.values[4] = metal_name + '2'
    lookup_new.columns.values[5] = metal_name + '3'
    lookup_new.columns.values[6] = metal_name + '4'
    lookup_new.columns.values[7] = metal_name + '5'
    lookup_new.columns.values[8] = metal_name + '6'

    return lookup_new

def arr_lookup(lookup_new, Z, metal_ion='CARB4', plot=False):

    ind = np.where(lookup_new['Z'] == Z)[0]
    hden = np.array(lookup_new['hden'][ind])
    te = np.array(lookup_new['Te'][ind])
    ion = np.array(lookup_new[metal_ion][ind])

    new_hden = np.reshape(hden, (61, 31)) # hden grids x temperature grids
    new_te = np.reshape(te, (61, 31))
    new_ion = np.reshape(ion, (61, 31))

    if plot:
        plt.title(metal_ion, fontsize=18)
        plt.imshow(new_ion, extent=[3, 6, -6, -1])
        plt.colorbar()
        plt.xticks(np.arange(3, 6.5, 0.5))
        plt.yticks(np.arange(-6, -0.5, 0.5))
        plt.xlabel('log10(T)', fontsize=15)
        plt.ylabel('log10(hden)', fontsize=15)
        plt.show()

    return new_ion

