# 10/13/2020


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

def x_civ_lookup(lookup):

    # as a function of temperature, HI volume density, and metallicity
    temp = lookup['Te']
    hden = lookup['hden']
    Z = lookup['METALS= %']
    civ_frac = lookup['IONI CARB 4 1']

    # use log10 values instead...?! Check skewers values
    lookup_new = pd.concat([civ_frac, np.log10(hden), np.log10(temp), Z], axis=1)

    lookup_new.columns.values[0] = 'CIV_fraction'
    lookup_new.columns.values[-1] = 'Z'

    lookup_arr = []
    return lookup_new

def arr_lookup(lookup_new, Z, plot=False):
    ind = np.where(lookup_new['Z'] == Z)[0]
    hden = np.array(lookup_new['hden'][ind])
    te = np.array(lookup_new['Te'][ind])
    civ = np.array(lookup_new['CIV_fraction'][ind])

    print(np.shape(hden), np.shape(te), np.shape(civ))

    new_hden = np.reshape(hden, (61, 31))
    new_te = np.reshape(te, (61, 31))
    new_civ = np.reshape(civ, (61, 31))
    print(np.min(new_civ), np.max(new_civ))

    if plot:
        plt.imshow(new_civ, vmin=0, vmax=0.6, extent=[3, 6, -6, -1])
        plt.colorbar()
        plt.xticks(np.arange(3, 6.5, 0.5))
        plt.yticks(np.arange(-6, -0.5, 0.5))
        plt.xlabel('log10(T)', fontsize=15)
        plt.ylabel('log10(hden)', fontsize=15)
        plt.show()

    return new_hden, new_te, new_civ

