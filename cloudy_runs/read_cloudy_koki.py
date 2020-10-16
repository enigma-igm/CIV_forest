# 10/13/2020
# modified from Koki (https://github.com/enigma-igm/xcorr_QSOfield/blob/master/cloudy/plot_cloudy.py)

import pandas as pd
import numpy as np

def read_cloudy(filename):
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
    x_civ = lookup['IONI CARB 4 1']

    lookup_new = pd.concat([x_CIV, hden, temp, Z], axis=1)

    lookup_new.columns.values[0] = 'CIV_fraction'
    lookup_new.columns.values[-1] = 'Z'

    return lookup_new



