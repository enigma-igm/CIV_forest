import pandas as pd
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline

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

def get_ion_frac(lookup, metal_ion, fixed_Z_value, want_hden_value, want_temp_value):

    # fixed_Z_value options are -3.5 or -1.5

    # selecting out a specific metallicity
    metal_ind = np.where(lookup['METALS= %'] == fixed_Z_value)[0]
    nh_grid = np.array(lookup['HDEN=%f L'][metal_ind])
    temp_grid = np.array(lookup['CONSTANT'][metal_ind])
    ion_frac = np.array(lookup[metal_ion][metal_ind])

    # defining additional arrays for 2D interpolation
    nh_grid_uniq = np.unique(nh_grid) # returns unique and sorted values
    temp_grid_uniq = np.unique(temp_grid)
    ion_frac2d = np.reshape(ion_frac, (len(nh_grid_uniq), len(temp_grid_uniq)))

    # RectBivariateSpline is a (faster) subclass of interp2d if x and y are rectangular grids
    # linear interpolation if kx=1 and ky=1
    # checked: x = density grid and y = temp grid
    interp_func = RectBivariateSpline(nh_grid_uniq, temp_grid_uniq, ion_frac2d, kx=1, ky=1)
    if np.ndim(want_hden_value) == 0:
        outfrac = interp_func(want_hden_value, want_temp_value)
    else:
        outfrac = interp_func.ev(want_hden_value, want_temp_value)
    return outfrac

def get_ion_frac1d(lookup, metal_ion, fixed_Z_value, want_hden_value=None, want_temp_value=None):

    # fixed_Z_value must be specified; options are -3.5 or -1.5

    # selecting out a specific metallicity
    metal_ind = np.where(lookup['METALS= %'] == fixed_Z_value)[0]
    nh_grid = np.array(lookup['HDEN=%f L'][metal_ind])
    temp_grid = np.array(lookup['CONSTANT'][metal_ind])
    ion_frac = np.array(lookup[metal_ion][metal_ind])

    if want_hden_value != None: # get 1D slice at fixed density
        ind_slice = np.where(nh_grid == want_hden_value)[0]
    elif want_temp_value != None: # get 1D slice at fixed temperature
        ind_slice = np.where(temp_grid == want_temp_value)[0]
    else: # if not fixing hden and not fixing temp (i.e. just fixing metallicity)
        return ion_frac, nh_grid, temp_grid # 1D array

    # return 1D slice; also 1D array
    return ion_frac[ind_slice], nh_grid[ind_slice], temp_grid[ind_slice]






