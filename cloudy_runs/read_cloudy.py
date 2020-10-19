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
    # pandas dataframe format
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

def get_ion_frac(lookup_new, metal_ion, fixed_Z_value, fixed_hden_value=None, fixed_temp_value=None):

    # lookup_new = read_cloudy_koki()
    # fixed_Z_value must be specified; options are -3.5 or -1.5

    metal_ind = np.where(lookup_new['METALS= %'] == fixed_Z_value)[0]
    if fixed_hden_value != None:
        ind_slice = np.where(lookup_new['HDEN=%f L'] == fixed_hden_value)[0]
    elif fixed_temp_value != None:
        ind_slice = np.where(lookup_new['CONSTANT'] == fixed_temp_value)[0]

    if fixed_hden_value != None or fixed_temp_value != None:
        comm_ind = np.intersect1d(metal_ind, ind_slice)
    else:
        comm_ind = metal_ind

    ion_frac = lookup_new[metal_ion][comm_ind].values
    nh_grid = lookup_new['HDEN=%f L'][comm_ind].values
    temp_grid = lookup_new['CONSTANT'][comm_ind].values

    return ion_frac, nh_grid, temp_grid

# IN PROG
def get_ion_frac2(lookup_new, metal_ion, fixed_Z_value, want_hden_value=None, want_temp_value=None):

    # lookup_new = read_cloudy_koki()
    # fixed_Z_value must be specified; options are -3.5 or -1.5

    metal_ind = np.where(lookup_new['METALS= %'] == fixed_Z_value)[0]
    nh_grid = np.array(lookup_new['HDEN=%f L'][metal_ind])
    temp_grid = np.array(lookup_new['CONSTANT'][metal_ind])
    ion_frac = np.array(lookup_new[metal_ion][metal_ind])

    # defining additional arrays
    nh_grid_uniq = np.unique(nh_grid)
    temp_grid_uniq = np.unique(temp_grid)
    ion_frac2d = np.reshape(ion_frac, (len(nh_grid_uniq), len(temp_grid_uniq))) # hden grids x temperature grids

    if want_hden_value != None and want_temp_value is None: # get 1D slice at fixed density
        ind_slice = np.where(nh_grid == want_hden_value)[0]
    elif want_temp_value != None and want_hden_value is None: # get 1D slice at fixed temperature
        ind_slice = np.where(temp_grid == want_temp_value)[0]
    elif want_hden_value != None and want_temp_value != None:
        # TO CHECK: row is temp grid and col is nh_grid, or is it vice versa?

        # RectBivariateSpline is a (faster) subclass of interp2d if x and y are regular grids
        interp_func = RectBivariateSpline(nh_grid_uniq, temp_grid_uniq, ion_frac2d, kx=1, ky=1) # linear 2D interpolation
        outfrac = interp_func(want_hden_value, want_temp_value)
        print("interpolating")
        return outfrac
    else: # if not fixing hden and not fixing temp (i.e. just fixing metallicity)
        return ion_frac, nh_grid, temp_grid

    return ion_frac[ind_slice], nh_grid[ind_slice], temp_grid[ind_slice]




