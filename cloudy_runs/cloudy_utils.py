"""
Functions in this module:
    - read_cloudy_koki
    - get_ion_frac
    - get_ion_frac1d
    - make_cldy_grid_script
"""

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

# cu.make_cldy_grid_script('cloudy_grid_more.in', -7, 0, 0.1, -3.5, -1.5, 2, 7, 0.1, 32, metals_list)
# metals_list = ['hydrogen 1 2', 'oxygen 1 7', 'carbon 1 7', 'silicon 1 7', 'nitrogen 1 7', 'magnesium 1 4']
def make_cldy_grid_script(outfile, hden_start, hden_end, hden_step, \
                          metals_start, metals_end, \
                          temp_start, temp_end, temp_step, \
                          ncpus, metals_list, title='interpolation table', z=4.5):

    prefix = outfile.split('.')[0]
    newfile = open(outfile, 'w')
    newfile.write('title %s\n' % title)
    newfile.write('cmb z=%0.1f\n' % z)
    newfile.write('table hm12 z=%0.1f\n' % z)
    newfile.write('hden -2. vary\n')
    newfile.write('grid %0.1f %0.1f %0.1f ncpus %d\n' % (hden_start, hden_end, hden_step, ncpus))

    if metals_end <= metals_start: # do not vary metallicity
        newfile.write('metals -3.5\n')
    else:
        newfile.write('metals -3.5 vary\n')
        newfile.write('grid %0.1f %0.1f 2\n' % (metals_start, metals_end))

    newfile.write('constant temperature 4. vary\n')
    newfile.write('grid %0.1f %0.1f %0.1f\n' % (temp_start, temp_end, temp_step))
    newfile.write('stop zone 1\n')
    newfile.write('iterate to convergence\n')
    newfile.write('print line faint -1\n')
    newfile.write('set save prefix "%s"\n' % prefix)
    newfile.write('save performance ".per"\n')
    newfile.write('save overview last ".ovr" no hash\n')
    newfile.write('save results last ".rlt"\n')
    newfile.write('save continuum last ".con"\n')
    newfile.write('save incident continuum last ".inc"\n')
    newfile.write('save ionization means last ".ion"\n')
    newfile.write('save grid ".grd"\n')
    newfile.write('save averages ".avr" last no hash\n')

    for metal in metals_list:
        name = metal.split(' ')[0]
        ion_start = int(metal.split(' ')[1])
        ion_end = int(metal.split(' ')[2])
        for i in range(ion_start, ion_end+1):
            newfile.write('ionization, %s %d over volume\n' % (name, int(i)))
    newfile.write('end of averages')
    newfile.close()






