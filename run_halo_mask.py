import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import halos_skewers

outfile = 'nyx_sim_data/rand_skewers_z45_halomask_test.fits'
#logM_grid, R_grid = halos_skewers.init_halo_grids()
par, ske, halos = halos_skewers.init_all()

R = [1.375, 0.34375, 2.75]
logM = [10, 9]

zpix_mask00 = halos_skewers.calc_distance_all_skewers(par, ske, halos, 1.375, 10)
zpix_mask11 = halos_skewers.calc_distance_all_skewers(par, ske, halos, 0.34375, 9)
zpix_mask21 = halos_skewers.calc_distance_all_skewers(par, ske, halos, 2.75, 9)

table_mask = Table([zpix_mask00, zpix_mask11, zpix_mask21], names=('mask00', 'mask11', 'mask21'))
param = Table([[R], [logM]], names=('r_Mpc', 'logM'))

print('Writing out to disk')
hdu_param = fits.BinTableHDU(param.as_array())
hdu_table = fits.BinTableHDU(table_mask.as_array())
hdulist = fits.HDUList()
hdulist.append(hdu_param)
hdulist.append(hdu_table)
hdulist.writeto(outfile, overwrite=True)