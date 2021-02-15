import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import halos_skewers

outfile = 'nyx_sim_data/rand_skewers_z45_halomask_test.fits'
#logM_grid, R_grid = halos_skewers.init_halo_grids()
par, ske, halos = halos_skewers.init_all()

zpix_mask00 = halos_skewers.calc_distance_all_skewers(par, ske, halos, 1.375, 10)
zpix_mask01 = halos_skewers.calc_distance_all_skewers(par, ske, halos, 0.34375, 9)

table_mask = Table([zpix_mask00, zpix_mask01], names=('mask00', 'mask01'))

print('Writing out to disk')
hdu_table = fits.BinTableHDU(table_mask.as_array())

hdulist = fits.HDUList()
hdulist.append(hdu_table)
hdulist.writeto(outfile, overwrite=True)