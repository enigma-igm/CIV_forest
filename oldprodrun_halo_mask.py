import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import halos_skewers

# production run using IGM cluster

halofile = '/home/sstie/CIV_forest/nyx_sim_data/z45_halo_logMmin_8.fits'
skewerfile = '/home/sstie/old_repo/Jan2021/CIV_forest/sim_data/rand_skewers_z45_ovt_tau.fits'

par, ske, halos = halos_skewers.init_all(halofile=halofile, skewerfile=skewerfile)
logM, R = halos_skewers.init_halo_grids(logMmin=8.0, logMmax=11.0, dlogM=0.5, Rmin=0.1, Rmax=3.0, dR=0.5) # 7 x 7 grid
outfile = '/home/sstie/CIV_forest/nyx_sim_data/rand_skewers_z45_halomask_prodrun.fits'

master_mask = []
master_mask_name = []

# testing
R = R[0:2]
logM = logM[-2:]

for i in range(len(R)-1):
    for j in range(len(logM)):
        zpix_mask = halos_skewers.calc_distance_all_skewers(par, ske, halos, R[i], logM[j])
        name = 'mask' + str(i) + str(j)
        master_mask.append(zpix_mask)
        master_mask_name.append(name)

table_mask = Table(master_mask, names=tuple(master_mask_name))
param = Table([[R], [logM]], names=('r_Mpc', 'logM'))

print('Writing out to disk')
hdu_param = fits.BinTableHDU(param.as_array())
hdu_table = fits.BinTableHDU(table_mask.as_array())
hdulist = fits.HDUList()
hdulist.append(hdu_param)
hdulist.append(hdu_table)
hdulist.writeto(outfile, overwrite=True)