from astropy.table import Table
from astropy.io import fits
import numpy as np

maskfile = 'nyx_sim_data/enrichment_models/rand_skewers_z45_halomask.fits'
ori_xciv_file = 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv.fits'
outfile_prefix = 'nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_'

xciv_par = Table.read(ori_xciv_file, hdu=1)
xciv_ske = Table.read(ori_xciv_file, hdu=2)

mask_par = Table.read(maskfile, hdu=1)
mask_all = Table.read(maskfile, hdu=2)
r_grid = mask_par['r_Mpc'][0]
logM_grid = mask_par['logM'][0]

for col in mask_all.columns:
    maski = mask_all[col].astype(int) # converting the bool arr to 1 and 0
    xciv_ske['X_CIV'] = xciv_ske['X_CIV'] * maski

    ri = int(col.strip('mask')[0])
    logMi = int(col.strip('mask')[1])
    rval = r_grid[ri]
    logMval = logM_grid[logMi]
    outfile = outfile_prefix + 'r%0.5f_logM%0.2f.fits' % (rval, logMval)
    print('Writing to %s' % outfile)

    hdu_param = fits.BinTableHDU(xciv_par.as_array())
    hdu_table = fits.BinTableHDU(xciv_ske.as_array())
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)




