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
    #masked_ske = xciv_ske
    #masked_ske['X_CIV'] = xciv_ske['X_CIV'] * maski

    # no easy way to make a True copy the original table
    masked_ske = Table([xciv_ske['XSKEW'], xciv_ske['YSKEW'], xciv_ske['ZSKEW'], \
                        xciv_ske['ISKEWX'], xciv_ske['ISKEWY'], xciv_ske['ISKEWZ'], \
                        xciv_ske['PROJ_AXIS'], xciv_ske['PROJ_SIGN'], xciv_ske['ODEN'], xciv_ske['VEL_Z'], xciv_ske['T'], \
                        xciv_ske['X_HI'], xciv_ske['TAU'], xciv_ske['ODEN_TAU'], xciv_ske['T_TAU'], xciv_ske['X_CIV'] * maski], \
                       names=('XSKEW','YSKEW','ZSKEW','ISKEWX','ISKEWY','ISKEWZ','PROJ_AXIS','PROJ_SIGN',\
                              'ODEN','VEL_Z','T','X_HI','TAU','ODEN_TAU','T_TAU','X_CIV'))

    ri = int(col.strip('mask')[0])
    logMi = int(col.strip('mask')[1])
    rval = r_grid[ri]
    logMval = logM_grid[logMi]
    outfile = outfile_prefix + 'r%0.5f_logM%0.2f.fits' % (rval, logMval)
    print('Writing to %s' % outfile)

    hdu_param = fits.BinTableHDU(xciv_par.as_array())
    hdu_table = fits.BinTableHDU(masked_ske.as_array())
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(outfile, overwrite=True)




