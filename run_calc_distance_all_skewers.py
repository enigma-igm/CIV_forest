import halos_skewers
from astropy.table import Table
from astropy.io import fits
import argparse
import numpy as np

def parser():

    parser = argparse.ArgumentParser(description='Create mask file for a given enrichment model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranskewerfile', type=str, required=True, help="Name of random skewerfile")
    parser.add_argument('--halofile', type=str, required=True, help="Name of halo file")
    parser.add_argument('--outfile', type=str, required=True, help="Name of output file")
    parser.add_argument('--Rmax', type=float, required=True, help="Maximum radius below which enrichment occurs")
    parser.add_argument('--logMmin', type=float, required=True, help="Minimum halo mass above which enrichment occurs")

    return parser.parse_args()

def main():

    args = parser()
    Rmax = args.Rmax
    logM_min = args.logMmin
    params = Table.read(args.ranskewerfile, hdu=1)
    skewers = Table.read(args.ranskewerfile, hdu=2)
    halos = Table.read(args.halofile)

    #skewers = skewers[0:10] # testing

    enrich_mask = halos_skewers.calc_distance_all_skewers(params, skewers, halos, Rmax, logM_min)
    enrich_mask = enrich_mask.astype(int)  # converting the bool arr to 1 and 0

    # output file contain all params and skewers as ranskewerfile, with extra columns as below
    params['R_val'] = Rmax
    params['logM_val'] = logM_min
    skewers['MASK'] = enrich_mask
    skewers['X_CIV'] = skewers['X_CIV'] * enrich_mask

    print('Writing out to disk')
    hdu_param = fits.table_to_hdu(params)
    hdu_table = fits.table_to_hdu(skewers)
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(hdu_table)
    hdulist.writeto(args.outfile, overwrite=True)

if __name__ == '__main__':
    main()