"""
Modified from enigma.reion_forest.run_reion_skewers.py
"""

from astropy.table import Table
from enigma.tpe.utils import make_reion_skewers_metal
import argparse

def main():

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranskewerfile', type=str, required=True, help="Random skewerfile")
    parser.add_argument('--outfile', type=str, required=True, help="Output file")
    parser.add_argument('--dmax', type=float, required=True, help="Maximum density threshold")
    parser.add_argument('--metal_colname', type=str, required=True, help="Column name for metal fraction")

    args = parser.parse_args()

    # Now read in the skewers
    params=Table.read(args.ranskewerfile,hdu=1)
    skewers=Table.read(args.ranskewerfile,hdu=2)
    
    #retval = make_reion_skewers(params, skewers, args.outfile, args.dmax)
    retval = make_reion_skewers_metal(params, skewers, args.outfile, args.dmax, args.metal_colname, IMPOSE_EOS=False)

if __name__ == '__main__':
    main()
