"""
Generate metal forest skewers and write out the results.
    - modified from enigma.reion_forest.run_reion_skewers.py
"""

from astropy.table import Table
from enigma.tpe.utils import make_reion_skewers_metal
import argparse
import time

def main():

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranskewerfile', type=str, required=True, help="Random skewerfile containing the metal ion fraction skewers")
    parser.add_argument('--outfile', type=str, required=True, help="Output file")
    parser.add_argument('--dmax', type=float, required=True, help="Maximum density threshold") # typically set to 3000
    parser.add_argument('--metal_colname', type=str, required=True, help="Column name for metal fraction, e.g. X_CIV")
    parser.add_argument('--metal_mass', type=float, required=True, help="Atomic mass of the metal, e.g. 12.0 for Carbon")

    args = parser.parse_args()

    # Now read in the skewers
    params=Table.read(args.ranskewerfile,hdu=1)
    skewers=Table.read(args.ranskewerfile,hdu=2)
    start = time.time()
    retval = make_reion_skewers_metal(params, skewers, args.outfile, args.dmax, args.metal_colname, args.metal_mass, \
                                      IMPOSE_EOS=False)
    end = time.time()

    print('Done in %0.2f min' % ((end-start)/60.))

if __name__ == '__main__':
    main()
