# Feb 24, 21: modified from enigma/reion_forest/nyx_skewers.py
# generate halo mask and the resulting metal skewers for a grid of enrichment models

from matplotlib import pyplot as plt
import os
import shutil
from IPython import embed
import argparse
import time

# ~30-60 min for distance computation for 10,000 skewers (?)
# 1.5 hrs for tau skewer generation for 10,000 skewers
def parser():

    # double check
    parser = argparse.ArgumentParser(description='Create random skewers for metal line forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranskewerfile', type=str, required=True, help="Name of random skewerfile containing the (uniformly enriched) metal ion fraction skewers")
    parser.add_argument('--halofile', type=str, required=True, help="Name of halo file")
    #parser.add_argument('--nran', type=int, help="Number of skewers to create") # 10000
    #parser.add_argument('--seed', type=int, help="Seed for random number generator") # 789
    return parser.parse_args()

def main():

    import numpy as np
    from astropy.io import fits
    from enigma.tpe.skewer_tools import random_skewers
    from enigma.reion_forest import utils
    from enigma.tpe.utils import calc_eosfit2
    from astropy.table import Table, vstack, hstack
    import halos_skewers

    args = parser()
    #nran = args.nran
    #seed = args.seed
    #dmax = args.dmax

    """
    # Nyx simulation files and paths
    zstr = 'z45'
    sim_path = '/mnt/quasar/sims/L100n4096S2/'
    outpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/' + zstr + '/'
    hdf5file = sim_path + 'z45.h5'

    ranovtfile = os.path.join(outpath, 'rand_skewers_' + zstr + '_OVT.fits')
    enrichment_path = os.path.join(outpath, 'enrichment_models') # make sure to create this directory beforehand

    # Create skewers
    ret = random_skewers(nran, hdf5file, ranovtfile, seed)

    ## Read in skewers
    params = Table.read(ranovtfile, hdu=1)
    skewers = Table.read(ranovtfile, hdu=2)
    nskew = len(skewers)
    ng = (skewers['ODEN'].shape)[1]

    # Fit equation of state (EOS)
    oden = skewers['ODEN'].reshape(nskew * ng)
    T = skewers['T'].reshape(nskew * ng)
    (logT0, gamma) = calc_eosfit2(oden, T, -0.5, 0.0)
    params['EOS-logT0'] = logT0
    params['EOS-GAMMA'] = gamma
    params['seed'] = seed
    """

    zstr = 'z45'
    outpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/' + zstr + '/'
    enrichment_path = os.path.join(outpath, 'enrichment_models')  # make sure to create this directory beforehand

    uniform_xciv_params = Table.read(args.ranskewerfile, hdu=1)
    uniform_xciv_skewers = Table.read(args.ranskewerfile, hdu=1)

    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.25)  # 11 x 13 models

    # testing entire code on subset of models and skewers
    logM = logM[-2:]
    R = R[4:6]
    uniform_xciv_skewers = uniform_xciv_skewers[0:100]

    nlogM, nR = len(logM), len(R)
    halos = Table.read(args.halofile)

    print('Creating tau skewers using {:d} logM and {:d} R models '.format(nlogM, nR))

    for i_R, Rval in enumerate(R):
        #time.sleep(600) # wait for 1 hr
        time.sleep(60)
        for i_logM, logMval in enumerate(logM):
            enrich_mask = halos_skewers.calc_distance_all_skewers(uniform_xciv_params, uniform_xciv_skewers, halos, Rval, logMval)
            enrich_mask = enrich_mask.astype(int)  # converting the bool arr to 1 and 0

            uniform_xciv_params['R_val'] = Rval
            uniform_xciv_params['logM_val'] = logMval
            uniform_xciv_skewers['MASK'] = enrich_mask
            uniform_xciv_skewers['X_CIV'] = uniform_xciv_skewers['X_CIV'] * enrich_mask

            # Write out the skewers with changes to the X_CIV skewers
            xciv_outfile = os.path.join(enrichment_path, 'rand_skewers_' + zstr + '_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')

            hdu_param = fits.table_to_hdu(uniform_xciv_params)
            hdu_table = fits.table_to_hdu(uniform_xciv_skewers)
            hdulist = fits.HDUList()
            hdulist.append(hdu_param)
            hdulist.append(hdu_table)
            hdulist.writeto(xciv_outfile, overwrite=True)

            tau_logfile = os.path.join(enrichment_path, 'rand_skewers_' + zstr + '_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '_tau.log')
            tau_outfile = os.path.join(enrichment_path, 'rand_skewers_' + zstr + '_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '_tau.fits')

            command = 'python run_reion_skewers_metal.py ' + '--ranskewerfile ' + xciv_outfile + ' --outfile ' + tau_outfile + \
                      ' --dmax 3000 --metal_colname X_CIV --metal_mass 12'

            test_command = "echo %s > %s" % (command, tau_logfile)
            ret = os.system(test_command)

if __name__ == '__main__':
    main()


