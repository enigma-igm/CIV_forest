# Feb 24, 21: modified from enigma/reion_forest/nyx_skewers.py
# generate halo mask and the resulting metal skewers for a grid of enrichment models

from matplotlib import pyplot as plt
import os
import shutil
from IPython import embed
import argparse
import time
from subprocess import Popen

# ~30-60 min for distance computation for 10,000 skewers (?)
# 1.5 hrs for tau skewer generation for 10,000 skewers
def parser():

    parser = argparse.ArgumentParser(description='Create random skewers for metal line forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranskewerfile', type=str, required=True, help="Name of random skewerfile containing the (uniformly enriched) metal ion fraction skewers")
    parser.add_argument('--halofile', type=str, required=True, help="Name of halo file")
    parser.add_argument('--nproc', type=int, required=False, help="Number of processes to run simultaneously")

    return parser.parse_args()

def main():

    #import numpy as np
    #from astropy.io import fits
    #from enigma.tpe.skewer_tools import random_skewers
    #from enigma.reion_forest import utils
    #from enigma.tpe.utils import calc_eosfit2
    #from astropy.table import Table, vstack, hstack
    import halos_skewers

    args = parser()
    nproc = args.nproc

    zstr = 'z45'
    outpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/' + zstr + '/enrichment_models/xciv_mask/' # make sure these directories exist

    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

    # testing entire code on subset of models, skewers, and halos
    logM = logM[-3:] # 3 models
    R = R[3:6] # 3 models
    #uniform_xciv_skewers = uniform_xciv_skewers[0:70]
    #rand_halos = np.random.choice(len(halos), replace=False, size=500)
    #halos = halos[rand_halos]
    #######

    nlogM, nR = len(logM), len(R)
    print('Creating masks using {:d} logM and {:d} R models for a total of {:d} models'.format(nlogM, nR, nR * nlogM))

    counter = 0
    counter_file = []
    for i_R, Rval in enumerate(R):
        for i_logM, logMval in enumerate(logM):

            mask_outfile = os.path.join(outpath, 'rand_skewers_' + zstr + '_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits')
            mask_logfile = os.path.join(outpath, 'rand_skewers_' + zstr + '_ovt_xciv_' + 'R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.log')
            command = 'python run_calc_distance_all_skewers.py ' + \
                      ' --ranskewerfile ' + args.ranskewerfile + \
                      ' --halofile' + args.halofile + \
                      ' --outfile ' + mask_outfile + \
                      ' --Rmax' + Rval + ' --logMmin' + logMval > '%s' % mask_logfile

            p = Popen(command, shell=True)

            counter_file.append(mask_outfile)
            counter += 1
            print("counter now", counter)

            if nproc != None:
                if counter % nproc == 0: # every n-th processes
                    print("checking if all files exist")

                    while True:
                        if all([os.path.isfile(f) for f in counter_file]):
                            counter_file = []
                            print("yep...proceeding")
                            break
                        else:
                            print('waiting....')
                            time.sleep(300) # wait 5 min before checking again

if __name__ == '__main__':
    main()

#nohup python prodrun_metal_skewers_enrichment.py --ranskewerfile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/rand_skewers_z45_ovt_xciv.fits --halofile /home/sstie/CIV_forest/nyx_sim_data/z45_halo_logMmin_8.fits --nproc 20 > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/prodrun_metal_skewers_enrichment.log &