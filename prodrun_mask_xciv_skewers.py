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

    #logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.25, 0.1, 3, 0.2)

    # extra models to run (3/30/2021)
    #logM = [8.6, 8.7, 8.8, 8.9, 9.1, 9.2, 9.3, 9.4, 9.6, 9.7, 9.8, 9.9, 10.1, 10.2, 10.3, 10.4, 10.6, 10.7, 10.8, 10.9] # 20 values
    #R = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0] # 15 values

    logM, R = halos_skewers.init_halo_grids(8.5, 11.0, 0.10, 0.1, 3, 0.1) # 26 logM x 30 R models

    # testing entire code on subset of models, skewers, and halos
    #logM = logM[-3:] # 3 models
    #R = R[3:6] # 3 models
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

            if os.path.exists(mask_outfile):
                print(mask_outfile, 'already exists... skipping')
            else:
                command = 'python run_calc_distance_all_skewers.py ' + \
                          ' --ranskewerfile ' + args.ranskewerfile + \
                          ' --halofile ' + args.halofile + \
                          ' --outfile ' + mask_outfile + \
                          ' --Rmax ' + str(Rval) + ' --logMmin ' + str(logMval) + ' > %s' % mask_logfile

                p = Popen(command, shell=True)

                counter_file.append(mask_outfile)
                counter += 1
                print(mask_outfile, ": counter now", counter)

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

# (4/1/21):
# nohup python prodrun_mask_xciv_skewers.py --ranskewerfile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/rand_skewers_z45_ovt_xciv.fits --halofile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/z45_halo_logMmin_8.fits --nproc 40 > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/prodrun_mask_xciv_skewers_extra2.log &
# start: 1415 - 0335 (~14 hrs)

# run command on IGM (3/30/21) for extra models:
# nohup python prodrun_mask_xciv_skewers.py --ranskewerfile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/rand_skewers_z45_ovt_xciv.fits --halofile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/z45_halo_logMmin_8.fits --nproc 50 > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/prodrun_mask_xciv_skewers_extra.log &
# runtime: 8.5 hrs (start: 3/30 15:21; end: 3/30 23:55)

# run command on IGM (2/27/21):
# nohup python prodrun_mask_xciv_skewers.py --ranskewerfile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/rand_skewers_z45_ovt_xciv.fits --halofile /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/z45_halo_logMmin_8.fits --nproc 20 > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/prodrun_mask_xciv_skewers.log &