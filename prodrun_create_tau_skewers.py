# Feb 24, 21: modified from enigma/reion_forest/nyx_skewers.py
# generate halo mask and the resulting metal skewers for a grid of enrichment models

from matplotlib import pyplot as plt
import os
import shutil
from IPython import embed
import argparse
import time
from subprocess import Popen

def parser():

    # double check
    parser = argparse.ArgumentParser(description='Create tau skewers for metal line forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--ranskewerfile', type=str, required=True, help="Name of random skewerfile containing the *masked* metal ion fraction skewers")
    #parser.add_argument('--halofile', type=str, required=True, help="Name of halo file")
    parser.add_argument('--nproc', type=int, required=False, help="Number of processes to run simultaneously")

    return parser.parse_args()

def main():

    #import numpy as np
    #from astropy.io import fits
    #from enigma.tpe.skewer_tools import random_skewers
    #from enigma.reion_forest import utils
    #from enigma.tpe.utils import calc_eosfit2
    #from astropy.table import Table, vstack, hstack
    import glob

    args = parser()
    nproc = args.nproc

    zstr = 'z45'
    outpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/' + zstr + '/enrichment_models/tau/'
    xciv_skewerpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/' + zstr + '/enrichment_models/xciv_mask/'

    all_files = glob.glob(xciv_skewerpath + '*.fits')
    print('Creating tau skewers for {:d} models'.format(len(all_files)))

    counter = 0
    counter_file = []

    for file in all_files:
        suffix = file.split(xciv_skewerpath + 'rand_skewers_z45_ovt_xciv_')[-1]
        tau_outfile = os.path.join(outpath, 'rand_skewers_' + zstr + '_ovt_xciv_tau_' + suffix)
        tau_logfile = os.path.join(outpath, 'rand_skewers_' + zstr + '_ovt_xciv_tau_' + suffix.split('.fits')[0] + '.log')

        command = 'python run_reion_skewers_metal.py ' + '--ranskewerfile ' + file + ' --outfile ' + tau_outfile + \
                  ' --dmax 3000 --metal_colname X_CIV --metal_mass 12 > %s' % tau_logfile

        p = Popen(command, shell=True)

        counter_file.append(tau_outfile)
        counter += 1
        print("counter now", counter)

        if nproc != None:
            if counter % nproc == 0:  # every n-th processes
                print("checking if all files exist")

                while True:
                    if all([os.path.isfile(f) for f in counter_file]):
                        counter_file = []
                        print("yep...proceeding")
                        break
                    else:
                        print('waiting....')
                        time.sleep(600)  # wait 10 min before checking because making tau skewers takes a while

if __name__ == '__main__':
    main()

# run command in IGM (2/27/21):
# nohup python prodrun_create_tau_skewers.py --nproc 30 > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/prodrun_create_tau_skewers.log &