import inference_enrichment as infen
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse

#logM_guess, R_guess, logZ_guess = 9.5, 0.5, -4.0
#logM_guess, R_guess, logZ_guess = 9.5, 1.5, -4.0
#logM_guess, R_guess, logZ_guess = 9.5, 2.5, -4.0


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logMguess', type=float, required=True)
    parser.add_argument('--Rguess', type=float, required=True)
    parser.add_argument('--logZguess', type=float, required=True)
    parser.add_argument('--seed', type=int, default=5377192, required=False)

    args = parser.parse_args()
    seed = args.seed
    logM_guess, R_guess, logZ_guess = args.logMguess, args.Rguess, args.logZguess

    # default
    modelfile = 'nyx_sim_data/igm_cluster/enrichment_models/corrfunc_models/fine_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
    nlogM, nR, nlogZ = 251, 291, 251

    #new_dir = 'plots/enrichment/inference_enrichment_debug/paramspace_contour/seed_%d_%0.2f_%0.2f_%0.2f' % (seed, logM_guess, R_guess, logZ_guess)
    new_dir = 'plots/enrichment/inference_enrichment_debug/seed_%d_%0.2f_%0.2f_%0.2f' % (seed, logM_guess, R_guess, logZ_guess)
    if os.path.exists(new_dir):
        print("Directory already exists...saving everything here")
    else:
        print("Creating new directory...")
        os.mkdir(new_dir)

    init_out = infen.init(modelfile, logM_guess, R_guess, logZ_guess, seed)
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, covar_array, icovar_array, lndet_array, vel_corr, _, _, _ = init_out
    lnlike_coarse, lnlike_fine, _, logM_fine, R_fine, logZ_fine = infen.interp_likelihood(init_out, nlogM, nR, nlogZ, interp_lnlike=False, interp_ximodel=False)

    coarse_outcsv = new_dir + '/lnlike_coarse_arbinterp.csv'
    want_fine_csv = new_dir + '/allpts_to_interp.csv'
    arbinterp_outnpy = new_dir + '/lnlike_fine_arbinterp.npy'

    infen.do_arbinterp(logM_coarse, R_coarse, logZ_coarse, lnlike_coarse, coarse_outcsv, logM_fine, R_fine, logZ_fine, want_fine_csv, arbinterp_outnpy)


    ##### plotting the interpolated results #####

    lnlike_fine = np.load(arbinterp_outnpy)
    logM_fine = logM_fine[10:240] # (after truncating) logM_fine[0], logM_fine[-1] = 8.6, 10.89
    R_fine = R_fine[10:280] # R_fine[0], R_fine[-1] = 0.2, 2.89
    logZ_fine = logZ_fine[10:240] # logZ_fine[0], logZ_fine[-1] = -4.4, -2.11

    ilogZ_fine = find_closest(logZ_fine, logZ_data)
    ilogM_fine = find_closest(logM_fine, logM_data)
    iR_fine = find_closest(R_fine, R_data)

    # plot 1D likelihood at true mock data values
    #infen.plot_likelihood_data(lnlike_fine, logM_fine, R_fine, logZ_fine, logM_data, R_data, logZ_data, savefig=None)

    # plot contour plots
    levels = [0.68, 0.95] # None
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    try:
        inference.plot_contour_lnlike(R_fine, logZ_fine, lnlike_fine[ilogM_fine,:,:], R_data, logZ_data, 'R (Mpc)', 'logZ', levels=levels)
    except:
        print("pass")
        pass

    plt.subplot(132)
    try:
        inference.plot_contour_lnlike(logM_fine, logZ_fine, lnlike_fine[:, iR_fine, :], logM_data, logZ_data, 'logM', 'logZ', levels=levels)
    except:
        print("pass")
        pass

    plt.subplot(133)
    try:
        inference.plot_contour_lnlike(logM_fine, R_fine, lnlike_fine[:,:,ilogZ_fine], logM_data, R_data, 'logM', 'R (Mpc)', levels=levels)
    except:
        print("pass")
        pass

    plt.tight_layout()
    plt.savefig(new_dir + '/lnlike_contour_arbinterp.png')

if __name__ == '__main__':
    main()