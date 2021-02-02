now=`date`
echo "Start: $now"

python ~/Codes/enigma/enigma/reion_forest/compute_model_grid_civ.py --nproc 4 --fwhm 10 --samp 3 --SNR 50 --nqsos 20 --delta_z 1.0 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 12976 --nlogZ 1

# Previous runs:
#--nproc 4 --fwhm 10 --samp 3 --SNR 100 --nqsos 20 --delta_z 0.2 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 1190 --nlogZ 9

now=`date`
echo "Finish: $now"

# <10 min run time for subset100 skewers and 1 logZ model
# 2.5 hrs run time for the full 10,000 skewers with "--SNR 100 --nqsos 20 --delta_z 0.2 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 1190 --nlogZ 9" ... bottlenck is computing 2PCF (which needs to be done 2x for noisy and noiseless data)

# 1 hr for 10,000 skewers and 1 model on IGM cluster node
