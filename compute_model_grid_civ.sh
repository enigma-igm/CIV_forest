now=`date`
echo "Start: $now"

python ~/Codes/enigma/enigma/reion_forest/compute_model_grid_civ.py --nproc 4 --fwhm 10 --samp 3 --SNR 100 --nqsos 20 --delta_z 0.2 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 1190 --nlogZ 9

now=`date`
echo "Finish: $now"

# 20 min run time for subset100 skewers
# 2.5 hrs run time for the full 100,000 skewers with "--SNR 100 --nqsos 20 --delta_z 0.2 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 1190 --nlogZ 9" ... bottlenck is computing 2PCF (which needs to be done 2x for noisy and noiseless data)
