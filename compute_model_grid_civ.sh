# 20 min to finish
now=`date`
echo "Start: $now"

python ~/Codes/enigma/enigma/reion_forest/compute_model_grid_civ.py --nproc 6 --fwhm 10 --samp 3 --SNR 40 --nqsos 20 --delta_z 0.2 --vmin 20 --vmax 2000 --dv 5 --ncovar 1000000 --nmock 500 --seed 1199 --nlogZ 9

now=`date`
echo "Finish: $now"
