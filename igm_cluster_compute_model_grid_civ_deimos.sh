now=`date`
echo "Start: $now"

python ~/codes/enigma/enigma/reion_forest/compute_model_grid_civ.py --nproc 31 --fwhm 60 --samp 3 --SNR 50 --nqsos 20 --delta_z 1.0 --vmin 20 --vmax 2000 --dv 20 --ncovar 1000000 --nmock 500 --seed 1259761 --logZmin -5.0 --logZmax -2.0 --nlogZ 31

now=`date`
echo "Finish: $now"
