now=`date`
echo "Start: $now"

python ~/codes/enigma/enigma/reion_forest/compute_model_grid_civ_new_scaling.py \
--nproc 1 --fwhm 10 --samp 3 --SNR 50 --nqsos 40 --delta_z 1.0 \
--vmin 10 --vmax 2000 --dv 10 \
--ncovar 1000000 --nmock 500 --seed 1259761 \
--logZmin -3.5 --logZmax -3.4 --nlogZ 1

now=`date`
echo "Finish: $now"