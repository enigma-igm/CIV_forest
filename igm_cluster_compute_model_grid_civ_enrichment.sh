now=`date`
echo "Start: $now"

python ~/codes/enigma/enigma/reion_forest/compute_model_grid_civ_new.py \
--nproc 50 --fwhm 10 --samp 3 --SNR 50 --nqsos 20 --delta_z 1.0 \
--vmin 10 --vmax 2000 --dv 10 \
--ncovar 1000000 --nmock 500 --seed 1259761 \
--logZmin -4.5 --logZmax -2.0 --nlogZ 26

now=`date`
echo "Finish: $now"

# ran in IGM on March 5, 2021:
# nohup ./igm_cluster_compute_model_grid_civ_enrichment.sh > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/igm_cluster_compute_model_grid_civ_enrichment.log &