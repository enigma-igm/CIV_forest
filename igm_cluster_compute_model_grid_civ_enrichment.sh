now=`date`
echo "Start: $now"

python ~/codes/enigma/enigma/reion_forest/compute_model_grid_civ_new.py \
--nproc 50 --fwhm 10 --samp 3 --SNR 50 --nqsos 20 --delta_z 1.0 \
--vmin 10 --vmax 2000 --dv 10 \
--ncovar 1000000 --nmock 500 --seed 1259761 \
--logZmin -4.5 --logZmax -2.0 --nlogZ 26

now=`date`
echo "Finish: $now"

# April 5, 2021:
# nohup ./igm_cluster_compute_model_grid_civ_enrichment.sh > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/igm_cluster_compute_model_grid_civ_enrichment_fine.log &

# Requested path length for nqsos=20 covering delta_z=1.000 corresponds to requested total dz_req = 20.000,
# or 92.743 total skewers. Rounding to 93 or dz_tot=20.055."

# ran in IGM on March 5, 2021:
# nohup ./igm_cluster_compute_model_grid_civ_enrichment.sh > /mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/igm_cluster_compute_model_grid_civ_enrichment.log &