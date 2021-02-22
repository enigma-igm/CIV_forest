#python run_reion_skewers_metal.py --ranskewerfile nyx_sim_data/rand_skewers_z45_ovt_tau_xciv.fits --outfile nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits --dmax 3000 --metal_colname X_CIV --metal_mass 12

python run_reion_skewers_metal.py \
--ranskewerfile nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_r2.75000_logM9.00.fits \
--outfile nyx_sim_data/enrichment_models/rand_skewers_z45_ovt_tau_xciv_flux_r2.75000_logM9.00.fits \
--dmax 3000 \
--metal_colname X_CIV \
--metal_mass 12
