## CIV_forest <br>
Codes to model the enrichment of the IGM, generate skewers of the CIV forest with Nyx and Cloudy simulations, analyze their auto-correlation, model and mask CGM absorbers, and perform forecasting on the model parameters with MCMC.  

### Main scripts: <br>

1. ```cloudy_run/cloudy_grid*.in``` - scripts to run <a href="https://trac.nublado.org">Cloudy</a> models <br>
2. ```cloudy_run/cloudy_utils.py``` - functions to automatically generate cloudy scripts, read in cloudy outputs, and compute the ionization fraction of a metal ion <br>
3. ```halos_skewers.py``` - extract the halo catalogs from Nyx simulation and create models of the metal distribution <br>
4. ```metal_frac_skewers.py``` - create skewers of ionic fraction by interpolating Cloudy outputs on Nyx skewers <br>
5. ```prodrun_create_tau_skewers.py``` - create the metal-line forest skewers for the uniform enrichment model <br>
6. ```prodrun_metal_skewers.enrichment.py``` - create the metal-line forest skewers for the inhomogeneous enrichment model <br>
7. ```metal_corrfunc.py``` - compute the CIV correlation function <br>
8. ```civ_cgm.py``` - modeling the abundance of CGM absorbers based on existing observations <br>
9. ```civ_find_new.py``` - automated detection and masking of CGM absorbers <br>
10. ```compute_model_grid_civ_new.py``` - generate mock datasets and compute their covariances for all models (note: this code lives at https://github.com/enigma-igm/enigma/enigma/reion_forest/) <br>
11. ```mcmc_inference.py``` - performing inference using MCMC sampler <br>
12. ```misc.py``` - various convenience functions <br>
