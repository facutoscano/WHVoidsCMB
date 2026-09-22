#%% Imports
import sys
import os
import json
import importlib

#%% Logger
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

#%% Config
config = {
    'main_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/',
    'data_folder': '/home/ftoscano/Doctorado/Data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/',


    # Data selection
    'cmb_map': 'act-pr4',               # one map: 
                                        # 'PLANCK_PR4' | 'PLANCK_PR3' | 'PLANCK_CIB' |
                                        # 'PLANCK_SZ'  | 'PLANCK_SZ_deproj' | 'PLANCK_inhom' | 'ACT'
                                        # full analysis:
                                        #  'All_Planck' (PR3 + CIB + inhom + Sz + Szdeproj + PR4)
                                        #  'All' ('All_Planck' + ACT)
                                        # comparison:
                                        #  'act-pr4' (PR4 vs ACT, ACT footprint)
    
    'min_footprint_coverage': 0.9,      # voids which disk max_Rvoid*Rv is min_footprint_coverage*100 within the mask
    'void_catalog': 'WH',               # 'WH' (Wen&Han --> galactic l,b) or 'BOSS' (BOSS --> equatorial ra,dec)
    'N_seeds': 100,                     # Number of random seeds used to identify voids.
                                        # If None, it uses the simplest void catalog without random seeds. Nmax = 100
    
    'delta_value': None,                # If None = no filter.
                                        # If it is positive only voids with delta_23 > delta_value are considered.
                                        # If it is negative, only voids with delta_23 < delta_value are considered.

    'seed_mode': 'merge',               # 'concat' (concatenate the N_seeds voids) |
                                        # 'merge'  (DBSCAN single catalogue) | 
                                        # 'both'   (compare both cases)

    'merge_eps_mpch': 8.0,              # DBSCAN linking length [Mpc/h] for the merge
    'merge_min_frac': 0.4,              # min fraction of seeds to keep a void (min_samples = merge_min_frac * N_seeds)
    'merge_use_catalog_xyz': False,     # False -> recompute comoving xyz from (l,b,z); 
                                        # True  -> use catalogue x,y,z_cart
    
    'zmin': 0.1, 'zmax': 0.5,            
    'rmin': 35.0, 'rmax': 70.0,         

    
    # Geometric setup
    'max_Rvoid': 2.5,                  
    'Rvoid_bin': 0.14,        
    'npix_stamp': 400,                  # Number of pixels in the stamp (square) for stacking 
    'filter_mode': 'wiener',            # 'none' | 
                                        # 'gaussian' |
                                        # 'wiener' (expected nlkk file)          
    'smooth_value_arcmin': 0.0,         # Just used if filter_mode is 'gaussian'
    'act_smooth_arcmin': 0.0,           # Extra gaussian (FWHM arcmin) over ACT 
                                        # 0.0 implies not extra filter


    # Binning setup
    'binning_mode': 'redshift',         # 'redshift' |
                                        # 'radius'
    'n_bins': 1,                        

   
    # Error estimation setup
    'exec_mode': 'errors',              # 'no_errors' |
                                        # 'errors'
    'n_subsamples': 60,                 # Number of jackknife subsamples for error estimation if 'exec_mode' is 'errors'           
    'n_rand_factor': 300,               # Number of realizations of random positions for cosmic variance estimation
    'n_rotations': 30,                  # Number of random rotations of the cmb map for cosmic variance estimation
    'random_pool': 'full',              # 'full'  -> randoms over the whole CMB-lensing footprint (common_mask)
                                        # 'survey'-> randoms restricted to the void survey footprint 
    'random_excl_factor': None,         # Exclude disks of (random_excl_factor * Rv) around real voids from the random pool
                                        # 0/None disables.


    # Cluster setup
    'n_workers': 120,                   # Number of worker processes
                                        # None = all CPU cores
                                        # 1 = serial mode

    # Step control
    'run_step_1':  True,
    'force_rerun': True
}

#%% Auxiliary functions
def main():
    
    lensing_out = os.path.join(config['output_folder'], 'lensing')
    os.makedirs(lensing_out, exist_ok=True)

    log_path = os.path.join(lensing_out, "pipeline_run.log")
    sys.stdout = DualLogger(log_path)

    title = "VOIDS x CMB LENSING PROFILES"
    print(f"### PIPELINE RUN: {title} ###")
    print(f"Output: {lensing_out}")

    with open(os.path.join(lensing_out, "VoidsCMB_pipeline_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    step1_module = 'S1_voids_parallel'
    s1_voids = importlib.import_module(step1_module)

    try:

        if config['run_step_1']:
            print('\n>>> STACKING (Lensing)')
            s1_voids.run_pipeline(config)

        with open(os.path.join(lensing_out, "SUCCESS"), 'w') as f: f.write("Done.")
        print("\n=== FINISHED ===")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
