#%% Imports
import sys
import os
import json
import S1_voids as s1_voids

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
    'release': 'PR4',                   # 'PR3' or 'PR4'
    'N_seeds': 10,                      # Number of random seeds used to identify voids. 
                                        # If None, it uses the simplest void catalog without random seeds. Nmax = 100
    
    'delta_value': None,                # If None = no filter. 
                                        # If it is positive only voids with delta_LOS > delta_value are considered. 
                                        # If it is negative, only voids with delta_LOS < delta_value are considered
    
    'zmin': 0.05, 'zmax': 0.3,        # zmin = 0.051 zmax = 0.583 
    'rmin': 35.0, 'rmax': 62.7,         # Mpc/h , rmin=35 rmax=62.7
    
    # Geometric setup
    'max_Rvoid': 2.5,                  
    'Rvoid_bin': 0.1,        
    'npix_stamp': 400,                  # Number of pixels in the stamp (square) for stacking 
    'filter_mode': 'wiener',            # 'none', 'gaussian', 'wiener'          
    'smooth_value_arcmin': 40.0,        # Just used if filter_mode is 'gaussian'
    'sigma_miscentering': 0.0,

    # Binning setup
    'binning_mode': 'redshift',         # 'redshift', 'radius'
    'n_bins': 1,                        

    # Fitting setup
    'rmin_fit_mpc': 0.5,  
    'rmax_fit_mpc': 10.0, 
    
    # Error estimation setup
    'exec_mode': 'errors',              # 'no_errors' or 'errors'
    'n_subsamples': 30,                 # Number of jackknife subsamples for error estimation if 'exec_mode' is 'errors'           
    'n_rand_factor': 1,                # Number of random positions for cosmic variance estimation, as a factor of the number of voids (e.g. 10 means 10 randoms per void)
    'n_rotations': 1,                  # Number of random rotations of the cmb map for cosmic variance estimation
    
    # MCMC setup
    'mcmc_walkers': 32, 
    'mcmc_steps': 2000, 
    'mcmc_discard': 500,

    # Step control
    'run_step_1': True, 
    
    '''
    # Not implemented yet for the analysis of the CMB, but the structure is left here for future implementation of the fit and MCMC steps.
    'run_step_2': False, 
    'run_step_3': False,
    '''

    'force_rerun': True
}

#%% Auxiliary functions
def main():
    
    log_path = os.path.join(config['output_folder'], "pipeline_run.log")
    sys.stdout = DualLogger(log_path)
    
    title = "WEN-HAN VOIDS x CMB LENSING PROFILES"
    print(f"### PIPELINE RUN: {title} ###")
    print(f"Output: {config['output_folder']}")

    with open(os.path.join(config['output_folder'], "WHVoidsCMB_pipeline_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    try:

        if config['run_step_1']:
            print('\n>>> STEP 1: STACKING (Lensing)')
            s1_voids.run_pipeline(config)

        ''' 
        # Not implemented yet for the analysis of the CMB, but the structure is left here for future implementation of the fit and MCMC steps.

        if config['run_step_2']:
            print('\n>>> STEP 2: FIT')
            s2_voids.run_pipeline(config)

        if config['run_step_3']:
            print('\n>>> STEP 3: MCMC')
            s3_voids.run_pipeline(config)
        ''' 

        with open(os.path.join(config['output_folder'], "SUCCESS"), 'w') as f: f.write("Done.")
        print("\n=== FINISHED ===")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
