#%% Imports
import sys
import os
import json
import numpy as np
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
    'zmin': 0.2, 'zmax': 0.65,          # zmax = 0.6 
    'rmin': 35.0, 'rmax': 150.0,        # Mpc/h , rmin=35
    
    # Geometric setup
    'physical_size_Mpc': 25.0,  
    'physical_bin': 1.5,        
    'npix_stamp': 400,          
    'smooth_value_arcmin': 0.0,         # Arcmin, 0 = No smoothing, >0 = CMB Gaussian smoothing kernel 
    'sigma_miscentering': 0.0,

    # Binning setup
    'binning_mode': 'richness',         # 'richness', 'redshift'
    'n_bins': 4,                        

    # Fitting setup
    'rmin_fit_mpc': 0.5,  
    'rmax_fit_mpc': 10.0, 
    
    # Error estimation setup
    'exec_mode': 'errors',              # 'full' or 'errors' (only error estimation, no fit)       
    'n_subsamples': 20,                 # Number of jackknife subsamples for error estimation           
    'n_rand_factor': 10,                # Number of random points per real void for error estimation (if applicable)          
    
    # MCMC setup
    'mcmc_walkers': 32, 
    'mcmc_steps': 2000, 
    'mcmc_discard': 500,

    # Step control
    'run_step_1': False, 
    'run_step_2': False, 
    'run_step_3': False,
    'force_rerun': True
}

#%% Auxiliary functions
def get_run_folder_path(cfg):
    
    reso = 2 * cfg['physical_size_Mpc'] / cfg['npix_stamp']
    bins_tag = f"{cfg['n_bins']}bins"

    folder_name = (f"{cfg['release']}_{bins_tag}_"
                   f"{cfg['exec_mode']}_{cfg['zmin']}_{cfg['zmax']}_"
                   f"{cfg['rmin']}_{cfg['rmax']}_"
                   f"{cfg['physical_size_Mpc']:.1f}Mpc_{reso}Mpcperpix")
    return os.path.join(cfg['output_folder'], folder_name)

def main():
    run_folder = get_run_folder_path(config)
    if not os.path.exists(run_folder): os.makedirs(run_folder)

    sys.stdout = DualLogger(os.path.join(run_folder, "WHVoidsCMB_pipeline_log.txt"))
    
    title = "WEN-HAN (M500c)"
    print(f"### PIPELINE RUN: {title} ###")
    print(f"Output: {run_folder}")
    
    with open(os.path.join(run_folder, "WHVoidsCMB_pipeline_config.json"), 'w') as f:
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
        with open(os.path.join(run_folder, "SUCCESS"), 'w') as f: f.write("Done.")
        print("\n=== FINISHED ===")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
