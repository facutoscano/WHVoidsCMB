import sys
import os
import json
import numpy as np

import S1_voids as s1_voids

from astropy.cosmology import Planck18
from astropy import units as u

class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

# ==========================================
# CONFIGURACIÓN
# ==========================================
config = {
    'main_folder': '/home/ftoscano/trabajo/Doctorado/WenHanClusters/',
    'data_folder': '/home/ftoscano/trabajo/Doctorado/Data/',
    'output_folder': '/home/ftoscano/trabajo/Doctorado/WenHanClusters/Results/',

    'DES_verification': False,
    'release': 'PR4',  
    
    # --- SELECCIÓN DE DATOS ---
    'zmin': 0.2, 'zmax': 0.65,
    'lambda_min': 45, 'lambda_max': 200, 
    'z_type': 'spec',
    
    # --- GEOMETRÍA ---
    'physical_size_Mpc': 25.0,  
    'physical_bin': 1.5,        
    'npix_stamp': 400,          
    'smooth_value_arcmin': 0.0, 
    'sigma_miscentering': 0.0,

    # --- CORTE FIT (S2/S3) ---
    'rmin_fit_mpc': 0.5,  
    'rmax_fit_mpc': 10.0, 
    
    # --- SETUP BINNING / DENSITY ---
    # Opciones: 'richness', 'redshift', 'density'
    'binning_mode': 'richness',
    'n_bins': 4, # Ignorado si es density
    
    # Para density mode: None = Referencia (Calcula n), Float = Target (Corta por n)
    'target_density': None, 

    # --- SETUP tSZ  ---
    'tsz_ell': 32,          # [0, 2, 5, 10, 16, 32, 64, 128, 256, 512, 1024].
    'tsz_wavelet': None,   # None o valor en arcmin (int/float) si tsz_ell=10 [2, 4, 6, 8]

    # --- EJECUCIÓN ---
    'exec_mode': 'errors',       
    'n_subsamples': 20,           
    'n_rand_factor': 10,          
    
    'mcmc_walkers': 32, 'mcmc_steps': 2000, 'mcmc_discard': 500,

    'run_step_1': False, 'run_step_1_tSZ': True, 'run_step_2': False, 'run_step_3': False,
    'force_rerun': True
}

def get_run_folder_path(cfg):
    suffix_spec = "_spec" if cfg['z_type'] == 'spec' else ""
    reso = 2 * cfg['physical_size_Mpc'] / cfg['npix_stamp']
    prefix = "DES_" if cfg.get('DES_verification', False) else ""
    
    if cfg['binning_mode'] == 'density':
        dens_tag = "DensityRef" if cfg['target_density'] is None else "DensityTarget"
        bins_tag = dens_tag
    else:
        bins_tag = "FixedBins" if cfg.get('DES_verification', False) else f"{cfg['n_bins']}bins"

    folder_name = (f"{prefix}{cfg['release']}_{bins_tag}_"
                   f"{cfg['exec_mode']}_{cfg['zmin']}_{cfg['zmax']}_"
                   f"{cfg['lambda_min']}_{cfg['lambda_max']}_"
                   f"{cfg['physical_size_Mpc']:.1f}Mpc_{reso}Mpcperpix{suffix_spec}")
    return os.path.join(cfg['output_folder'], folder_name)

def main():
    # --- LÓGICA DE SEGURIDAD tSZ ---
    # Si se pide tSZ pero NO Lensing, deshabilitamos S2 y S3
    if config['run_step_1_tSZ'] and not config['run_step_1']:
        if config['run_step_2'] or config['run_step_3']:
            print("(!) AVISO: Se seleccionó sólo tSZ (sin Lensing). Pasos 2 (Fit) y 3 (MCMC) deshabilitados automáticamente.")
            config['run_step_2'] = False
            config['run_step_3'] = False

    run_folder = get_run_folder_path(config)
    if not os.path.exists(run_folder): os.makedirs(run_folder)

    sys.stdout = DualLogger(os.path.join(run_folder, "pipeline_log.txt"))
    
    title = "DES Y3 (M200m)" if config.get('DES_verification') else "WEN-HAN (M500c)"
    print(f"### PIPELINE RUN: {title} ###")
    print(f"Output: {run_folder}")
    
    with open(os.path.join(run_folder, "pipeline_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    try:
        is_des = config.get('DES_verification', False)
        
        if config['run_step_1']:
            print('\n>>> STEP 1: STACKING (Lensing)')
            if is_des: s1_des.run_pipeline(config, run_folder_override=run_folder)
            else: s1.run_pipeline(config)

        if config['run_step_1_tSZ']:
            print('\n>>> STEP 1: STACKING (tSZ: Compton-Y & Free-SZ)')
            s1_tsz.run_pipeline(config)

        if config['run_step_2']:
            print('\n>>> STEP 2: FIT')
            if is_des: s2_des.run_pipeline(config)
            else: s2.run_pipeline(config)

        if config['run_step_3']:
            print('\n>>> STEP 3: MCMC')
            if is_des: s3_des.run_pipeline(config)
            else: s3.run_pipeline(config)

        with open(os.path.join(run_folder, "SUCCESS"), 'w') as f: f.write("Done.")
        print("\n=== FINISHED ===")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()
