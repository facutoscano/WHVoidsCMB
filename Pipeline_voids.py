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

    # ---------------- Data selection ----------------
    'cmb_map': 'ACT-PR4',            # one map:
                                        #   'PLANCK_PR4' | 'PLANCK_PR3' | 'PLANCK_CIBcomb' (lensing+CIB tracer) |
                                        #   'PLANCK_SZ' | 'PLANCK_SZ_deproj' | 'PLANCK_inhom' | 'ACT'
                                        # groups (same voids for every map: common coverage):
                                        #   'All_Planck' (PR4 + PR3 + CIBcomb + inhom + SZ + SZ_deproj)
                                        #   'All'        ('All_Planck' + ACT)
                                        #   'ACT-PR4'    (PR4 vs ACT on the ACT x PR4 footprint)
    'common_coverage': True,            # groups: select voids on the intersection of all the masks

    'void_catalog': 'BOTH',               # 'WH' (Wen & Han) | 'BOSS' | 'BOTH' (both + comparison in columns)
    'N_seeds': 100,                     # number of Sparkling seed catalogues (None = single catalogue)
    'seed_mode': 'merge',               # 'concat' | 'merge' (DBSCAN single catalogue) | 'both'
    'merge_eps_mpch': 8.0,              # DBSCAN linking length [Mpc/h]
    'merge_min_frac': 0.4,              # a void must appear in >= this fraction of the seeds
    'merge_use_catalog_xyz': False,     # False -> comoving xyz recomputed from (l,b,z)

    'delta_value': None,                # None: no cut | v>=0: delta_23 > v | v<0: delta_23 < v
    'zmin': 0.1, 'zmax': 0.5,
    'rmin': 35.0, 'rmax': 70.0,         # Mpc/h

    'min_footprint_coverage': 0.9,      # a void (and a random) is kept if >= 90% of its
                                        # max_Rvoid*Rv disc lies inside the coverage region
    'coverage_nside': 256,              # resolution used for coverage / random regions

    # ---------------- Geometry and filtering ----------------
    'max_Rvoid': 2.5,                   # stamp half-size in units of R_v
    'Rvoid_bin': 0.14,                  # radial bin width in units of R_v
    'npix_stamp': 400,                  # stamp side (capped so that the map is not oversampled)
    'filter_mode': 'wiener',            # 'none' | 'gaussian' | 'wiener' (not applied to pre-filtered maps)
    'smooth_value_arcmin': 0.0,         # FWHM, only for 'gaussian'
    'act_smooth_arcmin': 0.0,           # extra Gaussian FWHM on ACT (0 = none)

    # ---------------- Binning ----------------
    'binning_mode': 'redshift',         # 'redshift' | 'radius'
    'n_bins': 1,

    # ---------------- Errors, nulls and background ----------------
    'exec_mode': 'errors',              # 'errors' (jackknife) | 'no_errors'
    'n_subsamples': 60,                 # jackknife regions
    'n_rand_factor': 300,               # random realizations
    'n_rotations': 30,                  # map rotations (full-sky maps only)
    'random_pool': 'footprint',         # 'footprint': randoms inside map mask x survey footprint,
                                        #              with the same coverage criterion as the voids
                                        # 'mask'     : randoms inside the map mask only (same coverage criterion)
    'random_excl_factor': None,         # exclude discs of value*Rv around real voids (None disables)
    'bkg_mode': 'randoms',              # 'randoms'       : net = P - <P_randoms>
                                        # 'outer'         : net = P - mean(P, r/Rv in bkg_outer)
                                        # 'randoms+outer' : both
    'bkg_outer': (2.5, 3.0),            # r/R_v range used by 'outer'

    # ---------------- Cluster ----------------
    'n_workers': 120,                   # worker processes (None = all cores)

    # ---------------- Step control ----------------
    'run_step_1': True,
    'force_rerun': True,               # True: ignore the stacking cache
}


#%% Main
def main():
    lensing_out = os.path.join(config['output_folder'], 'lensing')
    os.makedirs(lensing_out, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(lensing_out, "pipeline_run.log"))

    print("### PIPELINE RUN: VOIDS x CMB LENSING PROFILES ###")
    print(f"Output: {lensing_out}")
    with open(os.path.join(lensing_out, "VoidsCMB_pipeline_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    s1_voids = importlib.import_module('S1_voids_parallel')
    if config['run_step_1']:
        print('\n>>> STACKING (Lensing)')
        s1_voids.run_pipeline(config)
    with open(os.path.join(lensing_out, "SUCCESS"), 'w') as f:
        f.write("Done.")
    print("\n=== FINISHED ===")


if __name__ == "__main__":
    main()
