#%% IMPORTS
import os
import numpy as np
import healpy as hp
import pandas as pd
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord
import Functions_module as fm

#%% Run pipeline function
def run_pipeline(config):
    data_folder = config['data_folder']
    output_folder = config['output_folder']
    release = config['release']
    zmin = config['zmin']
    zmax = config['zmax']
    rmin = config['rmin']
    rmax = config['rmax']

    max_Rvoid = config['max_Rvoid']
    Rvoid_bin = config['Rvoid_bin']
    npix_stamp = config['npix_stamp']
    
    bins_frac = np.arange(0, max_Rvoid + Rvoid_bin, Rvoid_bin)
    reso_rv_per_pix = (2 * max_Rvoid) / npix_stamp
    smooth_value_deg = config.get('smooth_value_arcmin', 0.0) / 60.0

    binning_mode = config['binning_mode']
    n_bins_quantile = config['n_bins']
    exec_mode = config['exec_mode']
    n_subsamples = config['n_subsamples']
    random_factor = config.get('n_rand_factor', 10)
    n_rotations = config.get('n_rotations', 10)
    
    mode_label = f"{binning_mode}_{n_bins_quantile}bins"
    file_suffix = f'{release}_{mode_label}_{exec_mode}_{zmin}_{zmax}_{rmin}_{rmax}_maxRv{max_Rvoid:.1f}_{reso_rv_per_pix}Rvperpix'
    
    run_folder = os.path.join(output_folder, file_suffix)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        
    stacks_cache_folder = os.path.join(output_folder, "Cache_Stacks/")
    if not os.path.exists(stacks_cache_folder):
        os.makedirs(stacks_cache_folder)
    
    print('######### CMB LENSING PROFILES USING WEN HAN ET AL. 2024 VOIDS CATALOGUE #########')
    print(f'Configuration: Release={release} | Mode={exec_mode} | Binning={binning_mode}')
    print(f'Output Run Folder: {run_folder}')
    print('')

    #%% CMB map and masks
    print(f'Reading {release} CMB Convergence map...')
    nside = 2048
    klm_file = f'{data_folder}CMB/Lensing/KAPPA_{release}klm_MV.fits'
    common_mask_file = f'{data_folder}CMB/Lensing/Common_mask_PR4Lensing_2048.fits'
    
    cmb_alm = hp.fitsfunc.read_alm(klm_file, hdu=1, return_mmax=False)
    common_mask = hp.read_map(common_mask_file)      

    if smooth_value_deg > 0.:
        lensing_map = hp.alm2map(cmb_alm, nside, fwhm=smooth_value_deg*np.pi/180.)
    else:
        lensing_map = hp.alm2map(cmb_alm, nside)
    print('CMB map and mask loaded.\n')
    print('')

    #%% Reading and selecting voids data
    print('Reading Wen-Han voids catalogue...')
    n_seeds = config.get('N_seeds', None)
    delta_LOS = config.get('delta_LOS', 0.)
    if n_seeds is not None:
        print(f"Reading {n_seeds} voids catalogues identified with different random seeds...")
        col_names = ['R_void', 'l', 'b', 'z', 'x', 'y', 'z_cart', 'delta_int', 'delta_23', 'completeness', 'delta_LOS']
        voids_data = {}
        final_data = {}
        for seed in range(n_seeds):
            file_path = f'{data_folder}/CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{(seed+1):03d}.dat'
            voids_data[seed] = pd.read_csv(file_path, sep='\s+', names=col_names, header=None)
            final_data[seed] = voids_data[seed][
                (voids_data[seed]['z'] >= zmin) & (voids_data[seed]['z'] < zmax) & 
                (voids_data[seed]['R_void'] >= rmin) & (voids_data[seed]['R_void'] <= rmax) & (voids_data[seed]['completeness'] == 2) & 
                (voids_data[seed]['delta_LOS'] < delta_LOS)
                ].copy()
            print(f'Seed {seed+1}: Total voids = {len(final_data[seed])}')
        print('All voids data loaded.\n')
    else:
        col_names = ['R_void', 'l', 'b', 'z']
        voids_data_raw = pd.read_csv(f'{data_folder}/CATALOGOS/WenHan_voids.dat', sep='\s+', names=col_names, header=None)
        final_data = voids_data_raw[
        (voids_data_raw['z'] >= zmin) & (voids_data_raw['z'] < zmax) & 
        (voids_data_raw['R_void'] >= rmin) & (voids_data_raw['R_void'] <= rmax)].copy()
        print(f'Total voids: {len(final_data)}')
        print('Voids data loaded.\n')
    print('')

    #%% BINNING DATA
    print(f'\nBinning data...')
    
    if binning_mode == 'redshift': metric_col = 'z'
    elif binning_mode == 'radius': metric_col = 'R_void'
    
    bins_info_list = []
    
    if n_seeds is None:
        final_data['bin_id'] = pd.qcut(final_data[metric_col], n_bins_quantile, labels=False)
        for i in sorted(final_data['bin_id'].unique()):
            subset = final_data[final_data['bin_id'] == i]
            if len(subset) == 0: continue
            info = {
                'id': int(i),
                'data': subset,
                'z_range': (subset['z'].min(), subset['z'].max()),
                'count': len(subset),
                'coords': SkyCoord(l=subset['l'].values*u.degree, b=subset['b'].values*u.degree, frame='galactic')
            }
            bins_info_list.append(info)
            print(f'Bin {i+1}: N={len(subset)}')
    else:
        for seed in range(n_seeds):
            final_data[seed]['bin_id'] = pd.qcut(final_data[seed][metric_col], n_bins_quantile, labels=False)
        for i in range(n_bins_quantile):
            bin_info = {
                'id': int(i),
                'data': {},
                'count_mean': 0
            }
            counts = []
            for seed in range(n_seeds):
                subset = final_data[seed][final_data[seed]['bin_id'] == i]
                counts.append(len(subset))
                bin_info['data'][seed] = {
                    'data': subset,
                    'z_range': (subset['z'].min(), subset['z'].max()),
                    'coords': SkyCoord(l=subset['l'].values*u.degree, b=subset['b'].values*u.degree, frame='galactic') if len(subset) > 0 else None
                }
            bin_info['count_mean'] = np.mean(counts)
            bins_info_list.append(bin_info)
            print(f'Bin {i+1}: Mean N={bin_info["count_mean"]:.1f} across {n_seeds} seeds')

    print(' ')
    print(f'Binning completed.')
    print('')

    #%% LOOP PRINCIPAL
    print(f"\n######### Doing profiles in mode: {exec_mode} #########\n")
    
    bin_results_list = []

    if n_seeds is None:
        for info in bins_info_list:
            data_bin = info['data']
            gal_coords = info['coords']
            coords_bin = (gal_coords.l.degree, gal_coords.b.degree)
            z_bin_min, z_bin_max = info['z_range']
        
            result = fm.process_bin_stacking(
                release=release,
                mode=exec_mode,
                z_min=z_bin_min, 
                z_max=z_bin_max, 
                data_sample_bin=data_bin, 
                coords_bin=coords_bin,
                max_Rvoid=max_Rvoid, 
                npix_stamp=npix_stamp, 
                nside=nside, 
                smooth_value=smooth_value_deg, 
                bins_frac=bins_frac,
                lensing_map=lensing_map, 
                common_mask=common_mask, 
                stacks_cache_folder=stacks_cache_folder, 
                n_random_factor=random_factor,
                n_rotations=n_rotations, 
                n_subsamples=n_subsamples
            )

            result['bin_id'] = info['id']
            result['binning_mode'] = binning_mode
            result['is_multiple_seeds'] = False
            bin_results_list.append(result)
    else:
        for info in bins_info_list:
            bin_id = info['id']
            print(f"--- Processing Bin {bin_id+1} for all {n_seeds} seeds ---")
            
            seed_results = []
            for seed in range(n_seeds):
                seed_info = info['data'][seed]
                data_bin = seed_info['data']
                gal_coords = seed_info['coords']
                
                if gal_coords is None or len(data_bin) == 0:
                    continue
                    
                coords_bin = (gal_coords.l.degree, gal_coords.b.degree)
                z_bin_min, z_bin_max = seed_info['z_range']
                
                seed_cache_folder = os.path.join(stacks_cache_folder, f"seed_{seed}")
                if not os.path.exists(seed_cache_folder):
                    os.makedirs(seed_cache_folder)
                
                result = fm.process_bin_stacking(
                    release=release, mode=exec_mode, z_min=z_bin_min, z_max=z_bin_max, 
                    data_sample_bin=data_bin, coords_bin=coords_bin, max_Rvoid=max_Rvoid, 
                    npix_stamp=npix_stamp, nside=nside, smooth_value=smooth_value_deg, 
                    bins_frac=bins_frac, lensing_map=lensing_map, common_mask=common_mask, 
                    stacks_cache_folder=seed_cache_folder, n_random_factor=random_factor,
                    n_rotations=n_rotations, n_subsamples=n_subsamples
                )
                result['seed'] = seed
                seed_results.append(result)
            
            if len(seed_results) > 0:
                combined_result = {
                    'bin_id': bin_id,
                    'binning_mode': binning_mode,
                    'is_multi_seed': True,
                    'seed_results': seed_results,
                    'z_mean': np.mean([r['z_mean'] for r in seed_results if 'z_mean' in r]),
                    'map': np.mean([r['map'] for r in seed_results], axis=0),
                    'r_frac': seed_results[0]['r_frac'],
                    'null_rand_mean': np.mean([r['null_rand_mean'] for r in seed_results], axis=0),
                    'null_rand_std': np.mean([r['null_rand_std'] for r in seed_results], axis=0),
                    'null_rot_mean': np.mean([r['null_rot_mean'] for r in seed_results], axis=0),
                    'null_rot_std': np.mean([r['null_rot_std'] for r in seed_results], axis=0)
                }
                bin_results_list.append(combined_result)
           

    # SAVING
    print('Saving results...')
    
    output_plot_path = os.path.join(run_folder, f'Voids_Lensing_Profiles_{file_suffix}.pdf')
    fm.plot_results(bin_results_list, output_plot_path, smooth_value_deg, max_Rvoid)

    data_save_path = os.path.join(run_folder, f'Data_FullRun_{file_suffix}.pkl')
    save_dict = {'bins_data': bin_results_list, 'parameters': config}
    
    with open(data_save_path, 'wb') as f: pickle.dump(save_dict, f)
    print(f"Data saved in: {data_save_path}")

if __name__ == "__main__":
    print("Please run from PIPELINE_SCRIPT.py")