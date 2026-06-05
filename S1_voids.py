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
    zmin, zmax = config['zmin'], config['zmax']
    rmin, rmax = config['rmin'], config['rmax']

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
    delta_23_value = config.get('delta_value', None)
    
    if delta_23_value is None:
        delta_label = "d23_all"
    elif delta_23_value > 0:
        delta_label = f"d23_gt{delta_23_value:.2f}"
    else:
        delta_label = f"d23_lt{abs(delta_23_value):.2f}"

    filter_label = config.get('filter_mode', 'none')
    if filter_label == 'gaussian':
        filter_label = f'gaussian_{smooth_value_deg:.1f}deg'
    elif filter_label == 'wiener':
        filter_label = 'wiener'
    else:
        filter_label = 'no_filter'

    file_suffix = (f'{release}_{mode_label}_{exec_mode}_'
                   f'{zmin}_{zmax}_{rmin}_{rmax}_'
                   f'maxRv{max_Rvoid:.1f}_{reso_rv_per_pix}Rvperpix_'
                   f'{delta_label}_{filter_label}')
    
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

    filter_mode = config.get('filter_mode', 'none')
    if filter_mode == 'wiener':
        nlkk_file = f'{data_folder}CMB/Lensing/nlkk_PR3_MV.dat'
        cmb_alm_filtered, W_ell = fm.apply_wiener_filter(cmb_alm, nlkk_file, lmax=2048)
        lensing_map = hp.alm2map(cmb_alm_filtered, nside=nside)
        print('CMB map filtered with Wiener filter.')
    elif filter_mode == 'gaussian' and smooth_value_deg > 0:
        lensing_map = hp.smoothing(hp.alm2map(cmb_alm, nside=nside), fwhm=np.radians(smooth_value_deg))
        print(f'CMB map smoothed with Gaussian kernel of FWHM={smooth_value_deg:.1f} deg.')
    else:
        lensing_map = hp.alm2map(cmb_alm, nside=nside)
        print('CMB map without additional filtering applied.')
    print('')

    #%% Reading and selecting voids data
    print('Reading Wen-Han voids catalogue...')
    n_seeds = config.get('N_seeds', None)

    def apply_delta_23_filter(df, delta_value):
        if delta_value is None:
            return df
        elif delta_value >= 0:
            return df[df['delta_23'] > delta_value]
        else:
            return df[df['delta_23'] < delta_value]

    if n_seeds is not None:
        print(f"Reading {n_seeds} voids catalogues identified with different random seeds...")
        col_names = ['R_void', 'l', 'b', 'z', 'x', 'y', 'z_cart', 'delta_int', 'delta_23', 'completeness', 'delta_LOS']
        voids_data = {}
        final_data = {}
        for seed in range(n_seeds):
            file_path = f'{data_folder}/CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{(seed+1):03d}.dat'
            voids_data[seed] = pd.read_csv(file_path, sep='\s+', names=col_names, header=None)
            
            base_filter = (voids_data[seed]['z'] >= zmin) & (voids_data[seed]['z'] < zmax) & (voids_data[seed]['R_void'] >= rmin) & (voids_data[seed]['R_void'] <= rmax) & (voids_data[seed]['completeness'] == 2)
            filtered_data = voids_data[seed][base_filter].copy()
            final_data[seed] = apply_delta_23_filter(filtered_data, delta_23_value)
        print('All voids data loaded.\n')
    else:
        col_names = ['R_void', 'l', 'b', 'z']
        voids_data_raw = pd.read_csv(f'{data_folder}/CATALOGOS/WenHan_voids.dat', sep='\s+', names=col_names, header=None)
        final_data = voids_data_raw[
            (voids_data_raw['z'] >= zmin) & (voids_data_raw['z'] < zmax) & 
            (voids_data_raw['R_void'] >= rmin) & (voids_data_raw['R_void'] <= rmax)
        ].copy()
        print(f'Total voids: {len(final_data)}')
        print('Voids data loaded.\n')

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
                bins_frac=bins_frac,
                lensing_map=lensing_map, 
                common_mask=common_mask, 
                stacks_cache_folder=stacks_cache_folder, 
                n_random_factor=random_factor,
                n_rotations=n_rotations, 
                n_subsamples=n_subsamples,
                delta_label=delta_label,
                filter_label=filter_label,
                force_rerun=config.get('force_rerun', False)
            )

            result.update({
                'bin_id': info['id'],
                'binning_mode': binning_mode,
                'is_multi_seed': False
                })
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
                    npix_stamp=npix_stamp, nside=nside, bins_frac=bins_frac, 
                    lensing_map=lensing_map, common_mask=common_mask, 
                    stacks_cache_folder=seed_cache_folder, n_random_factor=random_factor,
                    n_rotations=n_rotations, n_subsamples=n_subsamples, delta_label=delta_label, filter_label=filter_label, force_rerun=config.get('force_rerun', False)
                )
                result['seed'] = seed
                seed_results.append(result)
            
            if len(seed_results) > 0:
                z_min_combined = np.min([sd['z_range'][0] for s in info['data'].values() for sd in [s] if len(sd['data']) > 0])
                z_max_combined = np.max([sd['z_range'][1] for s in info['data'].values() for sd in [s] if len(sd['data']) > 0])
                all_data_bin = pd.concat([info['data'][s]['data'] for s in range(n_seeds) if len(info['data'][s]['data']) > 0], ignore_index=True)
                all_l = all_data_bin['l'].values
                all_b = all_data_bin['b'].values
                coords_combined = (all_l, all_b)

                print(f'Combined stack: {len(all_data_bin)} total voids across {n_seeds} seeds')

                combined_cache_folder = os.path.join(stacks_cache_folder, 'combined')
                if not os.path.exists(combined_cache_folder):
                    os.makedirs(combined_cache_folder)

                result_combined = fm.process_bin_stacking(
                    release=release, mode=exec_mode,
                    z_min=z_min_combined, z_max=z_max_combined,       data_sample_bin=all_data_bin,
                    coords_bin=coords_combined,
                    max_Rvoid=max_Rvoid, npix_stamp=npix_stamp, nside=nside,        bins_frac=bins_frac, lensing_map=lensing_map,        common_mask=common_mask,        stacks_cache_folder=combined_cache_folder,        n_random_factor=random_factor, n_rotations=n_rotations,        n_subsamples=n_subsamples,
                    delta_label=delta_label, filter_label=filter_label,        force_rerun=config.get('force_rerun', False))
                
                result_combined.update({
                    'bin_id': bin_id,
                    'binning_mode': binning_mode,
                    'is_multi_seed': True,
                    'seed_results': seed_results
                    })
                bin_results_list.append(result_combined)         

    # SAVING
    print('Saving results...')

    output_plot_path = os.path.join(run_folder, f'Stacked_Maps_NullTests_{file_suffix}.pdf')
    fm.plot_stacked_maps_and_profiles(bin_results_list, output_plot_path, max_Rvoid)

    jk_plot_path = os.path.join(run_folder, f'JK_Profile_Correlation_{file_suffix}.pdf')
    fm.plot_jackknife_and_correlation(bin_results_list, jk_plot_path, max_Rvoid)

    if any(d.get('is_multi_seed', False) for d in bin_results_list):
        seeds_plot_path = os.path.join(run_folder, f'Seed_Consistency_{file_suffix}.pdf')
        fm.plot_seed_consistency(bin_results_list, seeds_plot_path, max_Rvoid)

    data_save_path = os.path.join(run_folder, f'Data_FullRun_{file_suffix}.pkl')
    with open(data_save_path, 'wb') as f: 
        pickle.dump({'bins_data': bin_results_list, 'parameters': config}, f)
        print(f"Data saved in: {data_save_path}")

if __name__ == "__main__":
    print("Please run from PIPELINE_SCRIPT.py")