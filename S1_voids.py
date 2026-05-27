#%% IMPORTS
import os
import sys
import numpy as np
import healpy as hp
import pandas as pd
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord
import Functions_module as fm

#%% Run pipeline function
def run_pipeline(config):
    main_folder = config['main_folder']
    data_folder = config['data_folder']
    output_folder = config['output_folder']
    release = config['release']
    zmin = config['zmin']
    zmax = config['zmax']
    rmin = config['rmin']
    rmax = config['rmax']

    physical_size_Mpc = config['physical_size_Mpc']
    physical_bin = config['physical_bin']
    npix_stamp = config['npix_stamp']
    
    bins_physical = int(physical_size_Mpc / physical_bin) 
    reso = 2 * physical_size_Mpc / npix_stamp
    smooth_value_deg = config.get('smooth_value_arcmin', 6.0) / 60.0

    binning_mode = config['binning_mode']
    n_bins_quantile = config['n_bins']
    EXECUTION_MODE = config['exec_mode']
    N_subsamples = config['n_subsamples']
    N_RAND_FACTOR = config['n_rand_factor']
    
    mode_label = f"{binning_mode}_{n_bins_quantile}bins"
    file_suffix = f'{release}_{mode_label}_{EXECUTION_MODE}_{zmin}_{zmax}_{rmin}_{rmax}_{physical_size_Mpc:.1f}Mpc_{reso}Mpcperpix'
    
    run_folder = os.path.join(output_folder, file_suffix)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        
    stacks_cache_folder = os.path.join(output_folder, "Cache_Stacks/")
    if not os.path.exists(stacks_cache_folder):
        os.makedirs(stacks_cache_folder)
    
    print('######### CMB LENSING PROFILES USING WEN HAN ET AL. 2024 VOIDS CATALOGUE #########')
    print(f'Configuration: Release={release} | Mode={EXECUTION_MODE} | Binning={binning_mode}')
    print(f'Output Run Folder: {run_folder}')
    
    #%% CMB map and masks
    print(f'Reading {release} CMB Convergence map...')
    nside = 2048
    klm_file = f'{data_folder}CMB/KAPPA_{release}klm_MV.fits'
    common_mask_file = f'{data_folder}CMB/Common_mask_Lensing_2048.fits'
    
    cmb_alm = hp.fitsfunc.read_alm(klm_file, hdu=1, return_mmax=False)
    common_mask= hp.read_map(common_mask_file)      

    if smooth_value_deg != 0.:
        lensing_map = hp.alm2map(cmb_alm, nside, fwhm=smooth_value_deg*np.pi/180.)
    else:
        lensing_map = hp.alm2map(cmb_alm, nside)
    print('CMB map and mask loaded.\n')
    print('')

    #%% Reading and selecting voids data
    print('Reading Wen-Han voids catalogue...')
    col_names = [
        'ID', 'Name', 'RAdeg', 'DEdeg', 'zCl', 'f_zCl', 
        'zmag', 'W1mag', 'log(M*)', 'r500', 'lambda500', 
        'M500', 'Ngal', 'Gamma', 'e_Gamma', 'Source', 'Cat'
    ]
    clusters_data_raw = pd.read_csv(f'{data_folder}/CATALOGOS/WenHan_Clusters.dat', sep='\s+', skiprows=56, names=col_names, header=None)
    
    final_data = clusters_data_raw[
        (clusters_data_raw['zCl'] >= zmin) & (clusters_data_raw['zCl'] < zmax) & 
        (clusters_data_raw['lambda500'] >= lambda_min) # Solo lambda min como seguridad
    ].copy()

    if z_type == 'spec':
        final_data = final_data[final_data['f_zCl'] == 1]
    
    print(f'Total clusters (Pre-Selection): {len(final_data)}')

    #%% DENSITY SELECTION LOGIC
    if binning_mode == 'density':
        print("\n>>> EJECUTANDO ABUNDANCE MATCHING <<<")
        area_deg2 = am.compute_survey_area(final_data['RAdeg'].values, final_data['DEdeg'].values)
        target_n = config.get('target_density', None)
        
        final_data, _ = am.apply_density_selection(
            final_data, zmin, zmax, area_deg2, 'lambda500', target_n
        )
        print(f"Muestra Final Density Mode: {len(final_data)} clusters")

    #%% BINNING DATA
    print(f'\nBinning data...')
    
    if binning_mode == 'density':
        final_data['bin_id'] = 0 # Todo en un solo bin
    else:
        # Modo clásico (quantiles)
        if binning_mode == 'redshift': metric_col = 'zCl'
        elif binning_mode == 'richness': metric_col = 'lambda500'
        final_data['bin_id'] = pd.qcut(final_data[metric_col], n_bins_quantile, labels=False)

    bins_info_list = []
    # Iteramos solo los bines existentes (si es density, será range(1))
    actual_bins = final_data['bin_id'].nunique()
    
    for i in sorted(final_data['bin_id'].unique()):
         subset = final_data[final_data['bin_id'] == i]
         if len(subset) == 0: continue
         
         info = {
              'id': int(i),
              'data': subset,
              'z_range': (subset['zCl'].min(), subset['zCl'].max()),
              'lambda_range': (subset['lambda500'].min(), subset['lambda500'].max()),
              'count': len(subset),
              'coords': SkyCoord(ra=subset['RAdeg'].values*u.degree, dec=subset['DEdeg'].values*u.degree, frame='icrs')
         }
         bins_info_list.append(info)
         print(f'Bin {i+1}: N={len(subset)}')
    print(' ')

    #%% LOOP PRINCIPAL
    print(f"\n######### Doing profiles in mode: {EXECUTION_MODE} #########\n")
    
    bin_results_list = []
    for info in bins_info_list:
        data_bin = info['data']
        eq_coords = info['coords']
        coords_bin = (eq_coords.galactic.l.degree, eq_coords.galactic.b.degree)
        z_bin_min, z_bin_max = info['z_range']
        lambda_bin_min, lambda_bin_max = info['lambda_range']
        
        # Ajuste en nombre de cache para no pisar modos
        cache_release_tag = f"{release}_Dens" if binning_mode == 'density' else release

        result = fm.process_bin_stacking(
            release=cache_release_tag,
            mode=EXECUTION_MODE,
            z_min=z_bin_min, 
            z_max=z_bin_max, 
            data_sample_bin=data_bin, 
            coords_bin=coords_bin,
            physical_size_Mpc=physical_size_Mpc, 
            npix_stamp=npix_stamp, 
            nside=nside, 
            smooth_value=smooth_value_deg, 
            reso=reso, 
            bins_physical=bins_physical,
            lensing_map=lensing_map, 
            common_mask=common_mask, 
            stacks_cache_folder=stacks_cache_folder, 
            lambda_min=lambda_bin_min, 
            lambda_max=lambda_bin_max, 
            N_RAND_FACTOR=N_RAND_FACTOR, 
            fm_module=fm,
            n_subsamples=N_subsamples
        )

        result['bin_id'] = info['id']
        result['binning_mode'] = binning_mode
        result['lambda_mean'] = data_bin['lambda500'].mean()
        result['lambda_range'] = (lambda_bin_min, lambda_bin_max)
        bin_results_list.append(result)

    #%% COMBINING RESULTS
    print("\n--- Generating Combined Results ---")
    weights = np.array([r['n_clusters'] for r in bin_results_list])
    total_clusters = np.sum(weights)

    stacked_maps = np.array([r['map'] for r in bin_results_list])
    final_map = np.average(stacked_maps, axis=0, weights=weights)
    stacked_profs = np.array([r['profile'] for r in bin_results_list])
    final_profile = np.average(stacked_profs, axis=0, weights=weights)
    stacked_errs = np.array([r['error'] for r in bin_results_list])
    final_error = np.sqrt(np.sum((weights[:, None] * stacked_errs)**2, axis=0)) / total_clusters

    combined_result = {
        'z_mean': final_data['zCl'].mean(),
        'lambda_mean': final_data['lambda500'].mean(),
        'map': final_map,
        'profile': final_profile,
        'error': final_error,
        'r_mpc': bin_results_list[0]['r_mpc'],
        'n_clusters': total_clusters
    }

    #%% SAVING
    print('\n--- Saving results... ---')
    
    out_name_bins = f'{run_folder}/BINS_Profiles.pdf'
    fm.plot_results(bin_results_list, f"Bin ({binning_mode})", out_name_bins, smooth_value_deg, physical_size_Mpc, 'kappa')

    out_name_combined = f'{run_folder}/COMBINED_Profiles.pdf'
    fm.plot_results([combined_result], "Combined", out_name_combined, smooth_value_deg, physical_size_Mpc, 'kappa')

    data_save_path = f'{run_folder}/Data_FullRun_{file_suffix}.pkl'
    save_dict = {'bins_data': bin_results_list, 'combined_data': combined_result, 'parameters': config}
    
    with open(data_save_path, 'wb') as f: pickle.dump(save_dict, f)
    print(f"Data saved in: {data_save_path}")

if __name__ == "__main__":
    print("Please run from PIPELINE_SCRIPT.py")