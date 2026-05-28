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
    random_factor = config['n_rand_factor']
    
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
    final_data['bin_id'] = pd.qcut(final_data[metric_col], n_bins_quantile, labels=False)

    bins_info_list = []
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
    print(' ')
    print(f'Binning completed.')
    print('')

    #%% LOOP PRINCIPAL
    print(f"\n######### Doing profiles in mode: {exec_mode} #########\n")
    
    bin_results_list = []
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
            n_subsamples=n_subsamples
        )

        result['bin_id'] = info['id']
        result['binning_mode'] = binning_mode
        bin_results_list.append(result)

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