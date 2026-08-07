#%% IMPORTS
# Parallel twin of S1_voids.py: identical data loading / binning / plotting,
# but the per-bin stacking is delegated to Parallel_module.process_bin_stacking_parallel
# (random null, rotation null and jackknife/signal region stacks run across CPU cores).
import os
import numpy as np
import healpy as hp
import pandas as pd
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord
import Functions_module as fm
import Parallel_module as pm
import void_seed_merge as vsm


def _catalog_spec(catalog, data_folder):
    """Rutas, columnas de coordenadas y frame por catálogo. Ambos comparten el
    mismo layout de columnas; WH está en galáctico (l,b), BOSS en ecuatorial (ra,dec)."""
    label = (catalog or 'WH').upper()
    if label == 'BOSS':
        return {
            'label': 'BOSS',
            'seed_template': f'{data_folder}/CATALOGOS/BOSS_voids_merge/voids_boss_{{:03d}}.dat',
            'single_file': f'{data_folder}/CATALOGOS/BOSS_voids.dat',
            'lon_col': 'ra', 'lat_col': 'dec', 'frame': 'icrs',
        }
    return {
        'label': 'WH',
        'seed_template': f'{data_folder}/CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{{:03d}}.dat',
        'single_file': f'{data_folder}/CATALOGOS/WenHan_voids.dat',
        'lon_col': 'l', 'lat_col': 'b', 'frame': 'galactic',
    }


def _ensure_galactic(df, cat):
    """Garantiza columnas galácticas 'l','b'. WH ya las tiene; BOSS se convierte
    desde (ra,dec) ICRS para que todo el stacking/merge downstream use galácticas
    (el mapa de lensing de Planck está en galácticas)."""
    if cat['frame'] == 'galactic':
        return df
    c = SkyCoord(ra=df[cat['lon_col']].values * u.degree,
                 dec=df[cat['lat_col']].values * u.degree, frame='icrs').galactic
    df = df.copy()
    df['l'] = c.l.degree
    df['b'] = c.b.degree
    return df


#%% Run pipeline function
def run_pipeline(config):
    data_folder = config['data_folder']
    output_folder = os.path.join(config['output_folder'], 'lensing')   # Results/lensing/
    os.makedirs(output_folder, exist_ok=True)
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
    random_pool = config.get('random_pool', 'full')
    random_excl_factor = config.get('random_excl_factor', 1.0)
    n_workers = config.get('n_workers', None)

    # Multi-seed handling: 'concat' (legacy, duplicates), 'merge' (DBSCAN single
    # catalogue) or 'both' (run both and compare). Merge params reuse the tSZ ones.
    seed_mode = config.get('seed_mode', 'concat')
    do_concat = seed_mode in ('concat', 'both')
    do_merge = seed_mode in ('merge', 'both')
    merge_eps_mpch = config.get('merge_eps_mpch', 5.0)
    merge_min_frac = config.get('merge_min_frac', 0.2)
    merge_use_xyz = config.get('merge_use_catalog_xyz', False)

    # Void catalogue: 'WH' (galactic l,b) or 'BOSS' (equatorial ra,dec).
    void_catalog = config.get('void_catalog', 'WH')
    cat = _catalog_spec(void_catalog, data_folder)
    cat_label = cat['label']
    force_rerun = config.get('force_rerun', False)

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

    base_suffix = (f'{release}_{mode_label}_{exec_mode}_'
                   f'{zmin}_{zmax}_{rmin}_{rmax}_'
                   f'maxRv{max_Rvoid:.1f}_{reso_rv_per_pix}Rvperpix_'
                   f'{delta_label}_{filter_label}')
    file_suffix = f'{cat_label}_{base_suffix}'

    run_folder = os.path.join(output_folder, file_suffix)
    legacy_folder = os.path.join(output_folder, base_suffix)   # corridas previas sin prefijo (solo WH)

    # Migración de corridas WH viejas (carpeta sin prefijo de catálogo): si existe y
    # no se fuerza el rerun, se renombra a WH_... y se saltea el análisis.
    if (not force_rerun) and cat_label == 'WH' and os.path.isdir(legacy_folder) \
            and not os.path.isdir(run_folder):
        os.rename(legacy_folder, run_folder)
        print(f'[migrate] Carpeta legacy encontrada: renombrada\n'
              f'    {base_suffix}\n -> {file_suffix}\n'
              f'  y se saltea el análisis (force_rerun=False).')
        return

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Caché separada por catálogo para que WH y BOSS no colisionen (los nombres de
    # cache dependen de z/r/N, no del catálogo).
    stacks_cache_folder = os.path.join(output_folder, "Cache_Stacks", cat_label)
    if not os.path.exists(stacks_cache_folder):
        os.makedirs(stacks_cache_folder)

    print(f'######### CMB LENSING PROFILES USING {cat_label} VOIDS CATALOGUE [PARALLEL] #########')
    print(f'Configuration: Release={release} | Mode={exec_mode} | Binning={binning_mode} | n_workers={n_workers or os.cpu_count()}')
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
    print(f'Reading {cat_label} voids catalogue...')
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
        col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z', 'x', 'y', 'z_cart',
                     'delta_int', 'delta_23', 'completeness', 'delta_LOS']
        voids_data = {}
        final_data = {}
        for seed in range(n_seeds):
            file_path = cat['seed_template'].format(seed + 1)
            df_seed = pd.read_csv(file_path, sep='\s+', names=col_names, header=None)
            df_seed = _ensure_galactic(df_seed, cat)          # agrega l,b galácticas si es BOSS
            voids_data[seed] = df_seed

            base_filter = (df_seed['z'] >= zmin) & (df_seed['z'] < zmax) & (df_seed['R_void'] >= rmin) & (df_seed['R_void'] <= rmax) & (df_seed['completeness'] > 1.9)
            filtered_data = df_seed[base_filter].copy()
            final_data[seed] = apply_delta_23_filter(filtered_data, delta_23_value)
        print('All voids data loaded.\n')
    else:
        col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z']
        voids_data_raw = pd.read_csv(cat['single_file'], sep='\s+', names=col_names, header=None)
        voids_data_raw = _ensure_galactic(voids_data_raw, cat)
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
        # Un único juego de bordes de cuantiles, compartido por todas las seeds y
        # por el catálogo mergeado, para que el bin i cubra el mismo rango de z (o R).
        concat_all = pd.concat([final_data[s].assign(seed=s) for s in range(n_seeds)],
                               ignore_index=True)

        if do_merge:
            print(f'Merging {n_seeds} seeds into a single catalogue (DBSCAN dedup)...')
            merged_df, _ = vsm.merge_seeds(concat_all, eps_mpch=merge_eps_mpch,
                                           min_frac=merge_min_frac, n_seeds=n_seeds,
                                           use_catalog_xyz=merge_use_xyz)
            edge_src = merged_df
        else:
            merged_df = None
            edge_src = concat_all

        _, z_edges = pd.qcut(edge_src[metric_col], n_bins_quantile,
                             retbins=True, labels=False, duplicates='drop')
        z_edges = np.asarray(z_edges, dtype=float).copy()
        z_edges[0], z_edges[-1] = -np.inf, np.inf          # sin bins NaN en los bordes
        n_bins_eff = len(z_edges) - 1

        for s in range(n_seeds):
            final_data[s]['bin_id'] = pd.cut(final_data[s][metric_col], z_edges, labels=False)
        if merged_df is not None:
            merged_df['bin_id'] = pd.cut(merged_df[metric_col], z_edges, labels=False)

        for i in range(n_bins_eff):
            seed_subsets, counts = {}, []
            for s in range(n_seeds):
                sub = final_data[s][final_data[s]['bin_id'] == i]
                counts.append(len(sub))
                seed_subsets[s] = {
                    'data': sub,
                    'z_range': (sub['z'].min(), sub['z'].max()) if len(sub) else (np.nan, np.nan),
                    'coords': (sub['l'].values, sub['b'].values) if len(sub) else None,
                }
            entry = {'id': int(i), 'seed_subsets': seed_subsets,
                     'count_mean': float(np.mean(counts))}
            if merged_df is not None:
                msub = merged_df[merged_df['bin_id'] == i]
                entry['merged'] = {
                    'data': msub,
                    'z_range': (msub['z'].min(), msub['z'].max()) if len(msub) else (np.nan, np.nan),
                    'coords': (msub['l'].values, msub['b'].values) if len(msub) else None,
                }
                print(f'Bin {i+1}: concat mean N={entry["count_mean"]:.1f} | merged N={len(msub)}')
            else:
                print(f'Bin {i+1}: concat mean N={entry["count_mean"]:.1f} across {n_seeds} seeds')
            bins_info_list.append(entry)

    print(' ')
    print(f'Binning completed.')
    print('')

    #%% LOOP PRINCIPAL
    print(f"\n######### Doing profiles in mode: {exec_mode} [PARALLEL] #########\n")

    bin_results_list = []
    merged_results_list = []

    if n_seeds is None:
        for info in bins_info_list:
            data_bin = info['data']
            gal_coords = info['coords']
            coords_bin = (gal_coords.l.degree, gal_coords.b.degree)
            z_bin_min, z_bin_max = info['z_range']

            result = pm.process_bin_stacking_parallel(
                release=release,
                mode=exec_mode,
                z_min=z_bin_min,
                z_max=z_bin_max,
                r_min=rmin,
                r_max=rmax,
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
                force_rerun=config.get('force_rerun', False),
                random_pool=random_pool,
                random_excl_factor=random_excl_factor,
                n_workers=n_workers
            )

            result.update({
                'bin_id': info['id'],
                'binning_mode': binning_mode,
                'is_multi_seed': False,
                'R_void_median': float(np.median(data_bin['R_void'].values))
                })
            bin_results_list.append(result)
    else:
        for entry in bins_info_list:
            bin_id = entry['id']
            print(f"--- Processing Bin {bin_id+1} ---")

            # ---- concatenación multi-seed (+ stacks por seed) --------------
            if do_concat:
                seed_results = []
                for seed in range(n_seeds):
                    si = entry['seed_subsets'][seed]
                    if si['coords'] is None or len(si['data']) == 0:
                        continue
                    seed_cache_folder = os.path.join(stacks_cache_folder, f"seed_{seed}")
                    os.makedirs(seed_cache_folder, exist_ok=True)
                    z_bin_min, z_bin_max = si['z_range']
                    result = pm.process_bin_stacking_parallel(
                        release=release, mode=exec_mode,
                        z_min=z_bin_min, z_max=z_bin_max, r_min=rmin, r_max=rmax,
                        data_sample_bin=si['data'], coords_bin=si['coords'], max_Rvoid=max_Rvoid,
                        npix_stamp=npix_stamp, nside=nside, bins_frac=bins_frac,
                        lensing_map=lensing_map, common_mask=common_mask,
                        stacks_cache_folder=seed_cache_folder, n_random_factor=random_factor,
                        n_rotations=n_rotations, n_subsamples=n_subsamples, delta_label=delta_label,
                        filter_label=filter_label, force_rerun=config.get('force_rerun', False),
                        random_pool=random_pool, random_excl_factor=random_excl_factor, n_workers=n_workers
                    )
                    result['seed'] = seed
                    result['R_void_median'] = float(np.median(si['data']['R_void'].values))
                    seed_results.append(result)

                nonempty = [s for s in range(n_seeds) if len(entry['seed_subsets'][s]['data']) > 0]
                if nonempty:
                    all_data_bin = pd.concat([entry['seed_subsets'][s]['data'] for s in nonempty],
                                             ignore_index=True)
                    coords_combined = (all_data_bin['l'].values, all_data_bin['b'].values)
                    z_min_combined = np.nanmin([entry['seed_subsets'][s]['z_range'][0] for s in nonempty])
                    z_max_combined = np.nanmax([entry['seed_subsets'][s]['z_range'][1] for s in nonempty])

                    print(f'Concatenated stack: {len(all_data_bin)} voids across {n_seeds} seeds')
                    combined_cache_folder = os.path.join(stacks_cache_folder, 'combined')
                    os.makedirs(combined_cache_folder, exist_ok=True)

                    result_combined = pm.process_bin_stacking_parallel(release=release, mode=exec_mode,
                                                              z_min=z_min_combined, z_max=z_max_combined, r_min=rmin, r_max=rmax,
                                                              data_sample_bin=all_data_bin,
                                                              coords_bin=coords_combined, max_Rvoid=max_Rvoid,
                                                              npix_stamp=npix_stamp, nside=nside, bins_frac=bins_frac,
                                                              lensing_map=lensing_map, common_mask=common_mask,
                                                              stacks_cache_folder=combined_cache_folder,
                                                              n_random_factor=random_factor, n_rotations=n_rotations, n_subsamples=n_subsamples,
                                                              delta_label=delta_label, filter_label=filter_label,
                                                              force_rerun=config.get('force_rerun', False),
                                                              random_pool=random_pool, random_excl_factor=random_excl_factor, n_workers=n_workers)
                    result_combined.update({
                        'bin_id': bin_id, 'binning_mode': binning_mode,
                        'is_multi_seed': True, 'seed_results': seed_results,
                        'R_void_median': float(np.median(all_data_bin['R_void'].values))
                        })
                    bin_results_list.append(result_combined)

            # ---- catálogo único mergeado (DBSCAN) --------------------------
            if do_merge and entry.get('merged') and entry['merged']['coords'] is not None \
                    and len(entry['merged']['data']) > 0:
                msub = entry['merged']
                merged_cache_folder = os.path.join(stacks_cache_folder, 'merged')
                os.makedirs(merged_cache_folder, exist_ok=True)
                z_bin_min, z_bin_max = msub['z_range']
                print(f'Merged stack: {len(msub["data"])} unique voids')

                result_merged = pm.process_bin_stacking_parallel(release=release, mode=exec_mode,
                                                          z_min=z_bin_min, z_max=z_bin_max, r_min=rmin, r_max=rmax,
                                                          data_sample_bin=msub['data'],
                                                          coords_bin=msub['coords'], max_Rvoid=max_Rvoid,
                                                          npix_stamp=npix_stamp, nside=nside, bins_frac=bins_frac,
                                                          lensing_map=lensing_map, common_mask=common_mask,
                                                          stacks_cache_folder=merged_cache_folder,
                                                          n_random_factor=random_factor, n_rotations=n_rotations, n_subsamples=n_subsamples,
                                                          delta_label=delta_label, filter_label=filter_label,
                                                          force_rerun=config.get('force_rerun', False),
                                                          random_pool=random_pool, random_excl_factor=random_excl_factor, n_workers=n_workers)
                result_merged.update({
                    'bin_id': bin_id, 'binning_mode': binning_mode,
                    'is_multi_seed': False, 'is_merged': True,
                    'R_void_median': float(np.median(msub['data']['R_void'].values))
                    })
                (merged_results_list if do_concat else bin_results_list).append(result_merged)

    # SAVING
    # mediana de R_void por muestra: se imprime y se guarda dentro de 'parameters' del .pkl.
    rvoid_summary = fm.build_rvoid_summary(bin_results_list, merged_results_list)
    parameters = dict(config)
    parameters['R_void_median_per_sample'] = rvoid_summary

    print('Saving results...')

    output_plot_path = os.path.join(run_folder, f'Stacked_Maps_NullTests_{file_suffix}.pdf')
    fm.plot_stacked_maps_and_profiles(bin_results_list, output_plot_path, max_Rvoid)

    jk_plot_path = os.path.join(run_folder, f'JK_Profile_Correlation_{file_suffix}.pdf')
    fm.plot_jackknife_and_correlation(bin_results_list, jk_plot_path, max_Rvoid)

    if any(d.get('is_multi_seed', False) for d in bin_results_list):
        seeds_plot_path = os.path.join(run_folder, f'Seed_Consistency_{file_suffix}.pdf')
        fm.plot_seed_consistency(bin_results_list, seeds_plot_path, max_Rvoid)

    if do_concat and do_merge and merged_results_list:
        pairs = []
        for rc in bin_results_list:
            rm = next((m for m in merged_results_list if m['bin_id'] == rc.get('bin_id')), None)
            if rm is not None:
                pairs.append({'bin_id': rc['bin_id'], 'key': rc.get('key'),
                              'concat': rc, 'merge': rm})
        if pairs:
            cmp_path = os.path.join(run_folder, f'Merge_vs_Concat_{file_suffix}.pdf')
            fm.plot_merge_vs_concat(pairs, cmp_path, max_Rvoid)

    data_save_path = os.path.join(run_folder, f'Data_FullRun_{file_suffix}.pkl')
    with open(data_save_path, 'wb') as f:
        pickle.dump({'bins_data': bin_results_list,
                     'merged_data': merged_results_list, 'parameters': parameters}, f)
        print(f"Data saved in: {data_save_path}")

if __name__ == "__main__":
    print("Please run from Pipeline_voids.py")
