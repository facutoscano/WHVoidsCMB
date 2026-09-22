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


# Grupos multi-mapa (se corren en un loop y se plotean en un panel comun).
# El caso de exito de referencia es PLANCK_PR4 (con error y cosmic variance).
MAP_GROUPS = {
    'ALL_PLANCK': ['PLANCK_PR4', 'PLANCK_PR3', 'PLANCK_CIB',
                   'PLANCK_inhom', 'PLANCK_SZ', 'PLANCK_SZ_deproj'],
    'ALL':        ['PLANCK_PR4', 'PLANCK_PR3', 'PLANCK_CIB',
                   'PLANCK_inhom', 'PLANCK_SZ', 'PLANCK_SZ_deproj', 'ACT'],
}


def _map_spec(cmb_map, data_folder, act_smooth_arcmin=0.0):
    """Ruta del klm/mapa, mascara, nlkk, frame, nside nativo y modo de filtrado
    por mapa de lensing. cmb_map in
      {'PLANCK_PR4','PLANCK_PR3','PLANCK_CIB','PLANCK_SZ','PLANCK_SZ_deproj',
       'PLANCK_inhom','ACT'}.
    Planck: klm en galacticas, nside 2048, Wiener con su propio nlkk (CIB usa el de
    PR3, que no trae nlkk propio). Todo bajo CMB/PLANCK/Lensing/. ACT: mapa YA
    Wiener-filtrado, ecuatoriales, nside 512, sin nlkk, cielo cortado."""
    d = data_folder.rstrip('/')
    key = str(cmb_map).upper()
    L = f'{d}/CMB/PLANCK/Lensing'
    act = f'{d}/CMB/ACT'
    nlkk_pr3 = f'{L}/nlkk_PR3_MV.dat'

    specs = {
        'PLANCK_PR4': {'label': 'PLANCK_PR4', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/KAPPA_PR4klm_MV.fits',
                       'mask': f'{L}/Common_mask_PR4Lensing_2048.fits',
                       'nlkk': f'{L}/nlkk_PR4_MV.dat'},
        'PLANCK_PR3': {'label': 'PLANCK_PR3', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/KAPPA_PR3klm_MV.fits',
                       'mask': f'{L}/Common_mask_Lensing_2048.fits',
                       'nlkk': nlkk_pr3},
        'PLANCK_CIB': {'label': 'PLANCK_CIB', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/CIBdeproj/dat_klm_MV.fits',
                       'mask': f'{L}/CIBdeproj/mask.fits',
                       'nlkk': nlkk_pr3},                   # CIB: usa el nlkk de PR3
        'PLANCK_INHOM': {'label': 'PLANCK_inhom', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/Inhf/dat_klm_MV.fits',
                       'mask': f'{L}/Inhf/mask.fits',
                       'nlkk': f'{L}/Inhf/nlkk.dat'},
        'PLANCK_SZ': {'label': 'PLANCK_SZ', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/Sz/dat_klm_MV.fits',
                       'mask': f'{L}/Sz/mask.fits',
                       'nlkk': f'{L}/Sz/nlkk.dat'},
        'PLANCK_SZ_DEPROJ': {'label': 'PLANCK_SZ_deproj', 'kind': 'alm', 'frame': 'galactic',
                       'nside': 2048, 'full_sky': True, 'apply_wiener': True,
                       'klm':  f'{L}/Szdeproj/dat_klm_MV.fits',
                       'mask': f'{L}/Szdeproj/mask.fits',
                       'nlkk': f'{L}/Szdeproj/nlkk.dat'},
        'ACT': {'label': 'ACT', 'kind': 'map', 'frame': 'equatorial',
                       'nside': 512, 'full_sky': False, 'apply_wiener': False,
                       'smooth_arcmin': act_smooth_arcmin,   # gaussiana EXTRA sobre el mapa ya Wiener
                       'klm':  f'{act}/kappa_act_dr6_baseline_ns512_WF.fits',
                       'mask': f'{act}/mask_act_dr6_baseline_ns512.fits',
                       'nlkk': None},
    }
    if key == 'PLANCK':
        key = 'PLANCK_PR4'
    if key not in specs:
        raise KeyError(f"cmb_map desconocido: '{cmb_map}'. Opciones: "
                       f"{list(specs)} + {list(MAP_GROUPS)}")
    return specs[key]


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


#%% Run pipeline (un solo mapa)
def _run_single_map(config):
    data_folder = config['data_folder']
    output_folder = os.path.join(config['output_folder'], 'lensing')   # Results/lensing/
    os.makedirs(output_folder, exist_ok=True)
    zmin, zmax = config['zmin'], config['zmax']
    rmin, rmax = config['rmin'], config['rmax']

    max_Rvoid = config['max_Rvoid']
    Rvoid_bin = config['Rvoid_bin']
    npix_stamp = config['npix_stamp']

    # --- Spec del mapa (rutas/frame/nside) + cap de npix para no sobremuestrear ---
    # El void mas compacto angularmente (z=zmax, Rv=rmin) fija la resolucion minima:
    # npix tal que reso_arcmin >= tamano de pixel del mapa. Asi ningun void sobremuestrea.
    mspec = _map_spec(config['cmb_map'], data_folder, config.get('act_smooth_arcmin', 0.0))
    map_label = mspec['label']
    release = map_label      # etiqueta usada en los nombres de cache (antes: config['release'])
    # coverage_map: si se setea, la SELECCION de voids usa el footprint de OTRO mapa
    # (p.ej. PR4 apilado pero seleccionando en el footprint de ACT). El stacking sigue
    # sobre el mapa real. Solo se necesita su mascara.
    cov_map_key = config.get('coverage_map', None)
    cov_label = _map_spec(cov_map_key, data_folder)['label'] if cov_map_key else None
    pix_arcmin = hp.nside2resol(mspec['nside'], arcmin=True)
    box_deg_min = fm.get_angularsize_comoving(zmax, 2 * max_Rvoid * rmin)
    npix_cap = int(np.floor(box_deg_min * 60.0 / pix_arcmin))
    if npix_cap < npix_stamp:
        print(f'[npix] {map_label}: npix_stamp {npix_stamp} -> {npix_cap} '
              f'(pixel {pix_arcmin:.2f} arcmin @ nside={mspec["nside"]}; sin sobremuestreo).')
        npix_stamp = max(1, npix_cap)

    bins_frac = np.arange(0, max_Rvoid + Rvoid_bin, Rvoid_bin)
    reso_rv_per_pix = (2 * max_Rvoid) / npix_stamp
    smooth_value_deg = config.get('smooth_value_arcmin', 0.0) / 60.0

    binning_mode = config['binning_mode']
    n_bins_quantile = config['n_bins']
    exec_mode = config['exec_mode']
    n_subsamples = config['n_subsamples']
    random_factor = config.get('n_rand_factor', 10)
    n_rotations = config.get('n_rotations', 10)
    if not mspec['full_sky'] and n_rotations != 0:
        print(f'[nulls] {map_label}: mapa de cielo cortado -> fuerzo n_rotations=0 '
              f'(el null de rotacion saca los voids del footprint).')
        n_rotations = 0
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

    if mspec['kind'] == 'map':                 # ACT: mapa ya filtrado
        filter_label = 'prefiltered'
    else:
        filter_label = config.get('filter_mode', 'none')
        if filter_label == 'gaussian':
            filter_label = f'gaussian_{smooth_value_deg:.1f}deg'
        elif filter_label == 'wiener':
            filter_label = 'wiener'
        else:
            filter_label = 'no_filter'
    sm_extra = mspec.get('smooth_arcmin', 0.0)     # gaussiana EXTRA (p.ej. ACT)
    if sm_extra and sm_extra > 0:
        filter_label += f'_gsm{sm_extra:g}arcmin'

    base_suffix = (f'{mode_label}_{exec_mode}_'
                   f'{zmin}_{zmax}_{rmin}_{rmax}_'
                   f'maxRv{max_Rvoid:.1f}_{reso_rv_per_pix}Rvperpix_'
                   f'{delta_label}_{filter_label}'
                   + (f'_cov{cov_label}' if cov_label else ''))
    file_suffix = f'{cat_label}_{map_label}_{base_suffix}'

    run_folder = os.path.join(output_folder, file_suffix)
    legacy_folder = os.path.join(output_folder, base_suffix)   # corridas previas sin prefijo (solo WH)

    # Migración de corridas WH viejas (carpeta sin prefijo de catálogo): si existe y
    # no se fuerza el rerun, se renombra a WH_... y se saltea el análisis.
    if (not force_rerun) and cat_label == 'WH' and map_label == 'PLANCK_PR4' \
            and os.path.isdir(legacy_folder) and not os.path.isdir(run_folder):
        os.rename(legacy_folder, run_folder)
        print(f'[migrate] Carpeta legacy encontrada: renombrada\n'
              f'    {base_suffix}\n -> {file_suffix}\n'
              f'  y se saltea el análisis (force_rerun=False).')
        return

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Caché separada por catálogo para que WH y BOSS no colisionen (los nombres de
    # cache dependen de z/r/N, no del catálogo).
    stacks_cache_folder = os.path.join(output_folder, "Cache_Stacks", cat_label, map_label)
    if not os.path.exists(stacks_cache_folder):
        os.makedirs(stacks_cache_folder)

    print(f'######### CMB LENSING PROFILES USING {cat_label} VOIDS CATALOGUE [PARALLEL] #########')
    print(f'Configuration: Map={map_label} | Mode={exec_mode} | Binning={binning_mode} | n_workers={n_workers or os.cpu_count()}')
    print(f'Output Run Folder: {run_folder}')
    print('')

    #%% CMB map and masks
    print(f'Reading {map_label} CMB convergence map (frame={mspec["frame"]})...')
    filter_mode = config.get('filter_mode', 'none')

    if mspec['kind'] == 'alm':
        nside = mspec['nside']                          # Planck: 2048
        cmb_alm = hp.fitsfunc.read_alm(mspec['klm'], hdu=1, return_mmax=False)
        if mspec['apply_wiener'] and filter_mode == 'wiener':
            cmb_alm_filtered, W_ell = fm.apply_wiener_filter(cmb_alm, mspec['nlkk'], lmax=nside)
            lensing_map = hp.alm2map(cmb_alm_filtered, nside=nside)
            print('CMB map filtered with Wiener filter.')
        elif filter_mode == 'gaussian' and smooth_value_deg > 0:
            lensing_map = hp.smoothing(hp.alm2map(cmb_alm, nside=nside), fwhm=np.radians(smooth_value_deg))
            print(f'CMB map smoothed with Gaussian kernel of FWHM={smooth_value_deg:.1f} deg.')
        else:
            lensing_map = hp.alm2map(cmb_alm, nside=nside)
            print('CMB map without additional filtering applied.')
    else:                                               # ACT: mapa ya Wiener-filtrado
        lensing_map = hp.read_map(mspec['klm'])
        nside = hp.get_nside(lensing_map)               # 512 (maximo del mapa)
        print(f'Prefiltered map read (nside={nside}); no extra filtering applied.')

    # Suavizado gaussiano EXTRA por-mapa (sobre lo que ya tenga el mapa). Para ACT
    # es Wiener(ya aplicado) + gaussiana. Se hace full-sky: como usamos solo el
    # interior del footprint (corte de cobertura), el sangrado del borde no entra
    # al stack. La mascara NO se suaviza (sigue siendo footprint binario).
    if sm_extra and sm_extra > 0:
        lensing_map = hp.smoothing(lensing_map, fwhm=np.radians(sm_extra / 60.0))
        print(f'Extra Gaussian smoothing applied: FWHM={sm_extra:.1f} arcmin.')

    common_mask = hp.read_map(mspec['mask'])
    if hp.get_nside(common_mask) != nside:
        print(f'[mask] ud_grade {hp.get_nside(common_mask)} -> {nside}')
        common_mask = hp.ud_grade(common_mask, nside_out=nside)
    common_mask = np.where(common_mask >= 0.9, 1.0, 0.0)   # footprint binario (umbral 0.9)

    # Footprint usado para SELECCIONAR voids (por defecto el del propio mapa). Con
    # coverage_map se usa el de otro mapa (p.ej. seleccionar en ACT y apilar en PR4).
    if cov_map_key:
        cspec = _map_spec(cov_map_key, data_folder)
        cov_mask = hp.read_map(cspec['mask'])
        if hp.get_nside(cov_mask) != cspec['nside']:
            cov_mask = hp.ud_grade(cov_mask, nside_out=cspec['nside'])
        cov_mask = np.where(cov_mask >= 0.9, 1.0, 0.0)
        cov_nside, cov_frame = cspec['nside'], cspec['frame']
        print(f'[footprint] seleccion de voids con el footprint de {cspec["label"]} '
              f'(el stacking sigue sobre {map_label}).')
    else:
        cov_mask, cov_nside, cov_frame = common_mask, nside, mspec['frame']
    print('')

    #%% Reading and selecting voids data
    print(f'Reading {cat_label} voids catalogue...')
    n_seeds = config.get('N_seeds', None)
    sample_lb_gal = None      # coords galacticas de la muestra final (para el panel de footprint)

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
            fd = apply_delta_23_filter(filtered_data, delta_23_value)
            min_cov = config.get('min_footprint_coverage', 0.0)
            if min_cov and min_cov > 0 and len(fd):
                cov = fm.footprint_coverage(fd['l'].values, fd['b'].values, fd['z'].values,
                                            fd['R_void'].values, cov_mask, cov_nside,
                                            cov_frame, max_Rvoid)
                fd = fd[cov >= min_cov].copy()
            final_data[seed] = fd
        print('All voids data loaded.\n')
    else:
        col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z']
        voids_data_raw = pd.read_csv(cat['single_file'], sep='\s+', names=col_names, header=None)
        voids_data_raw = _ensure_galactic(voids_data_raw, cat)
        final_data = voids_data_raw[
            (voids_data_raw['z'] >= zmin) & (voids_data_raw['z'] < zmax) &
            (voids_data_raw['R_void'] >= rmin) & (voids_data_raw['R_void'] <= rmax)
        ].copy()
        min_cov = config.get('min_footprint_coverage', 0.0)
        if min_cov and min_cov > 0:
            n0 = len(final_data)
            cov = fm.footprint_coverage(final_data['l'].values, final_data['b'].values,
                                        final_data['z'].values, final_data['R_void'].values,
                                        cov_mask, cov_nside, cov_frame, max_Rvoid)
            final_data = final_data[cov >= min_cov].copy()
            print(f'[footprint] {map_label}: {len(final_data)}/{n0} voids con cobertura>={min_cov}.')
        sample_lb_gal = (final_data['l'].values, final_data['b'].values)
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
            min_cov = config.get('min_footprint_coverage', 0.0)
            if min_cov and min_cov > 0 and len(merged_df):
                mcov = fm.footprint_coverage(merged_df['l'].values, merged_df['b'].values,
                                             merged_df['z'].values, merged_df['R_void'].values,
                                             cov_mask, cov_nside, cov_frame, max_Rvoid)
                merged_df = merged_df[mcov >= min_cov].copy()
            edge_src = merged_df
            sample_lb_gal = (merged_df['l'].values, merged_df['b'].values)
        else:
            merged_df = None
            edge_src = concat_all
            if sample_lb_gal is None and len(final_data.get(0, [])):
                sample_lb_gal = (final_data[0]['l'].values, final_data[0]['b'].values)

        _, bin_edges = pd.qcut(edge_src[metric_col], n_bins_quantile,
                             retbins=True, labels=False, duplicates='drop')
        bin_edges = np.asarray(bin_edges, dtype=float).copy()
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf          # sin bins NaN en los bordes
        n_bins_eff = len(bin_edges) - 1

        for s in range(n_seeds):
            final_data[s]['bin_id'] = pd.cut(final_data[s][metric_col], bin_edges, labels=False)
        if merged_df is not None:
            merged_df['bin_id'] = pd.cut(merged_df[metric_col], bin_edges, labels=False)

        for i in range(n_bins_eff):
            seed_subsets, counts = {}, []
            for s in range(n_seeds):
                sub = final_data[s][final_data[s]['bin_id'] == i]
                counts.append(len(sub))
                seed_subsets[s] = {
                    'data': sub,
                    'z_range': (sub['z'].min(), sub['z'].max()) if len(sub) else (np.nan, np.nan),
                    'coords': fm.to_map_frame(sub['l'].values, sub['b'].values, mspec['frame']) if len(sub) else None,
                }
            entry = {'id': int(i), 'seed_subsets': seed_subsets,
                     'count_mean': float(np.mean(counts))}
            if merged_df is not None:
                msub = merged_df[merged_df['bin_id'] == i]
                entry['merged'] = {
                    'data': msub,
                    'z_range': (msub['z'].min(), msub['z'].max()) if len(msub) else (np.nan, np.nan),
                    'coords': fm.to_map_frame(msub['l'].values, msub['b'].values, mspec['frame']) if len(msub) else None,
                }
                print(f'Bin {i+1}: concat mean N={entry["count_mean"]:.1f} | merged N={len(msub)}')
            else:
                print(f'Bin {i+1}: concat mean N={entry["count_mean"]:.1f} across {n_seeds} seeds')
            bins_info_list.append(entry)

    print(' ')
    print(f'Binning completed.')
    print('')

    #%% LOOP PRINCIPAL
    print(f"\n######### Doing profiles #########\n")

    all_results = []

    if n_seeds is None:
        for info in bins_info_list:
            data_bin = info['data']
            gal_coords = info['coords']
            coords_bin = fm.to_map_frame(gal_coords.l.degree, gal_coords.b.degree, mspec['frame'])
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
                'catalog': 'single',
                'R_void_median': float(np.median(data_bin['R_void'].values))
                })
            all_results.append(result)
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
                    coords_combined = fm.to_map_frame(all_data_bin['l'].values, all_data_bin['b'].values, mspec['frame'])
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
                        'catalog': 'concat', 'seed_results': seed_results,
                        'R_void_median': float(np.median(all_data_bin['R_void'].values))
                        })
                    all_results.append(result_combined)

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
                    'catalog': 'merged',
                    'R_void_median': float(np.median(msub['data']['R_void'].values))
                    })
                all_results.append(result_merged)

    # SAVING
    # mediana de R_void por muestra: se imprime y se guarda dentro de 'parameters' del .pkl.
    rvoid_summary = fm.build_rvoid_summary(all_results)
    parameters = dict(config)
    parameters['R_void_median_per_sample'] = rvoid_summary

    print('Saving results...')

    output_plot_path = os.path.join(run_folder, f'Stacked_Maps_NullTests_{file_suffix}.pdf')
    fm.plot_stacked_maps_and_profiles(all_results, output_plot_path, max_Rvoid)

    jk_plot_path = os.path.join(run_folder, f'JK_Profile_Correlation_{file_suffix}.pdf')
    fm.plot_jackknife_and_correlation(all_results, jk_plot_path, max_Rvoid)

    if any(d.get('catalog') == 'concat' for d in all_results):
        seeds_plot_path = os.path.join(run_folder, f'Seed_Consistency_{file_suffix}.pdf')
        fm.plot_seed_consistency(all_results, seeds_plot_path, max_Rvoid)

    if do_concat and do_merge:
        concat = [r for r in all_results if r['catalog'] in ('single', 'concat')]
        merged = [r for r in all_results if r['catalog'] == 'merged']
        pairs = []
        for rc in concat:
            rm = next((m for m in merged if m['bin_id'] == rc.get('bin_id')), None)
            if rm is not None:
                pairs.append({'bin_id': rc['bin_id'], 'key': rc.get('key'),
                              'concat': rc, 'merge': rm})
        if pairs:
            cmp_path = os.path.join(run_folder, f'Merge_vs_Concat_{file_suffix}.pdf')
            fm.plot_merge_vs_concat(pairs, cmp_path, max_Rvoid)

    data_save_path = os.path.join(run_folder, f'Data_FullRun_{file_suffix}.pkl')
    with open(data_save_path, 'wb') as f:
        pickle.dump({'results': all_results,
                     'parameters': parameters},
                     f)
        print(f"Data saved in: {data_save_path}")

    return _primary_result(all_results, map_label, sample_lb_gal)


def _primary_result(all_results, map_label, sample_lb_gal=None):
    """Resultado representativo (bin 0) de una corrida, para el panel comparativo.
    Prioridad de catalogo: merged > concat > single. Adjunta las coords galacticas
    de la muestra (para dibujar el footprint + voids)."""
    if not all_results:
        return None
    out = None
    for cat in ('merged', 'concat', 'single'):
        for r in all_results:
            if r.get('catalog') == cat and int(r.get('bin_id', 0)) == 0:
                out = dict(r); break
        if out is not None:
            break
    if out is None:
        out = dict(all_results[0])
    out['map_label'] = map_label
    if sample_lb_gal is not None:
        out['void_l'], out['void_b'] = sample_lb_gal
    return out


#%% Dispatcher: un mapa, un grupo (All_Planck/All), o act-pr4 + panel comparativo
def _save_comparison_panel(collected, config, tag, success_label, order):
    if len(collected) < 2:
        print("[grupo] menos de 2 mapas con resultado; no genero panel comparativo.")
        return
    data_folder = config['data_folder']
    cat_label = _catalog_spec(config.get('void_catalog', 'WH'), data_folder)['label']
    out = os.path.join(config['output_folder'], 'lensing')
    fname = (f"MapComparison_{tag}_{cat_label}_{config['binning_mode']}_"
             f"{config['n_bins']}bins_{config['zmin']}_{config['zmax']}_"
             f"{config['rmin']}_{config['rmax']}.pdf")
    cmp_path = os.path.join(out, fname)
    fm.plot_map_comparison(collected, cmp_path, config['max_Rvoid'],
                           success_label=success_label, order=order)
    print(f"\n[grupo] panel comparativo guardado: {cmp_path}")


def _run_act_vs_pr4(config):
    """PR4 y ACT sobre EL MISMO conjunto de voids (los que entran en el footprint de
    ACT). PR4 se apila en su mapa pero selecciona voids con el footprint de ACT
    (coverage_map='ACT'); ACT usa su propio footprint (= el mismo). Asi la unica
    diferencia entre ambas curvas es el survey/mapa, no la muestra ni el parche."""
    print("\n######### act-pr4: PR4 vs ACT en el footprint de ACT #########")
    pr4_label = 'PLANCK_PR4 (ACT fp)'
    collected = {}

    cfg = dict(config); cfg['cmb_map'] = 'PLANCK_PR4'; cfg['coverage_map'] = 'ACT'
    print("\n================= PR4 (footprint de ACT) =================")
    try:
        res = _run_single_map(cfg)
        if res is not None:
            res['map_label'] = pr4_label
            collected[pr4_label] = res
    except FileNotFoundError as e:
        print(f"[act-pr4] PR4 salteado (archivo faltante): {e}")

    cfg = dict(config); cfg['cmb_map'] = 'ACT'; cfg['coverage_map'] = None
    print("\n================= ACT =================")
    try:
        res = _run_single_map(cfg)
        if res is not None:
            collected[res['map_label']] = res
    except FileNotFoundError as e:
        print(f"[act-pr4] ACT salteado (archivo faltante): {e}")

    if len(collected) < 2:
        print("[act-pr4] falta PR4 o ACT; no genero panel.")
        return

    # mascara de ACT (ecuatorial) para el panel de footprint (se muestra en galacticas)
    data_folder = config['data_folder']
    act_mask = hp.read_map(_map_spec('ACT', data_folder)['mask'])
    act_mask = np.where(act_mask >= 0.9, 1.0, 0.0)

    cat_label = _catalog_spec(config.get('void_catalog', 'WH'), data_folder)['label']
    out = os.path.join(config['output_folder'], 'lensing')
    fname = (f"ACTvsPR4_{cat_label}_{config['binning_mode']}_{config['n_bins']}bins_"
             f"{config['zmin']}_{config['zmax']}_{config['rmin']}_{config['rmax']}.pdf")
    cmp_path = os.path.join(out, fname)
    fm.plot_act_vs_pr4(collected, cmp_path, config['max_Rvoid'],
                       success_label=pr4_label, other_label='ACT',
                       footprint_mask=act_mask, footprint_coord='C')
    print(f"\n[act-pr4] panel guardado: {cmp_path}")


def run_pipeline(config):
    req = str(config.get('cmb_map', 'PLANCK_PR4'))
    key = req.upper().replace('-', '_')

    if key == 'ACT_PR4':
        _run_act_vs_pr4(config)
        return

    if key not in MAP_GROUPS:
        _run_single_map(config)                       # caso mapa unico (comportamiento previo)
        return

    keys = MAP_GROUPS[key]
    data_folder = config['data_folder']
    print(f"\n######### GRUPO {key}: {len(keys)} mapas -> {keys} #########")

    collected = {}
    for k in keys:
        cfg = dict(config); cfg['cmb_map'] = k
        print(f"\n================= MAPA {k}  ({key}) =================")
        try:
            res = _run_single_map(cfg)
        except FileNotFoundError as e:
            print(f"[grupo] '{k}' salteado (archivo faltante): {e}")
            continue
        if res is not None:
            collected[res['map_label']] = res

    tag = 'AllPlanck' if key == 'ALL_PLANCK' else 'All'
    order = [_map_spec(k, data_folder)['label'] for k in keys]
    _save_comparison_panel(collected, config, tag=tag,
                           success_label='PLANCK_PR4', order=order)


if __name__ == "__main__":
    print("Please run from Pipeline_voids.py")