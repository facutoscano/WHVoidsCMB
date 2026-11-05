#%% Imports
import os
import json
import hashlib
import numpy as np
import healpy as hp
import pandas as pd
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord
import Functions_module as fm
import Parallel_module as pm
import void_seed_merge as vsm


#%% Catalogues
CATALOG_GROUPS = {'BOTH': ['WH', 'BOSS']}


def _catalog_spec(catalog, data_folder):
    """
    Void catalogue: seed files, single file, coordinate columns and the tracer
    footprint (galactic) used to place the randoms.
      WH   : voids from Wen & Han clusters, galactic (l,b); footprint = occupancy of the
             WH clusters.
      BOSS : voids from BOSS galaxies, equatorial (ra,dec); footprint = occupancy of the
             void centres of the 100 seeds (approximate: no galaxy mask available).
    """
    d = data_folder.rstrip('/')
    label = (catalog or 'WH').upper()
    if label == 'BOSS':
        return {
            'label': 'BOSS',
            'seed_template': f'{d}/CATALOGOS/BOSS_voids_merge/voids_boss_{{:03d}}.dat',
            'single_file': f'{d}/CATALOGOS/BOSS_voids.dat',
            'lon_col': 'ra', 'lat_col': 'dec', 'frame': 'icrs',
            'footprint': {'kind': 'points', 'path': 'CATALOGOS/BOSS_voids_merge/voids_boss_{:03d}.dat',
                          'cols': (1, 2), 'frame': 'icrs', 'skiprows': 0, 'n_files': 100, 'nside': 16},
        }
    if label != 'WH':
        raise KeyError(f"unknown void_catalog '{catalog}'. Options: WH, BOSS, BOTH")
    return {
        'label': 'WH',
        'seed_template': f'{d}/CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{{:03d}}.dat',
        'single_file': f'{d}/CATALOGOS/WenHan_voids.dat',
        'lon_col': 'l', 'lat_col': 'b', 'frame': 'galactic',
        'footprint': {'kind': 'points', 'path': 'CATALOGOS/WenHan_Clusters.dat',
                      'cols': (2, 3), 'frame': 'icrs', 'skiprows': 56, 'n_files': None, 'nside': 64},
    }


def _ensure_galactic(df, cat):
    """WH is already galactic; BOSS (ra,dec) is converted to (l,b)."""
    if cat['frame'] == 'galactic':
        return df
    c = SkyCoord(ra=df[cat['lon_col']].values * u.degree,
                 dec=df[cat['lat_col']].values * u.degree, frame='icrs').galactic
    df = df.copy()
    df['l'] = c.l.degree
    df['b'] = c.b.degree
    return df


#%% Maps
MAP_GROUPS = {
    'ALL_PLANCK': ['PLANCK_PR4', 'PLANCK_PR3', 'PLANCK_CIBCOMB',
                   'PLANCK_INHOM', 'PLANCK_SZ', 'PLANCK_SZ_DEPROJ'],
    'ALL':        ['PLANCK_PR4', 'PLANCK_PR3', 'PLANCK_CIBCOMB',
                   'PLANCK_INHOM', 'PLANCK_SZ', 'PLANCK_SZ_DEPROJ', 'ACT'],
}
GROUP_TAGS = {'ALL_PLANCK': 'AllPlanck', 'ALL': 'All', 'ACT_PR4': 'ACTxPR4'}


def _map_key(cmb_map):
    return str(cmb_map).upper().replace('-', '_')


def _map_spec(cmb_map, data_folder, act_smooth_arcmin=0.0):
    """
    Convergence map: file, mask, frame, nside and filtering.
      prefiltered=True : the product is already filtered (ACT WF map, Planck lensing+CIB
                         combined tracer) -> no Wiener filter is applied on top.
      nlkk             : (L, N_L, C_L+N_L) used for the Wiener filter otherwise.
    Planck maps are galactic (nside 2048); ACT is equatorial (nside 512).
    """
    d = data_folder.rstrip('/')
    key = _map_key(cmb_map)
    L = f'{d}/CMB/PLANCK/Lensing'
    act = f'{d}/CMB/ACT'
    nlkk_pr3 = f'{L}/nlkk_PR3_MV.dat'

    specs = {
        'PLANCK_PR4': {'label': 'PLANCK_PR4', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                       'full_sky': True, 'prefiltered': False,
                       'klm': f'{L}/KAPPA_PR4klm_MV.fits',
                       'mask': f'{L}/Common_mask_PR4Lensing_2048.fits',
                       'nlkk': f'{L}/nlkk_PR4_MV.dat'},
        'PLANCK_PR3': {'label': 'PLANCK_PR3', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                       'full_sky': True, 'prefiltered': False,
                       'klm': f'{L}/KAPPA_PR3klm_MV.fits',
                       'mask': f'{L}/Common_mask_Lensing_2048.fits',
                       'nlkk': nlkk_pr3},
        # COM_Lensing-CIBcomb_2000_R3.00: lensing tracer built from the MV quadratic
        # estimator AND the GNILC 545 GHz CIB map. Already filtered (L in [8, 2000]).
        'PLANCK_CIBCOMB': {'label': 'PLANCK_CIBcomb', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                           'full_sky': True, 'prefiltered': True,
                           'klm': f'{L}/CIBdeproj/dat_klm_MV.fits',
                           'mask': f'{L}/CIBdeproj/mask.fits',
                           'nlkk': None},
        'PLANCK_INHOM': {'label': 'PLANCK_inhom', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                         'full_sky': True, 'prefiltered': False,
                         'klm': f'{L}/Inhf/dat_klm_MV.fits',
                         'mask': f'{L}/Inhf/mask.fits',
                         'nlkk': f'{L}/Inhf/nlkk.dat'},
        'PLANCK_SZ': {'label': 'PLANCK_SZ', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                      'full_sky': True, 'prefiltered': False,
                      'klm': f'{L}/Sz/dat_klm_MV.fits',
                      'mask': f'{L}/Sz/mask.fits',
                      'nlkk': f'{L}/Sz/nlkk.dat'},
        'PLANCK_SZ_DEPROJ': {'label': 'PLANCK_SZ_deproj', 'kind': 'alm', 'frame': 'galactic', 'nside': 2048,
                             'full_sky': True, 'prefiltered': False,
                             'klm': f'{L}/Szdeproj/dat_klm_MV.fits',
                             'mask': f'{L}/Szdeproj/mask.fits',
                             'nlkk': f'{L}/Szdeproj/nlkk.dat'},
        'ACT': {'label': 'ACT', 'kind': 'map', 'frame': 'equatorial', 'nside': 512,
                'full_sky': False, 'prefiltered': True, 'smooth_arcmin': act_smooth_arcmin,
                'klm': f'{act}/kappa_act_dr6_baseline_ns512_WF.fits',
                'mask': f'{act}/mask_act_dr6_baseline_ns512.fits',
                'nlkk': None},
    }
    if key == 'PLANCK':
        key = 'PLANCK_PR4'
    if key not in specs:
        raise KeyError(f"unknown cmb_map: '{cmb_map}'. Options: {list(specs)} + {list(MAP_GROUPS)} + ['ACT-PR4']")
    return specs[key]


def _effective_filter(mspec, config):
    """(label, action) of the filter that is REALLY applied to this map."""
    sm_extra = mspec.get('smooth_arcmin', 0.0) or 0.0
    extra = f'_{sm_extra:g}arcmin' if sm_extra > 0 else ''
    if mspec.get('prefiltered'):
        return 'prefiltered' + extra, 'none'
    mode = config.get('filter_mode', 'none')
    if mode == 'wiener' and mspec.get('nlkk'):
        return 'wiener' + extra, 'wiener'
    sm = config.get('smooth_value_arcmin', 0.0) or 0.0
    if mode == 'gaussian' and sm > 0:
        return f'gauss{sm:g}arcmin' + extra, 'gaussian'
    return 'nofilter' + extra, 'none'


def _load_map(mspec, config):
    """Convergence map in its native frame/nside with the effective filter applied."""
    flabel, action = _effective_filter(mspec, config)
    if mspec['kind'] == 'alm':
        nside = mspec['nside']
        alm = hp.fitsfunc.read_alm(mspec['klm'], hdu=1, return_mmax=False)
        if action == 'wiener':
            alm, _ = fm.apply_wiener_filter(alm, mspec['nlkk'], lmax=nside)
            m = hp.alm2map(alm, nside=nside)
        elif action == 'gaussian':
            fwhm = np.radians(config['smooth_value_arcmin'] / 60.0)
            m = hp.smoothing(hp.alm2map(alm, nside=nside), fwhm=fwhm)
        else:
            m = hp.alm2map(alm, nside=nside)
    else:
        m = hp.read_map(mspec['klm'])
        nside = hp.get_nside(m)
    sm_extra = mspec.get('smooth_arcmin', 0.0) or 0.0
    if sm_extra > 0:
        m = hp.smoothing(m, fwhm=np.radians(sm_extra / 60.0))
    print(f"[Data] {mspec['label']}: nside={nside}, frame={mspec['frame']}, filter={flabel}")
    return m, nside, flabel


#%% Coverage region and survey footprint
def _short(label):
    return str(label).replace('PLANCK_', '')


def _coverage_keys(config, own_key):
    cov = config.get('coverage_map')
    if cov is None:
        keys = []
    elif isinstance(cov, (list, tuple)):
        keys = list(cov)
    else:
        keys = [cov]
    keys = [k for k in keys if _map_key(k) != _map_key(own_key)]
    return [own_key] + keys


def _coverage_region(keys, data_folder, nside_cov, act_smooth=0.0):
    """
    Galactic map at nside_cov with the fraction of each pixel inside ALL the masks
    of `keys` (intersection). Voids AND randoms are selected on this region.
    """
    frac = np.ones(hp.nside2npix(nside_cov))
    labels, used = [], set()
    for k in keys:
        sp = _map_spec(k, data_folder, act_smooth)
        mpath = os.path.abspath(str(sp['mask']))
        if mpath in used:          # the same mask twice would square its boundary pixels
            continue
        used.add(mpath)
        frac *= fm.mask_fraction_map(sp['mask'], nside_cov, frame_in=fm.frame_code(sp['frame']),
                                     frame_out='G')
        labels.append(sp['label'])
    return frac, labels


def _hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _file_sig(path):
    """(path, size, mtime): a regenerated file (e.g. a new nlkk) invalidates the cache."""
    if not path or not os.path.exists(str(path)):
        return path
    st = os.stat(path)
    return [str(path), st.st_size, int(st.st_mtime)]


def _cosmo_sig():
    c = fm.Planck18                          # Websky.py swaps it for the Websky cosmology
    return [str(getattr(c, 'name', '')), round(float(c.h), 6), round(float(c.Om0), 6)]


def _sample_hash(df):
    a = np.round(df[['l', 'b', 'z', 'R_void']].values.astype(float), 6)
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


#%% Selection helpers
def profile_method_of(config):
    """'stamp' (profile from stacked gnomonic stamps) | 'healpix' (from the map pixels)."""
    m = str(config.get('profile_method', 'stamp')).lower()
    if m not in ('stamp', 'healpix'):
        raise ValueError(f"profile_method must be 'stamp' or 'healpix', not {m!r}")
    return m


def _method_tag(config):
    """Folder/file tag: none for 'stamp' (names as before), '_hpx' for 'healpix'."""
    return '_hpx' if profile_method_of(config) == 'healpix' else ''


def _seed_tag(config):
    """single | concat{N} | merge{N}e{eps}f{min_frac} | both{N}e{eps}f{min_frac}"""
    n = config.get('N_seeds')
    if n is None:
        return 'single'
    mode = config.get('seed_mode', 'concat')
    if mode == 'concat':
        return f'concat{n}'
    return f"{mode}{n}e{config.get('merge_eps_mpch', 8.0):g}f{config.get('merge_min_frac', 0.4):g}"


def _delta_label(v):
    if v is None:
        return 'd23all'
    return f'd23gt{v:g}' if v >= 0 else f'd23lt{v:g}'


def _apply_delta(df, v):
    """v=None: no cut | v>=0: delta_23 > v | v<0: delta_23 < v."""
    if v is None:
        return df
    return df[df['delta_23'] > v] if v >= 0 else df[df['delta_23'] < v]


def _coverage_cut(df, cov_frac, rad_frac, min_cov):
    if not min_cov or min_cov <= 0 or len(df) == 0:
        return df
    cov = fm.disc_coverage(df['l'].values, df['b'].values, df['z'].values, df['R_void'].values,
                           cov_frac, rad_frac)
    return df[cov >= min_cov].copy()


def load_void_samples(config, cat, cov_frac):
    """
    Read and select the voids (same cuts as always: z, R_v, completeness, delta_23 and
    coverage of the max_Rvoid*R_v disc on the coverage region).
    Returns dict with 'per_seed' {seed: df} (multi-seed) or 'single' df, plus 'merged'.
    """
    zmin, zmax = config['zmin'], config['zmax']
    rmin, rmax = config['rmin'], config['rmax']
    max_Rvoid = config['max_Rvoid']
    min_cov = config.get('min_footprint_coverage', 0.0)
    delta = config.get('delta_value', None)
    n_seeds = config.get('N_seeds', None)
    out = {'per_seed': None, 'single': None, 'merged': None}

    if n_seeds is None:
        if delta is not None:
            raise ValueError('delta_value needs the multi-seed catalogues (the single file has no delta_23)')
        raw = pd.read_csv(cat['single_file'], sep=r'\s+', header=None, comment='#')
        if raw.shape[1] > 4:
            print(f"[Data] (!) {os.path.basename(cat['single_file'])} has {raw.shape[1]} columns; "
                  f"using the first 4 as (R_void, lon, lat, z)")
        raw = raw.iloc[:, :4]
        raw.columns = ['R_void', cat['lon_col'], cat['lat_col'], 'z']
        raw = _ensure_galactic(raw, cat)
        df = raw[(raw['z'] >= zmin) & (raw['z'] < zmax) &
                 (raw['R_void'] >= rmin) & (raw['R_void'] <= rmax)].copy()
        n0 = len(df)
        df = _coverage_cut(df, cov_frac, max_Rvoid, min_cov)
        print(f'[Data] single catalogue: {len(df)}/{n0} voids with coverage >= {min_cov}')
        out['single'] = df
        return out

    print(f'[Data] Reading {n_seeds} seed catalogues...')
    col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z', 'x', 'y', 'z_cart',
                 'delta_int', 'delta_23', 'completeness', 'delta_LOS']
    per_seed = {}
    for seed in range(n_seeds):
        df = pd.read_csv(cat['seed_template'].format(seed + 1), sep=r'\s+', header=None, comment='#')
        df = df.iloc[:, :len(col_names)]
        df.columns = col_names[:df.shape[1]]
        df = _ensure_galactic(df, cat)
        sel = ((df['z'] >= zmin) & (df['z'] < zmax) & (df['R_void'] >= rmin) &
               (df['R_void'] <= rmax) & (df['completeness'] > 1.9))
        df = _apply_delta(df[sel].copy(), delta)
        per_seed[seed] = _coverage_cut(df, cov_frac, max_Rvoid, min_cov)
    out['per_seed'] = per_seed
    print(f"[Data] per seed: mean N={np.mean([len(v) for v in per_seed.values()]):.1f}")

    if config.get('seed_mode', 'concat') in ('merge', 'both'):
        concat_all = pd.concat([per_seed[s].assign(seed=s) for s in range(n_seeds)], ignore_index=True)
        merged, _ = vsm.merge_seeds(concat_all, eps_mpch=config.get('merge_eps_mpch', 8.0),
                                    min_frac=config.get('merge_min_frac', 0.4), n_seeds=n_seeds,
                                    use_catalog_xyz=config.get('merge_use_catalog_xyz', False))
        n0 = len(merged)
        merged = _coverage_cut(merged, cov_frac, max_Rvoid, min_cov)
        print(f'[Data] merged catalogue: {len(merged)}/{n0} voids with coverage >= {min_cov}')
        out['merged'] = merged
    return out


#%% Run one map
def _run_single_map(config):
    data_folder = config['data_folder']
    output_folder = os.path.join(config['output_folder'], 'lensing')
    os.makedirs(output_folder, exist_ok=True)
    zmin, zmax, rmin, rmax = config['zmin'], config['zmax'], config['rmin'], config['rmax']
    max_Rvoid, Rvoid_bin = config['max_Rvoid'], config['Rvoid_bin']
    min_cov = config.get('min_footprint_coverage', 0.0)
    nside_cov = config.get('coverage_nside', 256)
    act_smooth = config.get('act_smooth_arcmin', 0.0)

    if config.get('N_seeds') is None and config.get('delta_value') is not None:
        raise ValueError('delta_value needs the multi-seed catalogues (the single file has no delta_23)')
    mspec = _map_spec(config['cmb_map'], data_folder, act_smooth)
    map_label = mspec['label']
    filter_label, _ = _effective_filter(mspec, config)

    cat = _catalog_spec(config.get('void_catalog', 'WH'), data_folder)
    cat_label = cat['label']
    force_rerun = config.get('force_rerun', False)

    # stamp resolution: never oversample the map
    npix_stamp = config['npix_stamp']
    pix_arcmin = hp.nside2resol(mspec['nside'], arcmin=True)
    box_deg_min = fm.get_angularsize_comoving(zmax, 2 * max_Rvoid * rmin)
    npix_cap = int(np.floor(box_deg_min * 60.0 / pix_arcmin))
    if npix_cap < npix_stamp:
        print(f'[npix] {map_label}: npix_stamp {npix_stamp} -> {npix_cap} '
              f'(pixel {pix_arcmin:.2f} arcmin @ nside={mspec["nside"]}; not oversampling)')
        npix_stamp = max(1, npix_cap)
    bins_frac = np.arange(0, max_Rvoid + Rvoid_bin, Rvoid_bin)
    profile_method = profile_method_of(config)
    if profile_method == 'stamp':
        if 2 * max_Rvoid / npix_stamp > Rvoid_bin:
            print(f'[npix] (!) {map_label}: stamp pixel = {2 * max_Rvoid / npix_stamp:.3f} Rv > Rvoid_bin = '
                  f'{Rvoid_bin} Rv for the smallest voids: the innermost bins can be empty (NaN)')
    else:
        theta_in = np.radians(fm.get_angularsize_comoving(zmax, Rvoid_bin * rmin))     # innermost ring
        n_in = np.pi * theta_in ** 2 / hp.nside2pixarea(mspec['nside'])
        if n_in < 5:
            print(f'[pix] (!) {map_label}: the innermost ring (r < {Rvoid_bin} Rv) of the smallest voids '
                  f'(R={rmin:g} at z={zmax:g}) holds only ~{n_in:.1f} map pixels (nside={mspec["nside"]})')
    if 'outer' in config.get('bkg_mode', 'randoms'):
        o = tuple(config.get('bkg_outer', (2.0, 2.5)))
        rc = 0.5 * (bins_frac[:-1] + bins_frac[1:])
        if not np.any((rc >= o[0]) & (rc <= o[1])):
            raise ValueError(f"bkg_outer={o} has no radial bin: profiles reach r/Rv={bins_frac[-1]:.2f} "
                             f"(max_Rvoid={max_Rvoid}); increase max_Rvoid or change bkg_outer")

    binning_mode, n_bins_q = config['binning_mode'], config['n_bins']
    exec_mode, n_subsamples = config['exec_mode'], config['n_subsamples']
    n_rand = config.get('n_rand_factor', 100)
    n_rotations = config.get('n_rotations', 10)
    if not mspec['full_sky'] and n_rotations:
        print(f'[Null Test] {map_label}: partial sky -> n_rotations=0')
        n_rotations = 0
    random_pool = config.get('random_pool', 'footprint')
    random_excl = config.get('random_excl_factor', None)
    bkg_mode = config.get('bkg_mode', 'randoms')
    bkg_outer = tuple(config.get('bkg_outer', (2.0, 2.5)))
    n_workers = config.get('n_workers', None)
    seed_mode = config.get('seed_mode', 'concat')
    n_seeds = config.get('N_seeds', None)
    do_concat = n_seeds is not None and seed_mode in ('concat', 'both')
    do_merge = n_seeds is not None and seed_mode in ('merge', 'both')

    # coverage region (intersection of masks), survey footprint, random-centre region
    cov_keys = _coverage_keys(config, config['cmb_map'])
    cov_frac, cov_labels = _coverage_region(cov_keys, data_folder, nside_cov, act_smooth)
    cov_tag = config.get('coverage_tag') or ('' if len(cov_keys) == 1
                                            else '+'.join(_short(l) for l in cov_labels))
    fp_dir = os.path.join(output_folder, 'Cache_Stacks', 'footprints')
    survey_fp = (fm.survey_footprint(cat.get('footprint'), data_folder, nside_cov, fp_dir,
                                     label=cat_label, force=config.get('force_footprint', False))
                 if random_pool == 'footprint' else None)
    region = np.where(cov_frac >= 0.5, 1.0, 0.0)
    if survey_fp is not None:
        region = region * survey_fp

    delta_label = _delta_label(config.get('delta_value'))
    seed_tag = _seed_tag(config)
    excl_txt = f'_excl{random_excl:g}' if random_excl else ''
    file_suffix = (f'{cat_label}_{map_label}_{binning_mode}{n_bins_q}bins_{exec_mode}_'
                   f'{seed_tag}_z{zmin:g}-{zmax:g}_r{rmin:g}-{rmax:g}_maxRv{max_Rvoid:g}_{delta_label}_'
                   f'{filter_label}_rand{random_pool}{excl_txt}'
                   + (f'_cov{cov_tag}' if cov_tag else '') + f'_bkg{bkg_mode.replace("+", "")}'
                   + _method_tag(config))
    run_folder = os.path.join(output_folder, file_suffix)
    os.makedirs(run_folder, exist_ok=True)
    stacks_cache_root = os.path.join(output_folder, 'Cache_Stacks', cat_label, map_label)

    print(f'######### CMB LENSING PROFILES | {cat_label} voids x {map_label} #########')
    print(f'Mode={exec_mode} | Binning={binning_mode} | coverage={"+".join(cov_labels)} '
          f'(>= {min_cov} of {max_Rvoid}Rv disc) | randoms={random_pool} | bkg={bkg_mode} | '
          f'n_workers={n_workers or os.cpu_count()}')
    print(f'Output Run Folder: {run_folder}\n')

    # map and stamp mask (native frame)
    lensing_map, nside, _ = _load_map(mspec, config)
    stamp_mask = fm.binary_mask(mspec['mask'], nside)
    map_frame = mspec['frame']

    # voids
    samples = load_void_samples(config, cat, cov_frac)
    if survey_fp is not None:
        ref = samples['merged'] if samples['merged'] is not None else (
            samples['single'] if samples['single'] is not None else pd.concat(samples['per_seed'].values()))
        if len(ref):
            inside = survey_fp[hp.ang2pix(nside_cov, ref['l'].values, ref['b'].values, lonlat=True)] > 0.5
            print(f'[footprint] {inside.mean() * 100:.1f}% of the selected void centres fall inside the '
                  f'{cat_label} survey footprint' + ('' if inside.mean() > 0.99 else
                                                     '  (!) check the footprint definition'))
    sample_lb_gal = None

    # binning (shared quantile edges between concat and merge)
    metric_col = {'redshift': 'z', 'radius': 'R_void'}[binning_mode]
    if samples['single'] is not None:
        edge_src = samples['single']
    elif samples['merged'] is not None:
        edge_src = samples['merged']
    else:
        edge_src = pd.concat(samples['per_seed'].values(), ignore_index=True)
    _, bin_edges = pd.qcut(edge_src[metric_col], n_bins_q, retbins=True, labels=False, duplicates='drop')
    bin_edges = np.asarray(bin_edges, float).copy()
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    n_bins_eff = len(bin_edges) - 1

    def _bin(df, i):
        return df[pd.cut(df[metric_col], bin_edges, labels=False) == i]

    base_params = {
        'map': map_label, 'klm': _file_sig(mspec['klm']), 'mask': _file_sig(mspec['mask']),
        'nlkk': _file_sig(mspec.get('nlkk')), 'cosmo': _cosmo_sig(),
        'filter': filter_label, 'smooth': config.get('smooth_value_arcmin', 0.0), 'nside': nside,
        'frame': map_frame, 'npix_stamp': npix_stamp, 'max_Rvoid': max_Rvoid,
        'bins_frac': np.round(bins_frac, 6).tolist(), 'exec_mode': exec_mode, 'n_subsamples': n_subsamples,
        'min_cov': min_cov, 'cov_keys': cov_labels, 'nside_cov': nside_cov, 'random_pool': random_pool,
        'footprint': cat.get('footprint') if random_pool == 'footprint' else None,
        'random_excl': random_excl, 'version': pm.CACHE_VERSION,
    }
    if profile_method != 'stamp':          # stamp keys stay as before: existing caches are reused
        base_params['profile'] = profile_method

    def _stack(df, kind):
        lon, lat = fm.to_map_frame(df['l'].values, df['b'].values, map_frame)
        center = fm.exclusion_mask(region, df['l'].values, df['b'].values, df['z'].values,
                                   df['R_void'].values, random_excl)
        key = _hash({**base_params, 'sample': _sample_hash(df)})[:16]
        cache_dir = os.path.join(stacks_cache_root, kind)
        os.makedirs(cache_dir, exist_ok=True)
        res = pm.process_bin_stacking_parallel(
            release=map_label, mode=exec_mode, data_sample_bin=df, coords_bin=(lon, lat),
            max_Rvoid=max_Rvoid, npix_stamp=npix_stamp, bins_frac=bins_frac, lensing_map=lensing_map,
            stamp_mask=stamp_mask, center_mask=center, cov_map=cov_frac, map_frame=map_frame,
            min_cov=min_cov, stacks_cache_folder=cache_dir, cache_key=key, n_random_factor=n_rand,
            n_rotations=n_rotations, n_subsamples=n_subsamples, force_rerun=force_rerun,
            n_workers=n_workers, bkg_mode=bkg_mode, bkg_outer=bkg_outer,
            profile_method=profile_method)
        res['R_void_median'] = float(np.median(df['R_void'].values))
        return res

    print(f"\n######### Doing profiles ({n_bins_eff} bin(s)) #########\n")
    all_results = []
    for i in range(n_bins_eff):
        print(f'--- Bin {i + 1} ---')
        if samples['single'] is not None:
            sub = _bin(samples['single'], i)
            if len(sub):
                res = _stack(sub, 'single')
                res.update(bin_id=i, binning_mode=binning_mode, catalog='single')
                all_results.append(res)
                if i == 0:
                    sample_lb_gal = (sub['l'].values, sub['b'].values)
            continue
        if do_concat:
            seed_results, seed_subs = [], []
            for s in range(n_seeds):
                sub = _bin(samples['per_seed'][s], i)
                if len(sub) == 0:
                    continue
                r = _stack(sub, f'seed_{s}')
                r['seed'] = s
                seed_results.append(r)
                seed_subs.append(sub)
            if seed_subs:
                allsub = pd.concat(seed_subs, ignore_index=True)
                print(f'[Profiles] Concatenated stack: {len(allsub)} voids across {n_seeds} seeds')
                res = _stack(allsub, 'concat')
                res.update(bin_id=i, binning_mode=binning_mode, catalog='concat', seed_results=seed_results)
                all_results.append(res)
                if sample_lb_gal is None and i == 0:
                    sample_lb_gal = (allsub['l'].values, allsub['b'].values)
        if do_merge:
            sub = _bin(samples['merged'], i)
            if len(sub):
                print(f'[Profiles] Merged stack: {len(sub)} unique voids')
                res = _stack(sub, 'merged')
                res.update(bin_id=i, binning_mode=binning_mode, catalog='merged')
                all_results.append(res)
                if i == 0:
                    sample_lb_gal = (sub['l'].values, sub['b'].values)

    if not all_results:
        print('[Results] (!) no voids selected: nothing to save')
        return None

    # SAVING
    rvoid_summary = fm.build_rvoid_summary(all_results)
    parameters = dict(config)
    parameters.update({'profile_method': profile_method, 'map_label': map_label, 'catalog_label': cat_label, 'filter_label': filter_label,
                       'coverage_keys': cov_labels, 'coverage_tag': cov_tag, 'npix_stamp_eff': npix_stamp,
                       'run_folder': run_folder, 'R_void_median_per_sample': rvoid_summary})
    print('[Results] Saving results...')
    fm.plot_stacked_maps_and_profiles(all_results, os.path.join(run_folder, f'Stacked_Maps_NullTests_{file_suffix}.pdf'),
                                      max_Rvoid)
    fm.plot_jackknife_and_correlation(all_results, os.path.join(run_folder, f'JK_Profile_Correlation_{file_suffix}.pdf'),
                                      max_Rvoid)
    if any(d.get('catalog') == 'concat' for d in all_results):
        fm.plot_seed_consistency(all_results, os.path.join(run_folder, f'Seed_Consistency_{file_suffix}.pdf'),
                                 max_Rvoid)
    if do_concat and do_merge:
        pairs = []
        for rc in [r for r in all_results if r['catalog'] == 'concat']:
            rm = next((m for m in all_results if m['catalog'] == 'merged' and m['bin_id'] == rc['bin_id']), None)
            if rm is not None:
                pairs.append({'bin_id': rc['bin_id'], 'concat': rc, 'merge': rm})
        if pairs:
            fm.plot_merge_vs_concat(pairs, os.path.join(run_folder, f'Merge_vs_Concat_{file_suffix}.pdf'), max_Rvoid)

    pkl_path = os.path.join(run_folder, f'Data_FullRun_{file_suffix}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'results': all_results, 'parameters': parameters}, f)
    print(f'[Results] Data saved in: {pkl_path}')

    out = _primary_result(all_results, map_label, sample_lb_gal)
    out.update(pkl_path=pkl_path, run_folder=run_folder, catalog_label=cat_label)
    return out


def _primary_result(all_results, map_label, sample_lb_gal=None):
    out = None
    for kind in ('merged', 'concat', 'single'):
        out = next((dict(r) for r in all_results
                    if r.get('catalog') == kind and int(r.get('bin_id', 0)) == 0), None)
        if out is not None:
            break
    out = out or dict(all_results[0])
    out['map_label'] = map_label
    if sample_lb_gal is not None:
        out['void_l'], out['void_b'] = sample_lb_gal
    return out


#%% Dispatcher
def _sel_tag(config):
    return (f"{config['binning_mode']}{config['n_bins']}bins_{_seed_tag(config)}_z{config['zmin']:g}-{config['zmax']:g}_"
            f"r{config['rmin']:g}-{config['rmax']:g}_{_delta_label(config.get('delta_value'))}_"
            f"bkg{config.get('bkg_mode', 'randoms').replace('+', '')}{_method_tag(config)}")


def _run_group(config, keys, tag):
    """Several maps on the SAME voids: coverage = intersection of all their masks."""
    data_folder = config['data_folder']
    print(f"\n######### {tag}: {len(keys)} maps -> {keys} #########")
    common = config.get('common_coverage', True)
    collected = {}
    for k in keys:
        cfg = dict(config, cmb_map=k)
        if common:
            cfg.update(coverage_map=list(keys), coverage_tag=tag)
        print(f"\n================= MAP {k}  ({tag}) =================")
        try:
            res = _run_single_map(cfg)
        except FileNotFoundError as e:
            print(f"[Results] '{k}' skipped (missing file): {e}")
            continue
        if res is not None:
            collected[res['map_label']] = res
    if common and len({r['n_voids'] for r in collected.values()}) > 1:
        print('[Results] (!) different N between maps although the coverage region is common')
    cat_label = _catalog_spec(config.get('void_catalog', 'WH'), data_folder)['label']
    if len(collected) >= 2:
        order = [_map_spec(k, data_folder)['label'] for k in keys]
        path = os.path.join(config['output_folder'], 'lensing',
                            f"MapComparison_{tag}_{cat_label}_{_sel_tag(config)}.pdf")
        fm.plot_map_comparison(collected, path, config['max_Rvoid'], success_label='PLANCK_PR4', order=order,
                               title=f"{cat_label} voids | coverage: {tag if common else 'own mask'}")
    return collected


def _run_act_vs_pr4(config):
    """PR4 and ACT on the same voids: coverage = ACT mask x PR4 mask."""
    print("\n#########  PR4 vs ACT on the common ACT x PR4 footprint #########")
    data_folder = config['data_folder']
    keys = ['PLANCK_PR4', 'ACT']
    collected = {}
    for k in keys:
        cfg = dict(config, cmb_map=k, coverage_map=list(keys), coverage_tag=GROUP_TAGS['ACT_PR4'])
        print(f"\n================= {k} =================")
        try:
            res = _run_single_map(cfg)
        except FileNotFoundError as e:
            print(f"[act-pr4] {k} skipped (missing file): {e}")
            continue
        if res is not None:
            collected[res['map_label']] = res
    if len(collected) < 2:
        print('[act-pr4] missing results.')
        return collected

    dp, da = collected['PLANCK_PR4'], collected['ACT']
    if all(k in d for d in (dp, da) for k in ('void_l', 'void_b')):
        tol = 1e-4
        setp = set(zip(np.round(np.asarray(dp['void_l']) / tol).astype(np.int64),
                       np.round(np.asarray(dp['void_b']) / tol).astype(np.int64)))
        seta = set(zip(np.round(np.asarray(da['void_l']) / tol).astype(np.int64),
                       np.round(np.asarray(da['void_b']) / tol).astype(np.int64)))
        print(f"[act-pr4] PR4 N={len(setp)}, ACT N={len(seta)}, common={len(setp & seta)} -> "
              + ('OK: same voids' if setp == seta else '(!) the samples are different'))
    act_mask = fm.binary_mask(_map_spec('ACT', data_folder)['mask'])
    cat_label = _catalog_spec(config.get('void_catalog', 'WH'), data_folder)['label']
    path = os.path.join(config['output_folder'], 'lensing', f"ACTvsPR4_{cat_label}_{_sel_tag(config)}.pdf")
    fm.plot_act_vs_pr4(collected, path, config['max_Rvoid'], success_label='PLANCK_PR4', other_label='ACT',
                       footprint_mask=act_mask, footprint_coord='C')
    return collected


def _run_for_catalog(config):
    """Runs whatever cmb_map asks for (one map, a group or act-pr4) for ONE catalogue."""
    key = _map_key(config.get('cmb_map', 'PLANCK_PR4'))
    if key == 'ACT_PR4':
        return _run_act_vs_pr4(config), ['PLANCK_PR4', 'ACT'], GROUP_TAGS['ACT_PR4']
    if key in MAP_GROUPS:
        keys = MAP_GROUPS[key]
        order = [_map_spec(k, config['data_folder'])['label'] for k in keys]
        return _run_group(config, keys, GROUP_TAGS[key]), order, GROUP_TAGS[key]
    res = _run_single_map(config)
    return ({res['map_label']: res} if res is not None else {}), None, _short(_map_spec(key, config['data_folder'])['label'])


def run_pipeline(config):
    cat_req = str(config.get('void_catalog', 'WH')).upper()
    cats = CATALOG_GROUPS.get(cat_req, [cat_req])
    by_cat, order, tag = {}, None, ''
    for c in cats:
        print(f"\n\n%%%%%%%%%%%%%%%%%%%%  VOID CATALOGUE: {c}  %%%%%%%%%%%%%%%%%%%%")
        collected, order, tag = _run_for_catalog(dict(config, void_catalog=c))
        by_cat[c] = collected
    if len(cats) > 1:
        path = os.path.join(config['output_folder'], 'lensing',
                            f"CatalogColumns_{tag}_{'-'.join(cats)}_{_sel_tag(config)}.pdf")
        fm.plot_catalog_columns(by_cat, path, config['max_Rvoid'], order=order, ref_label='PLANCK_PR4',
                                suptitle=f"{tag} | " + _sel_tag(config))
    return by_cat


if __name__ == "__main__":
    print("Please run from Pipeline_voids.py")
