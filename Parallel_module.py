##### Stacking module for the CMB lensing voids profiles (multi-core) #####
# ---------------------------------------------------------------------

#%% Imports
import os
import pickle
import warnings
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
import Functions_module as fm

CACHE_VERSION = 'v2'

#%% Worker globals (inherited through fork)
_W = {}


def _set_worker_globals(**kw):
    _W.clear()
    _W.update(kw)


def _profile_worker(task):
    """Stack at the given positions (map frame) and return the radial profile."""
    pos_l, pos_b, z, rv, max_Rvoid, npix_stamp, bins_frac = task
    s, c = fm.stacking_gnomonic(pos_l, pos_b, z, rv, _W['cmb'], _W['stamp_mask'],
                                max_Rvoid, npix_stamp, range(len(pos_l)), silent=True)
    prof, _ = fm.radial_profile_weighted(s, c, max_Rvoid, bins_frac, silent=True)
    return prof


def _random_worker(task):
    """
    One random realization: the (z, R_v) of the voids are permuted and each one is
    placed at a random position inside the random region (map mask x survey
    footprint) whose disc passes the same coverage criterion as the real voids.
    """
    i, z, rv, max_Rvoid, npix_stamp, bins_frac, rad_frac, min_cov = task
    rng = np.random.default_rng(1000 + i)
    perm = rng.permutation(len(z))
    l, b, zz, rr, n_swap = fm.matched_randoms(z[perm], rv[perm], _W['center'], _W['cov'],
                                              rad_frac, min_cov, rng)
    lon, lat = fm.to_map_frame(l, b, _W['frame'])
    s, c = fm.stacking_gnomonic(lon, lat, zz, rr, _W['cmb'], _W['stamp_mask'],
                                max_Rvoid, npix_stamp, range(len(lon)), silent=True)
    prof, _ = fm.radial_profile_weighted(s, c, max_Rvoid, bins_frac, silent=True)
    return prof, n_swap


def _region_worker(task):
    """Stack a void subset and return its (sum_map, count_map) for JK/signal."""
    l_sub, b_sub, z_sub, rv_sub, max_Rvoid, npix_stamp = task
    return fm.stacking_gnomonic(l_sub, b_sub, z_sub, rv_sub, _W['cmb'], _W['stamp_mask'],
                                max_Rvoid, npix_stamp, range(len(l_sub)), silent=True)


#%% Null tests
def _rotation_null_parallel(executor, l, b, redshifts, r_voids, max_Rvoid, npix_stamp,
                            bins_frac, n_rotations, existing_profiles):
    """Rigid longitude shifts of the void catalogue over the map (map-only null)."""
    n_old = len(existing_profiles)
    if n_rotations <= n_old:
        return np.array(existing_profiles[:n_rotations])
    print(f'[Null Test] {n_rotations - n_old} longitude-shift rotations...')
    rng = np.random.default_rng(42 + n_old)
    angles = rng.uniform(10, 350, n_rotations - n_old)
    tasks = [((l - ang) % 360.0, b, redshifts, r_voids, max_Rvoid, npix_stamp, bins_frac)
             for ang in angles]
    new_profiles = list(executor.map(_profile_worker, tasks))
    return np.array(list(existing_profiles) + new_profiles)


def _random_null_parallel(executor, redshifts, r_voids, max_Rvoid, npix_stamp, bins_frac,
                          n_random, existing_profiles, rad_frac, min_cov):
    n_old = len(existing_profiles)
    if n_random <= n_old:
        return np.array(existing_profiles[:n_random]), 0
    print(f'[Null Test] {n_random - n_old} random realizations (footprint-matched)...')
    tasks = [(i, redshifts, r_voids, max_Rvoid, npix_stamp, bins_frac, rad_frac, min_cov)
             for i in range(n_old, n_random)]
    out = list(executor.map(_random_worker, tasks))
    n_swap = int(sum(o[1] for o in out))
    if n_swap:
        print(f'[Null Test] (!) {n_swap} random objects could not be placed with their own (z, Rv) '
              f'and were re-assigned another (z, Rv) of the sample '
              f'({n_swap / (len(out) * len(redshifts)) * 100:.3f}% of all random objects).')
    return np.array(list(existing_profiles) + [o[0] for o in out]), n_swap


#%% Signal + jackknife
def _signal_and_jk_parallel(executor, l, b, redshifts, r_voids, max_Rvoid, npix_stamp, bins_frac,
                            mode, n_subsamples, n_workers):
    n_voids = len(l)
    if mode == 'errors':
        lr, br = np.radians(l), np.radians(b)
        xyz = np.column_stack([np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)])
        n_sub = min(n_subsamples, n_voids)
        print(f'[Profiles] Dividing {n_voids} voids into {n_sub} jackknife regions...')
        labels = KMeans(n_clusters=n_sub, random_state=42, n_init=10).fit_predict(xyz)
        regions = [np.where(labels == k)[0] for k in range(n_sub)]
    else:
        regions = [idx for idx in np.array_split(np.arange(n_voids), max(1, n_workers)) if len(idx)]
    regions = [idx for idx in regions if len(idx) > 0]
    tasks = [(l[idx], b[idx], redshifts[idx], r_voids[idx], max_Rvoid, npix_stamp) for idx in regions]
    out = list(executor.map(_region_worker, tasks))
    sums = np.array([s for s, _ in out])
    counts = np.array([c for _, c in out])

    SUM, COUNT = sums.sum(axis=0), counts.sum(axis=0)
    signal_map = fm.stack_mean_map(SUM, COUNT)
    prof_total, r_frac = fm.radial_profile_weighted(SUM, COUNT, max_Rvoid, bins_frac, silent=True)

    if mode == 'errors':
        jk_profiles = np.array([fm.radial_profile_weighted(SUM - sums[k], COUNT - counts[k],
                                                           max_Rvoid, bins_frac, silent=True)[0]
                                for k in range(len(regions))])
        cov_matrix, err = fm.sample_covariance(jk_profiles, kind='jk')
    else:
        jk_profiles, cov_matrix = None, None
        err = np.zeros_like(prof_total)
    return signal_map, prof_total, r_frac, err, jk_profiles, cov_matrix


#%% Main
def process_bin_stacking_parallel(release, mode, data_sample_bin, coords_bin, max_Rvoid, npix_stamp,
                                  bins_frac, lensing_map, stamp_mask, center_mask, cov_map, map_frame,
                                  min_cov, stacks_cache_folder, cache_key, n_random_factor, n_rotations,
                                  n_subsamples=20, force_rerun=False, n_workers=None,
                                  bkg_mode='randoms', bkg_outer=(2.0, 2.5)):
    """
    Stack one void sample on one map, with jackknife errors and the two null tests.

    data_sample_bin : DataFrame with galactic l, b, z, R_void of the sample.
    coords_bin      : (lon, lat) of the sample in the map frame.
    stamp_mask      : map mask in the map frame (pixels used in each stamp).
    center_mask     : galactic mask where random centres may fall (map mask x survey footprint).
    cov_map         : galactic coverage map used to select voids (and randoms).
    cache_key       : hash of everything that determines the result (sample + map + settings).
    """
    z_mean, n_voids = data_sample_bin['z'].mean(), len(data_sample_bin)
    z_text = f"{data_sample_bin['z'].min():.2f}_{data_sample_bin['z'].max():.2f}"
    l, b = np.asarray(coords_bin[0], float), np.asarray(coords_bin[1], float)
    redshifts_all = data_sample_bin['z'].values.astype(float)
    r_voids_all = data_sample_bin['R_void'].values.astype(float)

    n_workers = n_workers or os.cpu_count()
    print(f'[Data] Sample z in [{z_text.replace("_", ",")}], {n_voids} voids (mean z={z_mean:.3f}), '
          f'n_workers={n_workers}...')

    n_rotations = n_rotations or 0
    n_random_factor = n_random_factor or 0

    tag = f'{release}_{CACHE_VERSION}_{cache_key}'
    signal_cache_file = os.path.join(stacks_cache_folder, f'signal_{tag}.pkl')
    null_cache_file = os.path.join(stacks_cache_folder, f'nulls_{tag}.npz')

    signal_data = None
    if not force_rerun and os.path.exists(signal_cache_file):
        with open(signal_cache_file, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('n_rotations_done', 0) >= n_rotations and cached.get('n_randoms_done', 0) >= n_random_factor:
            print(f'[Data] Loading fully cached result {os.path.basename(signal_cache_file)}.')
            cached.update(bkg_mode=bkg_mode, bkg_outer=tuple(bkg_outer))
            return cached
        print('[Data] Signal cached, but more null tests requested. Updating...')
        signal_data = cached

    existing_rot, existing_rand = [], []
    if not force_rerun and os.path.exists(null_cache_file):
        dc = np.load(null_cache_file)
        existing_rot = list(dc['null_profiles_rot']) if 'null_profiles_rot' in dc else []
        existing_rand = list(dc['null_profiles_rand']) if 'null_profiles_rand' in dc else []

    _set_worker_globals(cmb=lensing_map, stamp_mask=stamp_mask, center=center_mask,
                        cov=cov_map, frame=map_frame)
    ctx = mp.get_context('fork')
    n_swap = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        null_rot = (_rotation_null_parallel(executor, l, b, redshifts_all, r_voids_all, max_Rvoid,
                                            npix_stamp, bins_frac, n_rotations, existing_rot)
                    if n_rotations > 0 else np.empty((0, len(bins_frac) - 1)))
        if n_random_factor > 0:
            null_rand, n_swap = _random_null_parallel(executor, redshifts_all, r_voids_all, max_Rvoid,
                                                      npix_stamp, bins_frac, n_random_factor, existing_rand,
                                                      rad_frac=max_Rvoid, min_cov=min_cov)
        else:
            null_rand = np.empty((0, len(bins_frac) - 1))
        if n_rotations > 0 or n_random_factor > 0:
            np.savez(null_cache_file, null_profiles_rot=null_rot, null_profiles_rand=null_rand)

        if signal_data is None:
            print('[Profiles] Computing signal stack and errors...')
            signal_map, prof_total, r_frac, prof_err, jk_profiles, cov_jk = _signal_and_jk_parallel(
                executor, l, b, redshifts_all, r_voids_all, max_Rvoid, npix_stamp, bins_frac, mode,
                n_subsamples, n_workers)

    cov_cmb, err_cmb = fm.sample_covariance(null_rand, kind='cmb')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        nb = len(bins_frac) - 1
        stats = {
            'null_rot_mean': np.nanmean(null_rot, 0) if len(null_rot) else np.full(nb, np.nan),
            'null_rot_std': np.nanstd(null_rot, 0) if len(null_rot) else np.full(nb, np.nan),
            'null_rand_mean': np.nanmean(null_rand, 0) if len(null_rand) else np.full(nb, np.nan),
            'null_rand_std': np.nanstd(null_rand, 0) if len(null_rand) else np.full(nb, np.nan),
        }
    nulls = {'null_profiles_rot': null_rot, 'null_profiles_rand': null_rand,
             'cov_cmb': cov_cmb, 'err_cmb': err_cmb,
             'n_rotations_done': len(null_rot), 'n_randoms_done': len(null_rand),
             'random_swaps': n_swap, **stats}

    if signal_data is not None:
        signal_data.update(nulls)
        final_result = signal_data
    else:
        final_result = {
            'z_mean': z_mean, 'map': signal_map, 'r_frac': r_frac, 'profile': prof_total,
            'error': prof_err, 'jk_profiles': jk_profiles, 'cov_jk': cov_jk,
            'n_voids': n_voids, 'key': z_text, 'cache_key': cache_key, 'npix_stamp': npix_stamp,
            **nulls,
        }
    final_result.update(bkg_mode=bkg_mode, bkg_outer=tuple(bkg_outer))
    with open(signal_cache_file, 'wb') as f:
        pickle.dump(final_result, f)
    return final_result
