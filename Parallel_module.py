##### Parallel stacking module for the CMB lensing voids profiles #####
# Mirrors Functions_module.process_bin_stacking, but parallelizes the
# embarrassingly-parallel parts (random null, rotation null, jackknife/signal
# region stacks) over CPU cores with a fork-based ProcessPoolExecutor.
#
# Design notes
# ------------
# * The big read-only arrays (CMB map and stamp mask) are stored as module
#   globals and inherited by the worker processes via fork (copy-on-write), so
#   they are NOT pickled per task. Only the (small) per-void coordinate arrays
#   travel to the workers.
# * Rotation null tests are implemented as rigid longitude shifts of the void
#   positions (l -> l-ang) instead of rotating the whole healpix map. For a
#   pure z-rotation rot=[ang,0,0] this samples exactly the same sky pixels, so
#   it is equivalent to the serial map-rotation for radially-averaged profiles,
#   but avoids building 400 MB rotated maps in every worker.
# * Cache file names and the returned dict are kept identical to the serial
#   Functions_module.process_bin_stacking, so serial and parallel caches are
#   interchangeable.

import os
import pickle
import warnings
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans

import Functions_module as fm

# --- Worker globals (set before the pool is created; inherited via fork) ------
_W_CMB = None     # CMB lensing map
_W_STAMP = None   # stamp mask used while stacking (common_mask)


def _set_worker_globals(cmb_map, stamp_mask):
    global _W_CMB, _W_STAMP
    _W_CMB, _W_STAMP = cmb_map, stamp_mask


# --- Workers ------------------------------------------------------------------
def _profile_worker(task):
    """Stack at given positions and return the count-weighted radial profile."""
    pos_l, pos_b, z, rv, max_Rvoid, npix_stamp, bins_frac = task
    s, c = fm.stacking_gnomonic(pos_l, pos_b, z, rv, _W_CMB, _W_STAMP,
                                max_Rvoid, npix_stamp, range(len(pos_l)), silent=True)
    prof, _ = fm.radial_profile_weighted(s, c, max_Rvoid, bins_frac, silent=True)
    return prof


def _region_worker(task):
    """Stack a void subset and return its (sum_map, count_map) for JK/signal."""
    l_sub, b_sub, z_sub, rv_sub, max_Rvoid, npix_stamp = task
    s, c = fm.stacking_gnomonic(l_sub, b_sub, z_sub, rv_sub, _W_CMB, _W_STAMP,
                                max_Rvoid, npix_stamp, range(len(l_sub)), silent=True)
    return s, c


# --- Parallel null tests ------------------------------------------------------
def _rotation_null_parallel(executor, l, b, redshifts, r_voids, max_Rvoid, npix_stamp,
                            bins_frac, n_rotations, existing_profiles):
    n_old = len(existing_profiles)
    if n_rotations <= n_old:
        return np.array(existing_profiles[:n_rotations])
    print(f'[parallel] {n_rotations - n_old} additional longitude-shift rotations...')
    np.random.seed(42 + n_old)
    angles = np.random.uniform(10, 350, n_rotations - n_old)
    tasks = [((l - ang) % 360.0, b, redshifts, r_voids, max_Rvoid, npix_stamp, bins_frac)
             for ang in angles]
    new_profiles = list(executor.map(_profile_worker, tasks))
    return np.array(list(existing_profiles) + new_profiles)


def _random_null_parallel(executor, nside, redshifts, r_voids, position_mask, max_Rvoid,
                          npix_stamp, bins_frac, n_random_factor, existing_profiles):
    n_old = len(existing_profiles)
    if n_random_factor <= n_old:
        return np.array(existing_profiles[:n_random_factor])
    print(f'[parallel] {n_random_factor - n_old} additional random realizations...')
    n_voids = len(redshifts)
    tasks = []
    for i in range(n_old, n_random_factor):
        np.random.seed(1000 + i)  # deterministic & resumable
        rand_l, rand_b = fm.generate_random(position_mask, n_voids, nside)
        perm = np.random.permutation(n_voids)
        tasks.append((rand_l, rand_b, redshifts[perm], r_voids[perm],
                      max_Rvoid, npix_stamp, bins_frac))
    new_profiles = list(executor.map(_profile_worker, tasks))
    return np.array(list(existing_profiles) + new_profiles)


# --- Parallel signal + jackknife ----------------------------------------------
def _signal_and_jk_parallel(executor, 
                            l, b, redshifts, 
                            r_voids, max_Rvoid,
                            npix_stamp, bins_frac,
                            mode, n_subsamples, n_workers=None):
    n_voids = len(l)

    if mode == 'errors':
        ra_rad, dec_rad = np.radians(l), np.radians(b)
        xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad),
                               np.cos(dec_rad) * np.sin(ra_rad),
                               np.sin(dec_rad)])
        print(f'[parallel] Dividing {n_voids} voids into {n_subsamples} jackknife regions (KMeans)...')
        labels = KMeans(n_clusters=n_subsamples, random_state=42, n_init=10).fit_predict(xyz)
        regions = [np.where(labels == k)[0] for k in range(n_subsamples)]
    else:
        n_chunks = max(1, n_workers)
        regions = [idx for idx in np.array_split(np.arange(n_voids), n_chunks) if len(idx) > 0]

    tasks = []
    for idx in regions:
        if len(idx) == 0:
            tasks.append(None)
            continue
        tasks.append((l[idx], b[idx], redshifts[idx], r_voids[idx], max_Rvoid, npix_stamp))

    results, valid_tasks = [], [(j, t) for j, t in enumerate(tasks) if t is not None]
    out = list(executor.map(_region_worker, [t for _, t in valid_tasks]))
    shape = out[0][0].shape
    sums = [np.zeros(shape) for _ in tasks]
    counts = [np.zeros(shape) for _ in tasks]
    for (j, _), (s, c) in zip(valid_tasks, out):
        sums[j], counts[j] = s, c
    sums, counts = np.array(sums), np.array(counts)

    SUM, COUNT = sums.sum(axis=0), counts.sum(axis=0)
    signal_map = fm.stack_mean_map(SUM, COUNT)
    prof_total, r_frac = fm.radial_profile_weighted(SUM, COUNT, max_Rvoid, bins_frac, silent=True)

    if mode == 'errors':
        jk_profiles = []
        for k in range(len(regions)):
            prof, _ = fm.radial_profile_weighted(SUM - sums[k], COUNT - counts[k],
                                                 max_Rvoid, bins_frac, silent=True)
            jk_profiles.append(prof)
        jk_profiles = np.array(jk_profiles)

        cov_matrix, err = fm.sample_covariance(jk_profiles, kind='jk')
    else:
        jk_profiles, cov_matrix = None, None
        err = np.zeros_like(prof_total)

    return signal_map, prof_total, r_frac, err, jk_profiles, cov_matrix


# --- Main entry point  --------------
def process_bin_stacking_parallel(release, mode, z_min, z_max, r_min, r_max, data_sample_bin,
                                  coords_bin, max_Rvoid, npix_stamp, nside, bins_frac, lensing_map,
                                  common_mask, stacks_cache_folder, n_random_factor, n_rotations,
                                  n_subsamples=20, delta_label='d23_all', filter_label='none',
                                  force_rerun=False, random_pool='full', random_excl_factor=1.0,
                                  n_workers=None):
    z_text = f'{z_min:.2f}_{z_max:.2f}'
    r_text = f'{r_min:.1f}_{r_max:.1f}'
    z_mean, n_voids = data_sample_bin['z'].mean(), len(data_sample_bin)
    l, b, redshifts_all, r_voids_all = coords_bin[0], coords_bin[1], data_sample_bin['z'].values, data_sample_bin['R_void'].values

    n_workers = n_workers or os.cpu_count()
    print(f'[parallel] Bin z in [{z_min:.2f},{z_max:.2f}], {n_voids} voids (mean z={z_mean:.3f}), n_workers={n_workers}...')

    # --- Random null pool (same construction as the serial version) ---
    if random_pool == 'survey':
        random_base_mask = common_mask * fm.footprint_mask(l, b, output_nside=nside)
    else:
        random_base_mask = common_mask
    random_position_mask = fm.build_random_exclusion_mask(
        random_base_mask, l, b, redshifts_all, r_voids_all, random_excl_factor, nside)
    rand_label = f'rpool{random_pool}_excl{(random_excl_factor or 0):.1f}'

    n_rotations = n_rotations if n_rotations is not None else 0
    n_random_factor = n_random_factor if n_random_factor is not None else 0

    signal_cache_file = os.path.join(stacks_cache_folder, f'signal_cache_{release}_{z_text}_{r_text}_N{n_voids}_{delta_label}_{filter_label}_maxRv{max_Rvoid:.1f}_{rand_label}.pkl')

    if not force_rerun and os.path.exists(signal_cache_file):
        with open(signal_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        if cached_data.get('n_rotations_done', 0) >= n_rotations and cached_data.get('n_randoms_done', 0) >= n_random_factor:
            print(f'[parallel] Loading fully cached result from {signal_cache_file}.')
            return cached_data
        print('[parallel] Signal cached, but more null tests requested. Updating...')
        signal_data = cached_data
    else:
        signal_data = None

    null_cache_file = os.path.join(stacks_cache_folder, f'null_tests_{release}_{z_text}_{r_text}_N{n_voids}_{delta_label}_{filter_label}_maxRv{max_Rvoid:.1f}_{rand_label}.npz')
    existing_rot, existing_rand = [], []
    if not force_rerun and os.path.exists(null_cache_file):
        data_cache = np.load(null_cache_file)
        if 'null_profiles_rot' in data_cache: existing_rot = list(data_cache['null_profiles_rot'])
        if 'null_profiles_rand' in data_cache: existing_rand = list(data_cache['null_profiles_rand'])

    # Publish big read-only arrays to workers (inherited via fork), then open pool.
    _set_worker_globals(lensing_map, common_mask)
    ctx = mp.get_context('fork')

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        null_profiles_rot = (_rotation_null_parallel(executor, l, b, redshifts_all, r_voids_all,
                                                     max_Rvoid, npix_stamp, bins_frac, n_rotations, existing_rot)
                             if n_rotations > 0 else np.array([]))
        null_profiles_rand = (_random_null_parallel(executor, nside, redshifts_all, r_voids_all,
                                                    random_position_mask, max_Rvoid, npix_stamp, bins_frac,
                                                    n_random_factor, existing_rand)
                              if n_random_factor > 0 else np.array([]))

        cov_cmb, err_cmb = fm.sample_covariance(null_profiles_rand, kind='cmb')

        if n_rotations > 0 or n_random_factor > 0:
            np.savez(null_cache_file, null_profiles_rot=null_profiles_rot, null_profiles_rand=null_profiles_rand)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            null_rot_mean = np.nanmean(null_profiles_rot, axis=0) if len(null_profiles_rot) > 0 else np.full(len(bins_frac) - 1, np.nan)
            null_rot_std = np.nanstd(null_profiles_rot, axis=0) if len(null_profiles_rot) > 0 else np.full(len(bins_frac) - 1, np.nan)
            null_rand_mean = np.nanmean(null_profiles_rand, axis=0) if len(null_profiles_rand) > 0 else np.full(len(bins_frac) - 1, np.nan)
            null_rand_std = np.nanstd(null_profiles_rand, axis=0) if len(null_profiles_rand) > 0 else np.full(len(bins_frac) - 1, np.nan)

        if signal_data is not None:
            signal_data.update({
                'null_rot_mean': null_rot_mean, 'null_rot_std': null_rot_std,
                'null_rand_mean': null_rand_mean, 'null_rand_std': null_rand_std,
                'n_rotations_done': max(n_rotations, signal_data.get('n_rotations_done', 0)),
                'n_randoms_done': max(n_random_factor, signal_data.get('n_randoms_done', 0)),
                'cov_cmb': cov_cmb,
                'err_cmb': err_cmb
            })
            with open(signal_cache_file, 'wb') as f:
                pickle.dump(signal_data, f)
            return signal_data

        print('[parallel] Computing signal stack and errors...')
        signal_map, prof_total, r_frac, prof_err, jk_profiles, cov_jk = _signal_and_jk_parallel(
            executor, l, b, redshifts_all, r_voids_all, max_Rvoid, npix_stamp, bins_frac, mode, n_subsamples, n_workers)

    final_result = {
        'z_mean': z_mean, 'map': signal_map, 'r_frac': r_frac, 'profile': prof_total,
        'error': prof_err, 'jk_profiles': jk_profiles, 'cov_jk': cov_jk, 'cov_cmb': cov_cmb,
        'null_rot_mean': null_rot_mean, 'null_rot_std': null_rot_std,
        'null_rand_mean': null_rand_mean, 'null_rand_std': null_rand_std,
        'n_voids': n_voids, 'key': z_text, 'n_rotations_done': n_rotations, 'n_randoms_done': n_random_factor,
    }
    with open(signal_cache_file, 'wb') as f:
        pickle.dump(final_result, f)
    return final_result
