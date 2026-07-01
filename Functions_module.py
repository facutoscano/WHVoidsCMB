##### Function module for the CMB lensing voids profiles #####

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import healpy as hp
import pandas as pd
import pickle
from astropy import units as u
from astropy.cosmology import Planck18
from sklearn.cluster import KMeans
import warnings

plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',             
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.5, 'lines.markersize': 5,
    'legend.fontsize': 11, 'legend.frameon': False             
})

#%% AUXILIARY FUNCTIONS
def get_angularsize_comoving(z, size_mpch):
    h = Planck18.h
    d_c = Planck18.comoving_distance(z).value * h
    theta_rad = size_mpch / d_c
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def apply_wiener_filter(cmb_alm, nlkk_file, lmax=2048):
    nlkk_data = np.loadtxt(nlkk_file)
    ell_nlkk = nlkk_data[:, 0].astype(int)
    nl_kk = nlkk_data[:, 1]     # noise
    sn_kk = nlkk_data[:, 2]     # signal + noise

    cl_kk = sn_kk - nl_kk
    cl_kk = np.maximum(cl_kk, 0)        # avoiding negative Cls

    W = np.zeros(lmax + 1)
    for i, ell in enumerate(ell_nlkk):
        if ell > lmax:
            break
        denom = cl_kk[i] + nl_kk[i]
        W[ell] = cl_kk[i] / denom if denom > 0 else 0.0
    
    alm_filtered = hp.almxfl(cmb_alm.copy(), W)

    print(f'Wiener filter applied. W_ell range: '
          f'W[10]={W[10]:.3f}, W[100]={W[100]:.3f}, '
          f'W[500]={W[500]:.3f}, W[1000]={W[1000]:.3f}')

    return alm_filtered, W


def footprint_mask(l, b, output_nside, footprint_nside=32): 
    npix_footprint = hp.nside2npix(footprint_nside)
    footprint_mask = np.zeros(npix_footprint, dtype=np.float32)
    pix_indices = hp.ang2pix(footprint_nside, l, b, lonlat=True)
    footprint_mask[pix_indices] = 1.0
    output_mask = hp.ud_grade(footprint_mask, nside_out=output_nside)
    output_mask[output_mask > 0] = 1.0
    return output_mask


def generate_random(mask, n_random, nside):
    valid_l, valid_b, count = [], [], 0
    while count < n_random:
        l_batch = np.random.uniform(0.0, 360.0, int(n_random * 1.5))
        sin_b_batch = np.random.uniform(-1.0, 1.0, int(n_random * 1.5))
        b_batch = np.degrees(np.arcsin(sin_b_batch))
        pix_batch = hp.ang2pix(nside, l_batch, b_batch, lonlat=True)
        is_valid = mask[pix_batch] > 0.9 if not np.any(np.isnan(mask)) else (~np.isnan(mask[pix_batch]) & (mask[pix_batch] > 0.9))
        valid_l.extend(l_batch[is_valid])
        valid_b.extend(b_batch[is_valid])
        count = len(valid_l)
    return np.array(valid_l[:n_random]), np.array(valid_b[:n_random])


def rotate_map(map_data, rot_angles):
    nside = hp.npix2nside(len(map_data))
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    r = hp.rotator.Rotator(rot=rot_angles, deg=True, inv=True)
    theta_rot, phi_rot = r(theta, phi)
    pix_rot = hp.ang2pix(nside, theta_rot, phi_rot)
    return map_data[pix_rot]


def stacking_gnomonic(l, b, redshifts, r_voids, cmb_map, mask, max_Rvoid, npix_stamp, stacked_range, silent=False):
    if not silent: print(f'Stacking {len(stacked_range)} maps using scaled Rv...')
    sum_map   = np.zeros((npix_stamp, npix_stamp))
    count_map = np.zeros((npix_stamp, npix_stamp))
    nside = hp.npix2nside(len(cmb_map))
    vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    for i, idx in enumerate(stacked_range):
        cl_l, cl_b, cl_z, cl_rv = l[idx], b[idx], redshifts[idx], r_voids[idx]
        box_size_mpch = 2 * max_Rvoid * cl_rv
        box_size_deg  = get_angularsize_comoving(cl_z, box_size_mpch)
        reso_arcmin   = (box_size_deg * 60.) / npix_stamp

        proj = hp.projector.GnomonicProj(rot=[cl_l, cl_b, 0], xsize=npix_stamp, ysize=npix_stamp, reso=reso_arcmin)
        stamp_data = proj.projmap(cmb_map, vec2pix_func=vec2pix_func)
        stamp_mask = proj.projmap(mask,    vec2pix_func=vec2pix_func)
        valid = (stamp_mask > 0.9) & (~np.isnan(stamp_data))

        sum_map[valid]   += stamp_data[valid]
        count_map[valid] += 1
        if not silent and (i+1) % 150 == 0: print(f'Stacked {i+1} / {len(stacked_range)}')

    return sum_map, count_map

def stack_mean_map(sum_map, count_map):
    out = np.full_like(sum_map, np.nan)
    good = count_map > 0
    out[good] = sum_map[good] / count_map[good]
    return out

def radial_profile_flat(stack_map, max_Rvoid, bins_frac, silent=False):
    npix = stack_map.shape[0]
    center = npix // 2
    y, x = np.ogrid[-center:npix-center, -center:npix-center]
    r_pixels = np.sqrt(x*x + y*y)

    Rv_per_pixel = (2 * max_Rvoid) / npix
    r_units = r_pixels * Rv_per_pixel

    profile, r_centers = [], []
    
    if not silent: print(f'Computing radial profile with bins_frac={bins_frac}...')
    
    for i in range(len(bins_frac)-1):
        mask_ring = (r_units >= bins_frac[i]) & (r_units < bins_frac[i+1]) & (~np.isnan(stack_map))
        if np.any(mask_ring): profile.append(np.mean(stack_map[mask_ring]))
        else: profile.append(np.nan)
        r_centers.append((bins_frac[i] + bins_frac[i+1]) / 2.)

    r_centers = np.array(r_centers)
    profile = np.array(profile)

    return profile, r_centers


def radial_profile_weighted(sum_map, count_map, max_Rvoid, bins_frac, silent=False):
    npix = sum_map.shape[0]; center = npix // 2
    y, x = np.ogrid[-center:npix-center, -center:npix-center]
    r_units = np.sqrt(x*x + y*y) * (2 * max_Rvoid) / npix

    profile, r_centers = [], []
    if not silent: print('Computing count-weighted radial profile...')
    for i in range(len(bins_frac)-1):
        ring = (r_units >= bins_frac[i]) & (r_units < bins_frac[i+1])
        den = np.nansum(count_map[ring])
        profile.append(np.nansum(sum_map[ring]) / den if den > 0 else np.nan)
        r_centers.append((bins_frac[i] + bins_frac[i+1]) / 2.)
    return np.array(profile), np.array(r_centers)


def null_test_rotations(l, b, redshifts, r_voids, cmb_map, cmb_mask, max_Rvoid, npix_stamp, bins_frac, n_rotations, existing_profiles=[]):
    n_old = len(existing_profiles)
    if n_rotations <= n_old:
        return np.array(existing_profiles[:n_rotations])
    
    print(f'Performing {n_rotations - n_old} additional random rotations of the CMB map...')
    null_profiles = list(existing_profiles)

    np.random.seed(42+n_old)
    angles = np.random.uniform(10, 350, n_rotations-n_old)

    for ang in angles:
        rot_cmb = rotate_map(cmb_map, rot_angles=[ang, 0, 0])
        rot_mask = rotate_map(cmb_mask, rot_angles=[ang, 0, 0])
        rot_effective_mask = rot_mask

        stack_null, c = stacking_gnomonic(l, b, redshifts, r_voids, rot_cmb, rot_effective_mask, max_Rvoid, npix_stamp, range(len(l)), silent=True)
        prof_null, _ = radial_profile_weighted(stack_null, c, max_Rvoid, bins_frac, silent=True)
        null_profiles.append(prof_null)

    null_profiles = np.array(null_profiles)
    return null_profiles


def null_test_randoms(nside, redshifts, r_voids, cmb_map, stamp_mask, position_mask, max_Rvoid, npix_stamp, bins_frac, n_random_factor, existing_profiles=[]):
    n_old = len(existing_profiles)
    if n_random_factor <= n_old:
        return np.array(existing_profiles[:n_random_factor])
    
    print(f'Performing {n_random_factor-n_old} random positions per void...')
    null_profiles = list(existing_profiles)
    n_voids = len(redshifts)

    for i in range(n_random_factor - n_old):
        rand_l, rand_b = generate_random(position_mask, n_voids, nside)
        rand_idx = np.random.permutation(n_voids)
        stack_null, c = stacking_gnomonic(rand_l, rand_b, redshifts[rand_idx], r_voids[rand_idx], cmb_map, stamp_mask, max_Rvoid, npix_stamp, range(n_voids), silent=True)
        prof_null, _ = radial_profile_weighted(stack_null, c, max_Rvoid, bins_frac, silent=True)
        null_profiles.append(prof_null)

    null_profiles = np.array(null_profiles)
    return null_profiles


def profiles_with_errors(indices, l, b, redshifts, r_voids, lensing_map, mask, max_Rvoid, npix_stamp, bins_frac, n_subsamples=20):
    ra_rad, dec_rad = np.radians(l[indices]), np.radians(b[indices])
    coords_xyz = np.column_stack([np.cos(dec_rad)*np.cos(ra_rad), np.cos(dec_rad)*np.sin(ra_rad), np.sin(dec_rad)])
    print(f'Dividing {len(indices)} voids into {n_subsamples} jackknife regions (KMeans)...')
    labels = KMeans(n_clusters=n_subsamples, random_state=42, n_init=10).fit_predict(coords_xyz)

    sums, counts = [], []
    for k in range(n_subsamples):
        in_region = (labels == k); idxs_k = indices[in_region]
        if len(idxs_k) == 0:
            sums.append(np.zeros((npix_stamp, npix_stamp))); counts.append(np.zeros((npix_stamp, npix_stamp))); continue
        s_k, c_k = stacking_gnomonic(l[indices][in_region], b[indices][in_region],
                                     redshifts[indices][in_region], r_voids[indices][in_region],
                                     lensing_map, mask, max_Rvoid, npix_stamp, range(len(idxs_k)), silent=True)
        sums.append(s_k); counts.append(c_k)

    sums, counts = np.array(sums), np.array(counts)
    SUM, COUNT = sums.sum(axis=0), counts.sum(axis=0)
    best_prof, _ = radial_profile_weighted(SUM, COUNT, max_Rvoid, bins_frac, silent=True)

    jk_profiles = []
    for k in range(n_subsamples):     
        prof, _ = radial_profile_weighted(SUM - sums[k], COUNT - counts[k], max_Rvoid, bins_frac, silent=True)
        jk_profiles.append(prof)
    jk_profiles = np.array(jk_profiles)

    delta = np.nan_to_num(jk_profiles - best_prof)
    cov_matrix = (n_subsamples - 1) / n_subsamples * np.dot(delta.T, delta)
    return best_prof, np.sqrt(np.diag(cov_matrix)), jk_profiles, cov_matrix


def build_random_exclusion_mask(base_mask, l, b, redshifts, r_voids, excl_factor, nside, silent=False):
    """Return a copy of base_mask with angular disks of radius (excl_factor * Rv) around
    every real void zeroed out, so random null positions cannot land on real voids.
    excl_factor is in units of Rv; with excl_factor<=0 or None the mask is returned unchanged."""
    if excl_factor is None or excl_factor <= 0:
        return base_mask
    excl = base_mask.copy()
    if not silent:
        print(f'Building random-exclusion mask around {len(l)} voids (excl_factor={excl_factor}*Rv)...')
    for i in range(len(l)):
        theta_deg = get_angularsize_comoving(redshifts[i], excl_factor * r_voids[i])
        vec = hp.ang2vec(l[i], b[i], lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(theta_deg))
        excl[pix] = 0.0
    if not silent:
        f_in, f_out = base_mask.mean(), excl.mean()
        print(f'  random pool: {f_out/f_in*100:.1f}% of the base footprint remains after exclusion.')
    return excl


def process_bin_stacking(release, mode, z_min, z_max, r_min, r_max, data_sample_bin, coords_bin, max_Rvoid, npix_stamp, nside, bins_frac, lensing_map, common_mask, stacks_cache_folder, n_random_factor, n_rotations, n_subsamples=20, delta_label='d23_all', filter_label='none', force_rerun=False, random_pool='full', random_excl_factor=1.0):
    z_text = f'{z_min:.2f}_{z_max:.2f}'
    r_text = f'{r_min:.1f}_{r_max:.1f}'
    z_mean, n_voids = data_sample_bin['z'].mean(), len(data_sample_bin)
    l, b, redshifts_all, r_voids_all = coords_bin[0], coords_bin[1], data_sample_bin['z'].values, data_sample_bin['R_void'].values
    
    print(f'Starting stacking for bin with z in [{z_min:.2f}, {z_max:.2f}] containing {n_voids} voids (mean z={z_mean:.3f})...')

    # --- Random null pool ---------------------------------------------------
    # 'full'    -> randoms drawn over the whole CMB-lensing footprint (common_mask)
    # 'survey'  -> randoms restricted to the void survey footprint (legacy behaviour)
    # An exclusion mask removes disks of (random_excl_factor * Rv) around real voids
    # so the random null does not pick up real void signal.
    if random_pool == 'survey':
        random_base_mask = common_mask * footprint_mask(l, b, output_nside=nside)
    else:
        random_base_mask = common_mask
    random_position_mask = build_random_exclusion_mask(
        random_base_mask, l, b, redshifts_all, r_voids_all, random_excl_factor, nside)
    rand_label = f'rpool{random_pool}_excl{(random_excl_factor or 0):.1f}'

    n_rotations = n_rotations if n_rotations is not None else 0
    n_random_factor = n_random_factor if n_random_factor is not None else 0

    signal_cache_file = os.path.join(stacks_cache_folder, f'signal_cache_{release}_{z_text}_{r_text}_N{n_voids}_{delta_label}_{filter_label}_maxRv{max_Rvoid:.1f}_{rand_label}.pkl')

    if not force_rerun and os.path.exists(signal_cache_file):
        with open(signal_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        cached_nrot = cached_data.get('n_rotations_done', 0)
        cached_nrand = cached_data.get('n_randoms_done', 0)

        if cached_nrot >= n_rotations and cached_nrand >= n_random_factor:
            print(f'Loading fully cached signal and null tests from {signal_cache_file}.')
            return cached_data
        else:
            print(f'Signal found in cache, but more null tests requested. Updating...')
            signal_data = cached_data
    else:
        signal_data = None

    null_cache_file = os.path.join(stacks_cache_folder, f'null_tests_{release}_{z_text}_{r_text}_N{n_voids}_{delta_label}_{filter_label}_maxRv{max_Rvoid:.1f}_{rand_label}.npz')

    existing_rot, existing_rand = [], []
    if not force_rerun and os.path.exists(null_cache_file):
        data_cache = np.load(null_cache_file)
        if 'null_profiles_rot' in data_cache: existing_rot = list(data_cache['null_profiles_rot'])
        if 'null_profiles_rand' in data_cache: existing_rand = list(data_cache['null_profiles_rand'])
    
    null_profiles_rot = null_test_rotations(l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, bins_frac, n_rotations, existing_rot) if n_rotations > 0 else np.array([])
    null_profiles_rand = null_test_randoms(nside, redshifts_all, r_voids_all, lensing_map, common_mask, random_position_mask, max_Rvoid, npix_stamp, bins_frac, n_random_factor, existing_rand) if n_random_factor > 0 else np.array([])

    if n_rotations > 0 or n_random_factor > 0:
        np.savez(null_cache_file, null_profiles_rot=null_profiles_rot, null_profiles_rand= null_profiles_rand)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        null_rot_mean = np.nanmean(null_profiles_rot, axis=0) if len(null_profiles_rot) > 0 else np.full(len(bins_frac) -1, np.nan)
        null_rot_std = np.nanstd(null_profiles_rot, axis=0) if len(null_profiles_rot) > 0 else np.full(len(bins_frac) -1, np.nan)
        null_rand_mean = np.nanmean(null_profiles_rand, axis=0) if len(null_profiles_rand) > 0 else np.full(len(bins_frac)-1, np.nan)
        null_rand_std = np.nanstd(null_profiles_rand, axis=0) if len(null_profiles_rand) > 0 else np.full(len(bins_frac)-1, np.nan)

    if signal_data is not None:
        signal_data.update({
            'null_rot_mean': null_rot_mean, 'null_rot_std': null_rot_std,
            'null_rand_mean': null_rand_mean, 'null_rand_std': null_rand_std,
            'n_rotations_done': max(n_rotations, signal_data.get('n_rotations_done', 0)), 'n_randoms_done': max(n_random_factor, signal_data.get('n_randoms_done', 0))
        })
        with open(signal_cache_file, 'wb') as f: pickle.dump(signal_data, f)
        return signal_data
    
    print('Computing signal stack and errors...')
    SUM, COUNT = stacking_gnomonic(l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, range(n_voids))
    signal_map = stack_mean_map(SUM, COUNT)
    prof_total, r_frac = radial_profile_weighted(SUM, COUNT, max_Rvoid, bins_frac)

    jk_profiles, cov_matrix = None, None
    if mode == 'errors':
        _, prof_err, jk_profiles, cov_matrix = profiles_with_errors(np.arange(n_voids), l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, bins_frac, n_subsamples)
    else:
        prof_err = np.zeros_like(prof_total)

    final_result = {
        'z_mean': z_mean, 'map': signal_map, 'r_frac': r_frac, 'profile': prof_total, 
        'error': prof_err, 'jk_profiles': jk_profiles, 'cov_matrix': cov_matrix,
        'null_rot_mean': null_rot_mean, 'null_rot_std': null_rot_std,
        'null_rand_mean': null_rand_mean, 'null_rand_std': null_rand_std,
        'n_voids': n_voids, 'key': z_text, 'n_rotations_done': n_rotations, 'n_randoms_done': n_random_factor
    }

    with open(signal_cache_file, 'wb') as f:
        pickle.dump(final_result, f)

    return final_result

def plot_stacked_maps_and_profiles(data_list, output_path, max_Rvoid):
    n_bins = len(data_list)
    fig = plt.figure(figsize=(6 * n_bins, 12))
    
    gs = gridspec.GridSpec(3, n_bins + 1,
                           height_ratios=[1.2, 1.2, 0.6], 
                           width_ratios=[1] * n_bins + [0.05],
                           hspace=0.35, wspace=0.15)

    all_maps = np.array([d['map'] for d in data_list])
    v_max, v_min = np.percentile(all_maps, 99.5), np.percentile(all_maps, 0.5)
    extent = [-max_Rvoid, max_Rvoid, -max_Rvoid, max_Rvoid]

    # Maps
    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(data['map'], origin='lower', cmap='viridis',
                       extent=extent, vmin=v_min, vmax=v_max)
        n_voids = data.get('n_voids', '?')
        ax.set_title(f"Bin {data.get('key', i+1)}\n"
                     f"z={data['z_mean']:.3f}, N={n_voids}")
        
        ax.tick_params(labelbottom=False) 
        if i == 0:
            ax.set_ylabel(r'$r\,/\,R_v$')
        else:
            ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0, -1])
    plt.colorbar(im, cax=cax, label=r'$\kappa$')

    # Profiles
    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[1, i])

        if 'null_rand_mean' in data and not np.all(np.isnan(data['null_rand_mean'])):
            ax.fill_between(data['r_frac'],
                            (data['null_rand_mean'] - data['null_rand_std']) * 1e3,
                            (data['null_rand_mean'] + data['null_rand_std']) * 1e3,
                            color='xkcd:grey', alpha=0.5, zorder=1, label=r'$1\sigma$ randoms')
            
        if 'null_rot_mean' in data and not np.all(np.isnan(data['null_rot_mean'])):
            ax.fill_between(data['r_frac'],
                            (data['null_rot_mean'] - data['null_rot_std']) * 1e3,
                            (data['null_rot_mean'] + data['null_rot_std']) * 1e3,
                            color='xkcd:salmon', alpha=0.5, zorder=2,
                            label=r'$1\sigma$ rotations')

        ax.axhline(0,   color='k',    linestyle=':',  alpha=0.6, zorder=3)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.8, zorder=3)

        raw_profile = data['profile']
        if 'null_rand_mean' in data and not np.all(np.isnan(data['null_rand_mean'])):
            clean_profile = raw_profile - data['null_rand_mean']
            
            n_rands = data.get('n_randoms_done', 0) 
            
            err_rand_mean = data['null_rand_std'] / np.sqrt(n_rands)
            
            if n_rands > 0:
                net_error = np.sqrt(data['error']**2 + (data['null_rand_std'] / np.sqrt(n_rands))**2)
            else:
                net_error = data['error']
        else:
            clean_profile = raw_profile
            net_error = data['error']

        ax.plot(data['r_frac'], raw_profile * 1e3, '-', 
                color='xkcd:dark grey', alpha=0.8, linewidth=1.5, zorder=3,
                label='Raw Profile')

        ax.errorbar(data['r_frac'], clean_profile * 1e3,
                    yerr=net_error * 1e3,
                    fmt='o-', color='xkcd:steel blue', capsize=3,
                    linewidth=1.8, zorder=4, label='Net Signal')

        ax.set_xlim(-0.1, max_Rvoid + 0.1)
        ax.tick_params(labelbottom=False)
        ax.grid(True, alpha=0.25)

        if i == 0:
            ax.set_ylabel(r'$\kappa\;[10^{-3}]$')
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='lower right', frameon=True, fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    # Significance (S/N)
    for i, data in enumerate(data_list):
        ax_sig = fig.add_subplot(gs[2, i])
        
        raw_profile = data['profile']
        if 'null_rand_mean' in data and not np.all(np.isnan(data['null_rand_mean'])):
            clean_profile = raw_profile - data['null_rand_mean']
            n_rands = data.get('n_randoms_done', 0)
            if n_rands > 0:
                net_error = np.sqrt(data['error']**2 + (data['null_rand_std'] / np.sqrt(n_rands))**2)
            else:
                net_error = data['error']
        else:
            clean_profile = raw_profile
            net_error = data['error']
            
        with np.errstate(divide='ignore', invalid='ignore'):
            significance = clean_profile / net_error
            
        ax_sig.plot(data['r_frac'], significance, 'o-', color='xkcd:crimson', 
                    markersize=5, linewidth=1.5)
        
        ax_sig.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax_sig.axhline(2, color='gray', linestyle='--', alpha=0.8)
        ax_sig.axhline(-2, color='gray', linestyle='--', alpha=0.8)
        
        ax_sig.axhspan(-2, 2, color='gray', alpha=0.1, zorder=0)
        
        ax_sig.set_ylim(-5, 5)
        ax_sig.set_xlim(-0.1, max_Rvoid + 0.1)
        
        ax_sig.set_xlabel(r'$r\,/\,R_v$')
        ax_sig.grid(True, alpha=0.25)

        if i == 0:
            ax_sig.set_ylabel(r'$S/N\;(\sigma)$')
        else:
            ax_sig.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Stacked maps + profiles + significance plot saved to {output_path}")
    plt.close()


def plot_jackknife_and_correlation(bin_results_list, output_path, max_Rvoid):
    n_bins = len(bin_results_list)
    fig, axes = plt.subplots(2, n_bins, figsize=(5.5 * n_bins, 9), gridspec_kw={'hspace': 0.35, 'wspace': 0.2})
    if n_bins == 1: axes = axes[:, np.newaxis]

    for col, data in enumerate(bin_results_list):
        r_frac, profile, error, cov = data['r_frac'], data['profile'], data['error'], data.get('cov_matrix')
        label, z_mean, n_voids, is_ms = data.get('key', f"Bin {col+1}"), data['z_mean'], data.get('n_voids', '?'), data.get('is_multi_seed', False)

        ax_p = axes[0, col]
        ax_p.axhline(0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax_p.errorbar(r_frac, profile * 1e3, yerr=error * 1e3, fmt='o-', color='xkcd:steel blue', capsize=3, linewidth=1.8, label='Signal (JK err.)')
        ax_p.set_xlim(-0.05, max_Rvoid + 0.05)
        ax_p.set_xlabel(r'$r\,/\,R_v$')
        ax_p.set_title(f"Bin {label}  (z={z_mean:.3f}, N={n_voids})")
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc='lower right', frameon=True, fontsize=9)
        if col == 0: ax_p.set_ylabel(r'$\kappa\;[10^{-3}]$')
        else: ax_p.tick_params(labelleft=False)

        ax_c = axes[1, col]
        if cov is not None:
            std = np.sqrt(np.diag(cov))
            with np.errstate(invalid='ignore'): corr = cov / np.outer(std, std)
            corr = np.nan_to_num(corr)
            im = ax_c.imshow(corr, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=[r_frac[0], r_frac[-1], r_frac[0], r_frac[-1]], aspect='auto')
            plt.colorbar(im, ax=ax_c, label='Correlation', fraction=0.046, pad=0.04)
            ax_c.set_xlabel(r'$r\,/\,R_v$')
            ax_c.set_title(f"{'Seed-to-seed corr.' if is_ms else 'JK correlation matrix'} — Bin {label}")
            if col == 0: ax_c.set_ylabel(r'$r\,/\,R_v$')
            else: ax_c.tick_params(labelleft=False)
        else:
            ax_c.text(0.5, 0.5, 'No covariance data', ha='center', va='center', transform=ax_c.transAxes, fontsize=11, color='gray')
            ax_c.set_axis_off()

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_seed_consistency(bin_results_list, output_path, max_Rvoid):
    ms_bins = [d for d in bin_results_list if d.get('is_multi_seed', False) and d.get('seed_results') is not None]
    if len(ms_bins) == 0: return

    n_bins = len(ms_bins)
    fig, axes = plt.subplots(1, n_bins, figsize=(6 * n_bins, 5), sharey=False)
    if n_bins == 1: axes = [axes]

    for col, data in enumerate(ms_bins):
        ax, seed_results, r_frac = axes[col], data['seed_results'], data['r_frac']
        n_seeds = len(seed_results)
        colors = cm.plasma(np.linspace(0.05, 0.85, n_seeds))

        for j, s_res in enumerate(seed_results):
            ax.plot(r_frac, s_res['profile'] * 1e3, color=colors[j], alpha=0.4, linewidth=1.0, label=f'Seed {j+1}' if n_seeds <= 10 else None)

        ax.errorbar(r_frac, data['profile'] * 1e3, yerr=data['error'] * 1e3, fmt='o-', color='black', linewidth=2.0, capsize=3, zorder=5, label='Combined (JK err.)')
        ax.axhline(0, color='k', linestyle=':', alpha=0.4, linewidth=1)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f"Bin {data.get('key', f'Bin {col+1}')}  (z={data['z_mean']:.3f}, N={data.get('n_voids', '?')})\n{n_seeds} seeds")
        ax.set_xlabel(r'$r\,/\,R_v$')
        ax.set_xlim(-0.05, max_Rvoid + 0.05)
        ax.grid(True, alpha=0.25)

        if col == 0: ax.set_ylabel(r'$\kappa\;[10^{-3}]$')
        else: ax.tick_params(labelleft=False)
        ax.legend(loc='lower right', frameon=True, fontsize=8 if n_seeds <= 10 else 9, ncol=2 if n_seeds <= 10 else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_merge_vs_concat(pair_list, output_path, max_Rvoid):
    """
    Compara, por bin, el perfil kappa(r) del catálogo CONCATENADO (multi-seed con
    duplicados) contra el catálogo MERGEADO (DBSCAN, un void por fila).
      Fila superior: kappa(r) de ambos con error JK.
      Fila inferior: diferencia (concat - merge) con error combinado en cuadratura.
    Cada `pair` es {'bin_id', 'key', 'concat': res, 'merge': res}.
    """
    n = len(pair_list)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8),
                             gridspec_kw={'hspace': 0.28, 'wspace': 0.2}, squeeze=False)

    for col, pair in enumerate(pair_list):
        c, m = pair['concat'], pair['merge']
        r = c['r_frac']

        ax_p = axes[0, col]
        ax_p.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        ax_p.errorbar(r, c['profile'] * 1e3, yerr=c['error'] * 1e3, fmt='o-',
                      color='xkcd:steel blue', capsize=3, linewidth=1.8,
                      label=f"Concat (N={c.get('n_voids', '?')})")
        ax_p.errorbar(r, m['profile'] * 1e3, yerr=m['error'] * 1e3, fmt='s-',
                      color='xkcd:crimson', capsize=3, linewidth=1.8,
                      label=f"Merge (N={m.get('n_voids', '?')})")
        ax_p.set_title(f"Bin {pair.get('bin_id', col) + 1}  (z={c.get('z_mean', np.nan):.3f})")
        ax_p.set_xlim(-0.05, max_Rvoid + 0.05)
        ax_p.tick_params(labelbottom=False)
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc='lower right', frameon=True, fontsize=9)
        if col == 0: ax_p.set_ylabel(r'$\kappa\;[10^{-3}]$')
        else: ax_p.tick_params(labelleft=False)

        ax_d = axes[1, col]
        diff = (c['profile'] - m['profile']) * 1e3
        derr = np.sqrt(c['error'] ** 2 + m['error'] ** 2) * 1e3
        ax_d.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax_d.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        ax_d.errorbar(r, diff, yerr=derr, fmt='o-', color='xkcd:dark grey', capsize=3)
        ax_d.set_xlim(-0.05, max_Rvoid + 0.05)
        ax_d.set_xlabel(r'$r\,/\,R_v$')
        ax_d.grid(True, alpha=0.25)
        if col == 0: ax_d.set_ylabel(r'concat $-$ merge $\;[10^{-3}]$')
        else: ax_d.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Merge-vs-concat comparison plot saved to {output_path}")
