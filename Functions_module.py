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
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

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


def footprint_mask(l, b, output_nside, footprint_nside=16): 
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


def stacking_gnomonic(l, b, redshifts, r_voids, cmb_map, mask, max_Rvoid, npix_stamp, stacked_range):
    print(f'Stacking {len(stacked_range)} maps using scaled Rv...')
    stacked_map = np.zeros((npix_stamp, npix_stamp))
    count_map = np.zeros((npix_stamp, npix_stamp))
    nside = hp.npix2nside(len(cmb_map))
    vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    for i, idx in enumerate(stacked_range):
        cl_l, cl_b, cl_z, cl_rv = l[idx], b[idx], redshifts[idx], r_voids[idx]

        box_size_mpch = 2 * max_Rvoid * cl_rv 
        box_size_deg = get_angularsize_comoving(cl_z, box_size_mpch)
        reso_arcmin = (box_size_deg * 60.) / npix_stamp 

        proj = hp.projector.GnomonicProj(rot=[cl_l, cl_b, 0], xsize=npix_stamp, ysize=npix_stamp, reso=reso_arcmin)

        stamp_data = proj.projmap(cmb_map, vec2pix_func=vec2pix_func)
        stamp_mask = proj.projmap(mask, vec2pix_func=vec2pix_func)
        valid_pixels = (stamp_mask > 0.9) & (~np.isnan(stamp_data))

        stacked_map[valid_pixels] += stamp_data[valid_pixels]
        count_map[valid_pixels] += 1
        if (i+1) % 150 == 0: print(f'Stacked {i+1} / {len(stacked_range)}')
    
    final_stack = np.zeros_like(stacked_map)
    mask_final = count_map > 0
    final_stack[mask_final] = stacked_map[mask_final] / count_map[mask_final]

    return final_stack, count_map


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

def null_test_rotations(l, b, redshifts, r_voids, cmb_map, mask, max_Rvoid, npix_stamp, bins_frac, n_rotations):
    print(f'Performing null test with {n_rotations} random rotations of the CMB map...')
    null_profiles = []
    angles = np.linspace(360/n_rotations, 360, n_rotations, endpoint=False)

    for i, ang in enumerate(angles):
        rot_cmb = rotate_map(cmb_map, rot_angles=[ang, 0, 0])
        rot_mask = rotate_map(mask, rot_angles=[ang, 0, 0])

        stack_null, _ = stacking_gnomonic(l, b, redshifts, r_voids, rot_cmb, rot_mask, max_Rvoid, npix_stamp, range(len(l)))
        prof_null, _ = radial_profile_flat(stack_null, max_Rvoid, bins_frac, silent=True)
        null_profiles.append(prof_null)

    null_profiles = np.array(null_profiles)
    return np.nanmean(null_profiles, axis=0), np.nanstd(null_profiles, axis=0)

def null_test_randoms(nside, redshifts, r_voids, cmb_map, mask, max_Rvoid, npix_stamp, bins_frac, n_random_factor):
    print(f'Performing null test with {n_random_factor} random positions per void...')
    null_profiles = []
    n_voids = len(redshifts)

    for i in range(n_random_factor):
        rand_l, rand_b = generate_random(mask, n_voids, nside)
        rand_idx = np.random.permutation(n_voids)

        stack_null, _ = stacking_gnomonic(rand_l, rand_b, redshifts[rand_idx], r_voids[rand_idx], cmb_map, mask, max_Rvoid, npix_stamp, range(n_voids))
        prof_null, _ = radial_profile_flat(stack_null, max_Rvoid, bins_frac, silent=True)
        null_profiles.append(prof_null)

    null_profiles = np.array(null_profiles)
    return np.nanmean(null_profiles, axis=0), np.nanstd(null_profiles, axis=0)


def profiles_with_errors(indices, l, b, redshifts, r_voids, lensing_map, mask, max_Rvoid, npix_stamp, bins_frac, n_subsamples=20):
    ra_rad, dec_rad = np.radians(l[indices]), np.radians(b[indices])
    coords_xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])

    print(f'Dividing the {len(indices)} voids into {n_subsamples} jackknife subsamples using KMeans clustering...')
    labels = KMeans(n_clusters=n_subsamples, random_state=42, n_init=10).fit_predict(coords_xyz)
    
    partial_stacks, weights = [], []
    for k in range(n_subsamples):
        in_region = (labels == k)
        idxs_k = indices[in_region]
        if len(idxs_k) == 0:
            weights.append(0)
            partial_stacks.append(np.zeros((npix_stamp, npix_stamp)))
            continue
        stack_k, _ = stacking_gnomonic(l[indices][in_region], b[indices][in_region], redshifts[indices][in_region], r_voids[indices][in_region], lensing_map, mask, max_Rvoid, npix_stamp, range(len(idxs_k)))
        partial_stacks.append(stack_k)
        weights.append(len(idxs_k))
        
    partial_stacks, weights = np.array(partial_stacks), np.array(weights)
    jk_profiles = []

    for k in range(n_subsamples):
        mask_loo = np.arange(n_subsamples) != k
        valid_w = weights[mask_loo]
        if np.sum(valid_w) == 0: 
            jk_profiles.append(np.zeros(len(bins_frac)-1))
            continue

        stack_loo = np.sum(partial_stacks[mask_loo] * valid_w[:, None, None], axis=0) / np.sum(valid_w)
        prof, _ = radial_profile_flat(stack_loo, max_Rvoid, bins_frac, silent=True)
        jk_profiles.append(prof)
        
    jk_profiles = np.array(jk_profiles)
    prof_mean = np.mean(jk_profiles, axis=0)
    cov_matrix = (len(jk_profiles) - 1) / len(jk_profiles) * np.dot((jk_profiles - prof_mean).T, jk_profiles - prof_mean)
    
    total_stack_w = np.sum(partial_stacks * weights[:, None, None], axis=0) / np.sum(weights)
    best_prof, _ = radial_profile_flat(total_stack_w, max_Rvoid, bins_frac)

    return best_prof, np.sqrt(np.diag(cov_matrix)), jk_profiles


def process_bin_stacking(release, mode, z_min, z_max, data_sample_bin, coords_bin, max_Rvoid, npix_stamp, nside, smooth_value, bins_frac, lensing_map, common_mask, stacks_cache_folder, n_random_factor, n_rotations, n_subsamples=20):
    z_text = f'{z_min:.2f}_{z_max:.2f}'
    z_mean, n_voids = data_sample_bin['z'].mean(), len(data_sample_bin)
    l, b, redshifts_all, r_voids_all = coords_bin[0], coords_bin[1], data_sample_bin['z'].values, data_sample_bin['R_void'].values
    
    print(f'Starting stacking for bin with z in [{z_min:.2f}, {z_max:.2f}] containing {n_voids} voids (mean z={z_mean:.3f})...')  
    survey_mask = footprint_mask(l, b, output_nside=nside, footprint_nside=16)
    effective_mask = common_mask * survey_mask

    # Null tests
    cache_file = os.path.join(stacks_cache_folder, f'null_tests_{release}_{z_text}_maxRv{max_Rvoid:.1f}_{smooth_value:.2f}deg_nrand{n_random_factor}_nrot{n_rotations}.npz')
    if os.path.exists(cache_file):
        print(f'Loading null test results from cache: {cache_file}')
        data_cache = np.load(cache_file)
        null_rot_mean, null_rot_std = data_cache['null_rot_mean'], data_cache['null_rot_std']
        null_rand_mean, null_rand_std = data_cache['null_rand_mean'], data_cache['null_rand_std']
    else:
        null_rot_mean, null_rot_std = null_test_rotations(l, b, redshifts_all, r_voids_all, lensing_map, effective_mask, max_Rvoid, npix_stamp, bins_frac, n_rotations)
        null_rand_mean, null_rand_std = null_test_randoms(nside, redshifts_all, r_voids_all, lensing_map, effective_mask, max_Rvoid, npix_stamp, bins_frac, n_random_factor)
        np.savez(cache_file, null_rot_mean=null_rot_mean, null_rot_std=null_rot_std, null_rand_mean=null_rand_mean, null_rand_std=null_rand_std)
        print(f'Null test results saved to cache: {cache_file}')

    # Signal
    print('Computing mean random stack for signal estimation...')
    stack_cl, _ = stacking_gnomonic(l, b, redshifts_all, r_voids_all, lensing_map, effective_mask, max_Rvoid, npix_stamp, range(n_voids))
    signal_map = stack_cl

    if mode == 'errors':
        prof_mean, prof_err, _ = profiles_with_errors(np.arange(n_voids), l, b, redshifts_all, r_voids_all, lensing_map, effective_mask, max_Rvoid, npix_stamp, bins_frac, n_subsamples)
        _, r_frac = radial_profile_flat(signal_map, max_Rvoid, bins_frac)
    else:
        prof_mean, r_frac = radial_profile_flat(signal_map, max_Rvoid, bins_frac)
        prof_err = np.zeros_like(prof_mean)

    return {
        'z_mean': z_mean, 
        'map': signal_map, 
        'r_frac': r_frac, 
        'profile': prof_mean, 
        'error': prof_err, 
        'null_rot_mean': null_rot_mean,
        'null_rot_std': null_rot_std,
        'null_rand_mean': null_rand_mean,
        'null_rand_std': null_rand_std,
        'n_voids': n_voids, 
        'key': z_text
        }


def plot_results(data_list, output_path, smooth_value, max_Rvoid):
    n_plots = len(data_list)
    fig = plt.figure(figsize=(6 * n_plots, 9))
    gs = gridspec.GridSpec(2, n_plots + 1, width_ratios=[1]*n_plots + [0.05], hspace=0.25, wspace=0.15)

    all_maps = np.array([d['map'] for d in data_list])
    v_max, v_min = np.percentile(all_maps, 99.5), np.percentile(all_maps, 0.5)
    extent = [-max_Rvoid, max_Rvoid, -max_Rvoid, max_Rvoid]

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[0, i])
        map_to_plot = gaussian_filter(data['map'], sigma=smooth_value) if smooth_value > 0 else data['map']
        im = ax.imshow(map_to_plot, origin='lower', cmap='viridis', extent=extent, vmin=v_min, vmax=v_max)
        t = f"Bin {data.get('key', 'Comb')} (z={data['z_mean']:.3f})"
        ax.set_title(t)
        if i == 0: ax.set_ylabel(r"$r / R_v$")
        else: ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0, -1])
    plt.colorbar(im, cax=cax, label=r'$\kappa$')

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[1, i])

        ax.fill_between(data['r_frac'], data['null_rand_mean']*1e3 - data['null_rand_std']*1e3, data['null_rand_mean']*1e3 + data['null_rand_std']*1e3, color='xkcd:grey', alpha=0.5, zorder=1, label=r'$1\sigma$ randoms')

        ax.fill_between(data['r_frac'], data['null_rot_mean']*1e3 - data['null_rot_std']*1e3, data['null_rot_mean']*1e3 + data['null_rot_std']*1e3, color='xkcd:salmon', alpha=0.5, zorder=2, label=r'$1\sigma$ rotations')

        ax.axhline(0, color='k', linestyle=':', alpha=0.6, zorder=3)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.8, zorder=3)

        if data.get('is_multi_seed', False):
            seed_results = data['seed_results']
            
            colors = cm.plasma(np.linspace(0, 0.8, len(seed_results)))
            
            for j, s_res in enumerate(seed_results):
                label_seed = 'Seeds profiles' if j == 0 else None
                ax.plot(data['r_frac'], s_res['profile']*1e3, color=colors[j], alpha=0.3, linewidth=1, zorder=4, label=label_seed)
            
            mean_profile = np.mean([s['profile'] for s in seed_results], axis=0)
            mean_error = np.mean([s['error'] for s in seed_results], axis=0)
            
            ax.errorbar(data['r_frac'], mean_profile*1e3, yerr=mean_error*1e3, fmt='o-', color='xkcd:red', capsize=3, label='Mean Signal', zorder=5, linewidth=2)
            
        else:
            ax.errorbar(data['r_frac'], data['profile']*1e3, yerr=data['error']*1e3, fmt='o-', color='xkcd:steel blue', capsize=3, label='Signal', zorder=4)
            
        ax.set_xlabel(r"Radius [$r / R_v$]")
        ax.set_xlim(-0.1, max_Rvoid + 0.1)
        ax.grid(True, alpha=0.3)
        if i == 0: 
            ax.set_ylabel(r'$\kappa [10^{-3}]$')
            ax.legend(loc='lower right', frameon=True, fontsize=10)
        else: 
            ax.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    print('')
    plt.close()
