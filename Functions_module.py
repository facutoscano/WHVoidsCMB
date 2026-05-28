##### Function module for the CMB lensing voids profiles #####

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def get_filename(release, z_text, max_Rvoid, npix, smooth_value):
    return f'{release}_z{z_text}_maxRv{max_Rvoid:.1f}_npix{npix}_{smooth_value}deg.npz'


def manage_randoms_incremental(filename, n_needed, z_sample_dist, rv_sample_dist, max_Rvoid, nside, mask, lensing_map, npix_stamp):
    if os.path.exists(filename):
        data = np.load(filename)
        stack_mean_old, count_map_old, n_stacked_old = data['stack_mean'], data['count_map'], int(data['n_total_randoms'])
        print(f"Found existing randoms: {n_stacked_old} total. Needed: {n_needed}.")
    else:
        stack_mean_old, count_map_old, n_stacked_old = np.zeros((npix_stamp, npix_stamp)), np.zeros((npix_stamp, npix_stamp)), 0
        print(f"Not previous randoms found. Starting from scratch.")

    n_missing = n_needed - n_stacked_old
    if n_missing > 0:
        print(f"Doing incremental stacking with {n_missing} new randoms...")
        rand_l, rand_b = generate_random(mask, n_random=n_missing, nside=nside)
        
        rand_idx = np.random.choice(len(z_sample_dist), size=n_missing, replace=True)
        rand_z = z_sample_dist[rand_idx]
        rand_rv = rv_sample_dist[rand_idx]
        
        stack_mean_new, count_map_new = stacking_gnomonic(rand_l, rand_b, rand_z, rand_rv, lensing_map, mask, max_Rvoid, npix_stamp, range(n_missing))
        
        sum_old, sum_new = stack_mean_old * count_map_old, stack_mean_new * count_map_new
        total_count_map = count_map_old + count_map_new
        total_stack_mean = np.zeros_like(sum_old)
        valid_pixels = total_count_map > 0
        total_stack_mean[valid_pixels] = (sum_old + sum_new)[valid_pixels] / total_count_map[valid_pixels]
        
        np.savez(filename, stack_mean=total_stack_mean, count_map=total_count_map, n_total_randoms=n_stacked_old + n_missing)
        return total_stack_mean
    
    else:
        print(f"Enough randoms already stacked.")
        return stack_mean_old


def profiles_with_errors(indices, l, b, redshifts, r_voids, lensing_map, mask, max_Rvoid, npix_stamp, stack_rand_mean, bins_frac, n_subsamples=20):
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
        stack_k, _, _ = stacking_gnomonic(l[indices][in_region], b[indices][in_region], redshifts[indices][in_region], r_voids[indices][in_region], lensing_map, mask, max_Rvoid, npix_stamp, range(len(idxs_k)))
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
        prof, _ = radial_profile_flat(stack_loo - stack_rand_mean, max_Rvoid, bins_frac, silent=True)
        jk_profiles.append(prof)
        
    jk_profiles = np.array(jk_profiles)
    prof_mean = np.mean(jk_profiles, axis=0)
    cov_matrix = (len(jk_profiles) - 1) / len(jk_profiles) * np.dot((jk_profiles - prof_mean).T, jk_profiles - prof_mean)
    
    total_stack_w = np.sum(partial_stacks * weights[:, None, None], axis=0) / np.sum(weights)
    best_prof, _ = radial_profile_flat(total_stack_w - stack_rand_mean, max_Rvoid, bins_frac)

    return best_prof, np.sqrt(np.diag(cov_matrix)), jk_profiles


def process_bin_stacking(release, mode, z_min, z_max, data_sample_bin, coords_bin, max_Rvoid, npix_stamp, nside, smooth_value, bins_frac, lensing_map, common_mask, stacks_cache_folder, n_random_factor=10, n_subsamples=20):
    z_text = f'{z_min:.2f}_{z_max:.2f}'
    z_mean, n_voids = data_sample_bin['z'].mean(), len(data_sample_bin)
    l, b, redshifts_all, r_voids_all = coords_bin[0], coords_bin[1], data_sample_bin['z'].values, data_sample_bin['R_void'].values
    
    file_random = stacks_cache_folder + get_filename(release, z_text, max_Rvoid, npix_stamp, smooth_value)
    
    survey_mask = footprint_mask(l, b, output_nside=nside, footprint_nside=32)
    effective_mask = common_mask * survey_mask

    stack_rand_mean = manage_randoms_incremental(file_random, n_voids * n_random_factor, redshifts_all, r_voids_all, max_Rvoid, nside, effective_mask, lensing_map, npix_stamp)
    
    if mode == 'errors':
        prof_mean, prof_err, jk_profiles = profiles_with_errors(np.arange(n_voids), l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, stack_rand_mean, bins_frac, n_subsamples)
        stack_cl, _ = stacking_gnomonic(l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, range(n_voids))
        signal_map = stack_cl - stack_rand_mean
        _, r_frac = radial_profile_flat(signal_map, max_Rvoid, bins_frac)
    else:
        stack_cl, _ = stacking_gnomonic(l, b, redshifts_all, r_voids_all, lensing_map, common_mask, max_Rvoid, npix_stamp, range(n_voids))
        signal_map = stack_cl - stack_rand_mean
        prof_mean, r_frac = radial_profile_flat(signal_map, max_Rvoid, bins_frac)
        prof_err = np.zeros_like(prof_mean)

    return {'z_mean': z_mean, 'map': signal_map, 'r_frac': r_frac, 'profile': prof_mean, 'error': prof_err, 'n_voids': n_voids, 'key': z_text}

def plot_results(data_list, output_path, smooth_value, max_Rvoid):
    n_plots = len(data_list)
    fig = plt.figure(figsize=(5 * n_plots, 8))
    gs = gridspec.GridSpec(2, n_plots + 1, width_ratios=[1]*n_plots + [0.05], hspace=0.2, wspace=0.15)

    all_maps = np.array([d['map'] for d in data_list])
    v_max, v_min = np.percentile(all_maps, 99.9), np.percentile(all_maps, 0.1)
    extent = [-max_Rvoid, max_Rvoid, -max_Rvoid, max_Rvoid]

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[0, i])
        map_to_plot = gaussian_filter(data['map'], sigma=smooth_value) if smooth_value > 0 else data['map']
        im = ax.imshow(map_to_plot, origin='lower', cmap='viridis', extent=extent, vmin=v_min, vmax=v_max)
        t = f"Bin {data.get('key', 'Comb')} (z={data['z_mean']:.3f})"
        ax.set_title(t)
        if i == 0: ax.set_ylabel(r"$r / R_v")
        else: ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0, -1])
    plt.colorbar(im, cax=cax, label=r'$\kappa$')

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[1, i])
        ax.errorbar(data['r_frac'], data['profile'], yerr=data['error'], fmt='o-', color='xkcd:blue violet', capsize=3)
        ax.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.8)

        ax.set_xlabel("Radius [$r / R_v$]")
        ax.set_xlim(0, max_Rvoid + 0.5)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel(r'$\kappa$')
        else: ax.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    print('')
    plt.close()
