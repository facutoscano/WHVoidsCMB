##### Function module for the CMB lensing profiles analysis #####

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
def get_angularsize(z, physical_size_Mpc):
    d_a = Planck18.angular_diameter_distance(z).to(u.Mpc).value
    theta_rad = physical_size_Mpc / d_a
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def stacking_gnomonic(lon, lat, redshifts, cmb_map, mask, physical_size_Mpc, npix_stamp, stacked_range):
    print(f'Stacking {len(stacked_range)} maps using PHYSICAL scaling (per-object z)...')
    stacked_map = np.zeros((npix_stamp, npix_stamp))
    count_map = np.zeros((npix_stamp, npix_stamp))
    nside = hp.npix2nside(len(cmb_map))
    vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    for i, idx in enumerate(stacked_range):
        cl_lon, cl_lat, cl_z = lon[idx], lat[idx], redshifts[idx]
        box_size_deg = get_angularsize(cl_z, physical_size_Mpc)
        reso_arcmin = (box_size_deg * 60.) / npix_stamp 
        proj = hp.projector.GnomonicProj(rot=[cl_lon, cl_lat, 0], xsize=npix_stamp, ysize=npix_stamp, reso=reso_arcmin)

        stamp_data = proj.projmap(cmb_map, vec2pix_func=vec2pix_func)
        stamp_mask = proj.projmap(mask, vec2pix_func=vec2pix_func)
        valid_pixels = (stamp_mask > 0.9) & (~np.isnan(stamp_data))

        stacked_map[valid_pixels] += stamp_data[valid_pixels]
        count_map[valid_pixels] += 1
        if (i+1) % 500 == 0: print(f'   -> Stacked {i+1} / {len(stacked_range)}')
    
    final_stack = np.zeros_like(stacked_map)
    mask_final = count_map > 0
    final_stack[mask_final] = stacked_map[mask_final] / count_map[mask_final]
    avg_box = get_angularsize(np.mean(redshifts), physical_size_Mpc)
    extent = [-avg_box/2, avg_box/2, -avg_box/2, avg_box/2]
    return final_stack, count_map, extent

def radial_profile_flat(stack_map, box_size_deg_dummy, n_bins, z_mean, physical_size_Mpc=None, silent=False):
    npix = stack_map.shape[0]
    center = npix // 2
    y, x = np.ogrid[-center:npix-center, -center:npix-center]
    r_pixels = np.sqrt(x*x + y*y)

    if physical_size_Mpc is not None:
        mpc_per_pixel = physical_size_Mpc / npix
        r_units = r_pixels * mpc_per_pixel
        max_r = physical_size_Mpc / 2.0
    else: 
        deg_per_pixel = box_size_deg_dummy / npix
        r_units = r_pixels * deg_per_pixel
        max_r = box_size_deg_dummy / 2.0

    bins = np.linspace(0, max_r, n_bins + 1)
    profile, r_centers = [], []

    if not silent: print(f'Computing radial profile...')

    for i in range(n_bins):
        mask_ring = (r_units >= bins[i]) & (r_units < bins[i+1]) & (~np.isnan(stack_map))
        if np.any(mask_ring): profile.append(np.mean(stack_map[mask_ring]))
        else: profile.append(np.nan)
        r_centers.append((bins[i] + bins[i+1]) / 2.)
    
    r_centers = np.array(r_centers)
    profile = np.array(profile)

    if physical_size_Mpc is not None: R_Mpc = r_centers 
    else: R_Mpc = np.radians(r_centers) * Planck18.angular_diameter_distance(z_mean).value

    return profile, r_centers, R_Mpc

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

def get_filename(release, type_str, z_text, l_min, l_max, size_mpc, npix, smooth_value):
    return f'{release}_{type_str}_z{z_text}_lambda{l_min}-{l_max}_box{size_mpc:.1f}Mpc_npix_{npix}_{smooth_value}deg.npz'

def manage_randoms_incremental(filename, n_needed, z_sample_dist, physical_size_Mpc, nside, mask, lensing_map, npix_stamp, fm_module):
    if os.path.exists(filename):
        data = np.load(filename)
        stack_mean_old, count_map_old, n_stacked_old = data['stack_mean'], data['count_map'], int(data['n_total_randoms'])
        print(f"   -> Randoms encontrados: {n_stacked_old} acumulados.")
    else:
        stack_mean_old, count_map_old, n_stacked_old = np.zeros((npix_stamp, npix_stamp)), np.zeros((npix_stamp, npix_stamp)), 0
        print(f"   -> No hay randoms previos. Se iniciará de cero.")

    n_missing = n_needed - n_stacked_old
    if n_missing > 0:
        print(f"   -> Generando {n_missing} randoms adicionales (muestreando Z)...")
        rand_lon, rand_lat = fm_module.generate_random(mask, n_random=n_missing, nside=nside)
        rand_z = np.random.choice(z_sample_dist, size=n_missing, replace=True)
        stack_mean_new, count_map_new, _ = stacking_gnomonic(rand_lon, rand_lat, rand_z, lensing_map, mask, physical_size_Mpc, npix_stamp, range(n_missing))
        
        sum_old, sum_new = stack_mean_old * count_map_old, stack_mean_new * count_map_new
        total_count_map = count_map_old + count_map_new
        total_stack_mean = np.zeros_like(sum_old)
        valid_pixels = total_count_map > 0
        total_stack_mean[valid_pixels] = (sum_old + sum_new)[valid_pixels] / total_count_map[valid_pixels]
        
        np.savez(filename, stack_mean=total_stack_mean, count_map=total_count_map, n_total_randoms=n_stacked_old + n_missing)
        return total_stack_mean
    else:
        print(f"   -> Suficientes randoms guardados.")
        return stack_mean_old

def profiles_with_errors(indices, lon, lat, redshifts, lensing_map, mask, physical_size_Mpc, npix_stamp, stack_rand_mean, bins_physical, z_mean, n_subsamples=20):
    ra_rad, dec_rad = np.radians(lon[indices]), np.radians(lat[indices])
    coords_xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])
    print(f'Dividiendo catálogo en {n_subsamples} regiones para Jackknife...')
    labels = KMeans(n_clusters=n_subsamples, random_state=42, n_init=10).fit_predict(coords_xyz)
    
    partial_stacks, weights = [], []
    for k in range(n_subsamples):
        in_region = (labels == k)
        idxs_k = indices[in_region]
        if len(idxs_k) == 0:
            weights.append(0)
            partial_stacks.append(np.zeros((npix_stamp, npix_stamp)))
            continue
        stack_k, _, _ = stacking_gnomonic(lon[indices][in_region], lat[indices][in_region], redshifts[indices][in_region], lensing_map, mask, physical_size_Mpc, npix_stamp, range(len(idxs_k)))
        partial_stacks.append(stack_k)
        weights.append(len(idxs_k))
        
    partial_stacks, weights = np.array(partial_stacks), np.array(weights)
    jk_profiles = []
    for k in range(n_subsamples):
        mask_loo = np.arange(n_subsamples) != k
        valid_w = weights[mask_loo]
        if np.sum(valid_w) == 0: 
            jk_profiles.append(np.zeros(len(bins_physical)))
            continue
        stack_loo = np.sum(partial_stacks[mask_loo] * valid_w[:, None, None], axis=0) / np.sum(valid_w)
        prof, _, _ = radial_profile_flat(stack_loo - stack_rand_mean, None, bins_physical, z_mean, physical_size_Mpc=physical_size_Mpc, silent=True)
        jk_profiles.append(prof)
        
    jk_profiles = np.array(jk_profiles)
    prof_mean = np.mean(jk_profiles, axis=0)
    cov_matrix = (len(jk_profiles) - 1) / len(jk_profiles) * np.dot((jk_profiles - prof_mean).T, jk_profiles - prof_mean)
    
    total_stack_w = np.sum(partial_stacks * weights[:, None, None], axis=0) / np.sum(weights)
    best_prof, _, _ = radial_profile_flat(total_stack_w - stack_rand_mean, None, bins_physical, z_mean, physical_size_Mpc=physical_size_Mpc)
    return best_prof, np.sqrt(np.diag(cov_matrix)), jk_profiles

def process_bin_stacking(release, mode, z_min, z_max, data_sample_bin, coords_bin, physical_size_Mpc, npix_stamp, nside, smooth_value, reso, bins_physical, lensing_map, common_mask, stacks_cache_folder, lambda_min, lambda_max, N_RAND_FACTOR, fm_module, n_subsamples=20):
    z_text = f'{z_min:.2f}_{z_max:.2f}'
    z_mean, n_clusters = data_sample_bin['zCl'].mean(), len(data_sample_bin)
    lon, lat, redshifts_all = coords_bin[0], coords_bin[1], data_sample_bin['zCl'].values
    
    #print(f"\n--- Procesando Bin {z_text} (N={n_clusters}) | Mode: {mode} ---")
    file_random = stacks_cache_folder + get_filename(release, "Random", z_text, lambda_min, lambda_max, physical_size_Mpc, npix_stamp, smooth_value)
    
    survey_mask = footprint_mask(lon, lat, output_nside=nside, footprint_nside=16)
    effective_mask = common_mask * survey_mask
    stack_rand_mean = manage_randoms_incremental(file_random, n_clusters * N_RAND_FACTOR, redshifts_all, physical_size_Mpc, nside, effective_mask, lensing_map, npix_stamp, fm_module)
    
    if mode == 'errors':
        prof_mean, prof_err, jk_profiles = profiles_with_errors(np.arange(n_clusters), lon, lat, redshifts_all, lensing_map, common_mask, physical_size_Mpc, npix_stamp, stack_rand_mean, bins_physical, z_mean, n_subsamples)
        stack_cl, _, _ = stacking_gnomonic(lon, lat, redshifts_all, lensing_map, common_mask, physical_size_Mpc, npix_stamp, range(n_clusters))
        signal_map = stack_cl - stack_rand_mean
        _, _, r_mpc = radial_profile_flat(signal_map, None, bins_physical, z_mean, physical_size_Mpc=physical_size_Mpc)
    else:
        stack_cl, _, _ = stacking_gnomonic(lon, lat, redshifts_all, lensing_map, common_mask, physical_size_Mpc, npix_stamp, range(n_clusters))
        signal_map = stack_cl - stack_rand_mean
        prof_mean, _, r_mpc = radial_profile_flat(signal_map, None, bins_physical, z_mean, physical_size_Mpc=physical_size_Mpc)
        prof_err = np.zeros_like(prof_mean)

    return {'z_mean': z_mean, 'map': signal_map, 'r_mpc': r_mpc, 'profile': prof_mean, 'error': prof_err, 'n_clusters': n_clusters, 'key': z_text}

def plot_results(data_list, title_prefix, output_path, smooth_value, physical_size_Mpc, type_str):
    n_plots = len(data_list)
    fig = plt.figure(figsize=(5 * n_plots, 8))
    gs = gridspec.GridSpec(2, n_plots + 1, width_ratios=[1]*n_plots + [0.05], hspace=0.2, wspace=0.15)
    all_maps = np.array([d['map'] for d in data_list])
    v_max, v_min = np.percentile(all_maps, 99.9), np.percentile(all_maps, 0.1)
    extent = [-physical_size_Mpc/2, physical_size_Mpc/2, -physical_size_Mpc/2, physical_size_Mpc/2]

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[0, i])
        map_to_plot = gaussian_filter(data['map'], sigma=smooth_value) if smooth_value > 0 else data['map']
        im = ax.imshow(map_to_plot, origin='lower', cmap='viridis', extent=extent, vmin=v_min, vmax=v_max)
        t = f"Bin {data.get('key','Comb')} (z={data['z_mean']:.3f})"
        ax.set_title(t)
        if i == 0: ax.set_ylabel("Mpc")
        else: ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0, -1])
    plt.colorbar(im, cax=cax, label=r'$\kappa$' if type_str == 'kappa' else r'$\Delta y$')

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[1, i])
        ax.errorbar(data['r_mpc'], data['profile'], yerr=data['error'], fmt='o-', color='xkcd:blue violet', capsize=3)
        ax.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax.set_xlabel("Radius [Mpc]")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel(r'$\kappa$' if type_str == 'kappa' else r'$\Delta y$')
        else: ax.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot guardado en: {output_path}")

def plot_tsz_comparison(bin_results_list, output_path, smooth_value, physical_size_Mpc, execution_mode='errors', tsz_label=""):
    n_plots = len(bin_results_list)
    fig = plt.figure(figsize=(5 * n_plots, 9)) 
    
    if tsz_label:
        fig.suptitle(f"tSZ Analysis: {tsz_label}", fontsize=16, y=0.96)
    
    gs = gridspec.GridSpec(2, n_plots, hspace=0.15, wspace=0.1, top=0.92) 
    
    all_y_profs = [d['y_profile'] for d in bin_results_list]
    all_y_errs = [d['y_error'] for d in bin_results_list]
    all_free_profs = [d['free_profile'] for d in bin_results_list]
    all_free_errs = [d['free_error'] for d in bin_results_list]
    
    y_min = min([(p - e).min() for p, e in zip(all_y_profs, all_y_errs)])
    y_max = max([(p + e).max() for p, e in zip(all_y_profs, all_y_errs)])
    dy = y_max - y_min
    ylim_y = (y_min - 0.1*dy, y_max + 0.1*dy)

    f_min = min([(p - e).min() for p, e in zip(all_free_profs, all_free_errs)])
    f_max = max([(p + e).max() for p, e in zip(all_free_profs, all_free_errs)])
    df = f_max - f_min
    ylim_free = (f_min - 0.1*df, f_max + 0.1*df)

    for i, data in enumerate(bin_results_list):
        r_mpc = data['r_mpc']
        
        # --- Compton Y ---
        ax1 = fig.add_subplot(gs[0, i])
        ax1.errorbar(r_mpc, data['y_profile'], yerr=data['y_error'], fmt='o-', color='xkcd:blue violet', capsize=3, label='Compton-Y')
        ax1.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(ylim_y)
        ax1.tick_params(labelbottom=False)
        
        param_val = data['lambda_mean']
        param_name = r"$\bar{\lambda}$"
        title_str = f"Bin {data['bin_id']} ({param_name}={param_val:.1f})"
        ax1.set_title(title_str, fontsize=12)
        
        if i == 0: ax1.set_ylabel(r'$\Delta y$ (Compton)')
        else: ax1.tick_params(labelleft=False)

        # --- SZ Free ---
        ax2 = fig.add_subplot(gs[1, i])
        ax2.errorbar(r_mpc, data['free_profile'], yerr=data['free_error'], fmt='o-', color='xkcd:orange red', capsize=3, label='SZ-Free')
        ax2.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(ylim_free) 
        
        ax2.set_xlabel("Radius [Mpc]")
        
        if i == 0: ax2.set_ylabel(r'$\Delta T$ (SZ-Free)')
        else: ax2.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot comparativo guardado en: {output_path}")