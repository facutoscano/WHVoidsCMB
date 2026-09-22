##### Function module for the CMB lensing voids profiles #####

#%% Imports
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

#%% Auxiliary Functions
def get_angularsize_comoving(z, size_mpch):
    h = Planck18.h
    d_c = Planck18.comoving_distance(z).value * h
    theta_rad = size_mpch / d_c
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def apply_wiener_filter(cmb_alm, nlkk_file, lmax=2048):
    nlkk_data = np.loadtxt(nlkk_file)
    ell_nlkk = nlkk_data[:, 0].astype(int)
    nl_kk = nlkk_data[:, 1]             # noise
    sn_kk = nlkk_data[:, 2]             # signal + noise

    cl_kk = sn_kk - nl_kk
    cl_kk = np.maximum(cl_kk, 0)        # avoiding negative Cls

    W = np.zeros(lmax + 1)
    for i, ell in enumerate(ell_nlkk):
        if ell > lmax:
            break
        denom = cl_kk[i] + nl_kk[i]
        W[ell] = cl_kk[i] / denom if denom > 0 else 0.0
    
    alm_filtered = hp.almxfl(cmb_alm.copy(), W)

    print(f'[Data] Wiener filter applied.')
    return alm_filtered, W

def footprint_mask(l, b, output_nside, footprint_nside=32): 
    npix_footprint = hp.nside2npix(footprint_nside)
    footprint_mask = np.zeros(npix_footprint, dtype=np.float32)
    pix_indices = hp.ang2pix(footprint_nside, l, b, lonlat=True)
    footprint_mask[pix_indices] = 1.0
    output_mask = hp.ud_grade(footprint_mask, nside_out=output_nside)
    output_mask[output_mask > 0] = 1.0
    return output_mask

def to_map_frame(l_gal, b_gal, frame):
    """
    (l,b) galactic [deg]
    'galactic' -> same |
    'equatorial'/'icrs' -> (ra,dec)
    """
    if frame in ('galactic', 'gal'):
        return np.asarray(l_gal, float), np.asarray(b_gal, float)
    if frame in ('equatorial', 'icrs', 'celestial'):
        c = SkyCoord(l=np.asarray(l_gal) * u.degree,
                     b=np.asarray(b_gal) * u.degree, frame='galactic').icrs
        return c.ra.degree, c.dec.degree
    raise ValueError(f'unknown frame: {frame}')

def footprint_coverage(l_gal, b_gal, z, r_void, mask, nside, frame,
                       max_Rvoid, cov_radius_frac=None):
    """
    Coverage = Void fraction within the footprint
    """
    lon, lat = to_map_frame(l_gal, b_gal, frame)
    rad = max_Rvoid if cov_radius_frac is None else cov_radius_frac
    cov = np.empty(len(lon))
    for i in range(len(lon)):
        theta_deg = get_angularsize_comoving(z[i], rad * r_void[i])
        vec = hp.ang2vec(lon[i], lat[i], lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(theta_deg))
        cov[i] = mask[pix].mean() if len(pix) else 0.0
    return cov

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
    if not silent: print(f'[Profiles] Stacking {len(stacked_range)} maps...')
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
    
    if not silent: print(f'[Profiles] Computing radial profile with bins_frac={bins_frac}...')
    
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
    if not silent: print('[Profiles] Computing count-weighted radial profile...')
    for i in range(len(bins_frac)-1):
        ring = (r_units >= bins_frac[i]) & (r_units < bins_frac[i+1])
        den = np.nansum(count_map[ring])
        profile.append(np.nansum(sum_map[ring]) / den if den > 0 else np.nan)
        r_centers.append((bins_frac[i] + bins_frac[i+1]) / 2.)
    return np.array(profile), np.array(r_centers)

def sample_covariance(profiles, kind='jk'):
    if len(profiles) < 2 or profiles.ndim != 2:
        return None, None
    else:
        good = np.isfinite(profiles).all(axis=0)
        p = profiles.shape[1]
        C = np.full((p, p), np.nan)
        if kind=='jk':
            delta = profiles[:,good] - profiles[:,good].mean(axis=0)
            n = profiles.shape[0]
            cov_matrix = (n - 1) / n * np.dot(delta.T, delta)
            C[np.ix_(good, good)] = cov_matrix
        elif kind == 'cmb':
            cov_matrix = np.cov(profiles[:,good], rowvar=False)
            C[np.ix_(good, good)] = cov_matrix
        else:
            raise ValueError(f'Unknown kind: {kind}')
        err = np.sqrt(np.diag(C))
    return C, err

def build_random_exclusion_mask(base_mask, l, b, redshifts, r_voids, excl_factor, nside, silent=False):
    """
    Return a copy of base_mask with angular disks of radius (excl_factor * Rv) around
    every real void zeroed out, so random null positions cannot land on real voids.
    excl_factor is in units of Rv; with excl_factor<=0 or None the mask is returned unchanged
    """
    if excl_factor is None or excl_factor <= 0:
        return base_mask
    excl = base_mask.copy()
    if not silent:
        print(f'[Mask] Building random-exclusion mask around {len(l)} voids (excl_factor={excl_factor}*Rv)...')
    for i in range(len(l)):
        theta_deg = get_angularsize_comoving(redshifts[i], excl_factor * r_voids[i])
        vec = hp.ang2vec(l[i], b[i], lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(theta_deg))
        excl[pix] = 0.0
    if not silent:
        f_in, f_out = base_mask.mean(), excl.mean()
        print(f'[Mask] random pool: {f_out/f_in*100:.1f}% of the base footprint remains after exclusion.')
    return excl

def build_rvoid_summary(bin_results, print_table=True):
    """
    Summary of the sample according median radius of the stacked voids
    """
    def _kind(res):
        return res.get('catalog','single')

    def _entries(results):
        out = []
        for i, res in enumerate(results or []):
            entry = {
                'bin_id': int(res.get('bin_id', i)),
                'catalog': _kind(res),
                'z_range': res.get('key'),
                'n_voids': int(res.get('n_voids', 0)),
                'R_void_median': float(res.get('R_void_median', np.nan)),
            }
            r_seeds = [float(s['R_void_median']) for s in res.get('seed_results', [])
                       if 'R_void_median' in s]
            if r_seeds:
                entry['R_void_median_seeds_mean'] = float(np.mean(r_seeds))
                entry['R_void_median_seeds_std'] = float(np.std(r_seeds))
            out.append(entry)
        return out

    summary = _entries(bin_results)
    summary.sort(key=lambda e: (e['bin_id'], e['catalog']))

    if print_table:
        print('\n######### MEDIAN VOID RADIUS PER SAMPLE [Mpc/h] #########')
        print(f"{'Bin':>4} {'Catalog':>8} {'z_range':>12} {'N':>8} {'med(Rv)':>8}   med(Rv) per seed")
        for e in summary:
            seeds_txt = ''
            if 'R_void_median_seeds_mean' in e:
                seeds_txt = (f"   {e['R_void_median_seeds_mean']:.2f} "
                             f"+/- {e['R_void_median_seeds_std']:.2f}")
            print(f"{e['bin_id']+1:>4} {e['catalog']:>8} {str(e['z_range']):>12} "
                  f"{e['n_voids']:>8} {e['R_void_median']:>8.2f}{seeds_txt}")
        print('')
    return summary

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
    print(f"[Plots] Stacked maps + profiles + significance plot saved to {output_path}")
    plt.close()

def plot_jackknife_and_correlation(bin_results, output_path, max_Rvoid):
    n_bins = len(bin_results)
    fig, axes = plt.subplots(n_bins, 3, figsize=(16, 4.2* n_bins), gridspec_kw={'hspace': 0.35, 'wspace': 0.3})
    if n_bins == 1: 
        axes = axes[np.newaxis, :]

    for row, data in enumerate(bin_results):
        r_frac, profile, error, cov, cov_cmb = data['r_frac'], data['profile'], data['error'], data.get('cov_jk'), data.get('cov_cmb')
        label, z_mean, n_voids, is_ms = data.get('key', f"Bin {row+1}"), data['z_mean'], data.get('n_voids', '?'), data.get('catalog') == 'concat'
        error_cmb = np.sqrt(np.diag(cov_cmb))
        
        ax_p = axes[row, 0]
        ax_p.axhline(0, color='k', linestyle=':', alpha=0.5, linewidth=1)
        ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax_p.errorbar(r_frac, profile * 1e3, yerr=error * 1e3, fmt='o-', color='xkcd:steel blue', capsize=3, linewidth=1.8, label='JK error')
        ax_p.errorbar(r_frac + 0.05, profile * 1e3, yerr=error_cmb * 1e3, fmt='none', ecolor='xkcd:dark red', alpha=0.3, capsize=3, label='CMB error')
        ax_p.set_xlim(-0.05, max_Rvoid + 0.05)
        ax_p.set_xlabel(r'$r\,/\,R_v$')
        ax_p.set_title(f"Bin {label}  (z={z_mean:.3f}, N={n_voids})")
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc='lower right', frameon=True, fontsize=9)
        if row == 0: ax_p.set_ylabel(r'$\kappa\;[10^{-3}]$')
        else: ax_p.tick_params(labelleft=False)

        def _plot_corr(ax, cov_mat, title, first_col):
            if cov_mat is None:
                ax.text(0.5, 0.5, 'No covariance data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                ax.set_axis_off()
                return
            std = np.sqrt(np.diag(cov_mat))
            with np.errstate(invalid='ignore'):
                corr = cov_mat / np.outer(std, std)
            corr = np.nan_to_num(corr)
            im = ax.imshow(corr, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                           extent=[r_frac[0], r_frac[-1], r_frac[0], r_frac[-1]], aspect='auto')
            plt.colorbar(im, ax=ax, label='Correlation', fraction=0.046, pad=0.04)
            ax.set_xlabel(r'$r\,/\,R_v$')
            ax.set_title(title)
            if first_col: ax.set_ylabel(r'$r\,/\,R_v$')
            else: ax.tick_params(labelleft=False)

        jk_title = f"{'Seed-to-seed corr.' if is_ms else 'JK corr.'} — Bin {label}"
        _plot_corr(axes[row, 1], cov, 
                   jk_title,
                   True)
        _plot_corr(axes[row, 2], 
                   cov_cmb, 
                   f"CMB corr. — Bin {label}",
                   False)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_seed_consistency(bin_results_list, output_path, max_Rvoid):
    ms_bins = [d for d in bin_results_list if d.get('catalog') == 'concat' and d.get('seed_results') is not None]
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
    print(f"[Plots] Merge-vs-concat comparison plot saved to {output_path}")

def plot_map_comparison(collected, output_path, max_Rvoid,
                        success_label='PLANCK_PR4', order=None):
    labels = [l for l in (order or list(collected.keys())) if l in collected]
    if not labels:
        print("[Plots] Nothing for plot")
        return

    def _net(d):
        prof, err = np.asarray(d['profile']), np.asarray(d['error'])
        nrm, nrs = d.get('null_rand_mean'), d.get('null_rand_std')
        nr = d.get('n_randoms_done', 0) or 0
        if nrm is not None and not np.all(np.isnan(nrm)):
            net = prof - np.asarray(nrm)
            nerr = np.sqrt(err ** 2 + (np.asarray(nrs) / np.sqrt(nr)) ** 2) if nr > 0 else err
            band = np.asarray(nrs)
        else:
            net, nerr, band = prof, err, None
        return net, nerr, band

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color='k', linestyle=':', alpha=0.6, zorder=2)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7, zorder=2)

    others = [l for l in labels if l != success_label]
    colors = cm.viridis(np.linspace(0.05, 0.9, max(len(others), 1)))
    for lab, col in zip(others, colors):
        d = collected[lab]
        net, _, _ = _net(d)
        ax.plot(d['r_frac'], net * 1e3, '-', color=col, linewidth=1.7, alpha=0.9,
                zorder=3, label=f"{lab} (N={d.get('n_voids', '?')})")

    if success_label in collected:
        d = collected[success_label]
        net, nerr, band = _net(d)
        if band is not None:
            ax.fill_between(d['r_frac'], -band * 1e3, band * 1e3, color='xkcd:grey',
                            alpha=0.35, zorder=1, label=r'PR4 $1\sigma$ cosmic var')
        ax.errorbar(d['r_frac'], net * 1e3, yerr=nerr * 1e3, fmt='o-', color='k',
                    capsize=3, linewidth=2.0, zorder=5,
                    label=f"{success_label} (N={d.get('n_voids', '?')})")

    ax.set_xlim(-0.1, max_Rvoid + 0.1)
    ax.set_xlabel(r'$r\,/\,R_v$')
    ax.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Plots] Map-comparison panel saved to {output_path}")


def _net_and_err(d):
    prof, err = np.asarray(d['profile']), np.asarray(d['error'])
    nrm, nrs = d.get('null_rand_mean'), d.get('null_rand_std')
    nr = d.get('n_randoms_done', 0) or 0
    if nrm is not None and not np.all(np.isnan(nrm)):
        net = prof - np.asarray(nrm)
        nerr = np.sqrt(err ** 2 + (np.asarray(nrs) / np.sqrt(nr)) ** 2) if nr > 0 else err
        band = np.asarray(nrs)
    else:
        net, nerr, band = prof, err, None
    return net, nerr, band


def plot_act_vs_pr4(collected, output_path, max_Rvoid, success_label, other_label,
                    footprint_mask=None, footprint_coord='C'):
    if success_label not in collected or other_label not in collected:
        print("[Plots] Not so much profiles")
        return
    ds, do = collected[success_label], collected[other_label]
    r = np.asarray(ds['r_frac'])
    net_s, err_s, band_s = _net_and_err(ds)
    net_o, err_o, _ = _net_and_err(do)

    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[3, 1],
                           hspace=0.06, wspace=0.22)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_p)

    ax_p.axhline(0, color='k', linestyle=':', alpha=0.6)
    ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    if band_s is not None:
        ax_p.fill_between(r, -band_s * 1e3, band_s * 1e3, color='xkcd:grey',
                          alpha=0.30, zorder=1, label=r'PR4 $1\sigma$ cosmic var')
    ax_p.errorbar(r, net_o * 1e3, yerr=err_o * 1e3, fmt='s-', color='xkcd:teal',
                  capsize=3, linewidth=1.7, zorder=4,
                  label=f"{other_label} (N={do.get('n_voids','?')})")
    ax_p.errorbar(r, net_s * 1e3, yerr=err_s * 1e3, fmt='o-', color='k',
                  capsize=3, linewidth=2.0, zorder=5,
                  label=f"{success_label} (N={ds.get('n_voids','?')})")
    ax_p.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc='lower right', frameon=True, fontsize=9)
    ax_p.tick_params(labelbottom=False)

    denom = np.sqrt(err_s ** 2 + err_o ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = (net_s - net_o) / denom
    ax_s.axhline(0, color='k', alpha=0.5)
    for y in (-2, 2):
        ax_s.axhline(y, color='gray', linestyle='--', alpha=0.7)
    ax_s.axhspan(-2, 2, color='gray', alpha=0.10, zorder=0)
    ax_s.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax_s.plot(r, sig, 'o-', color='xkcd:crimson', markersize=4, linewidth=1.4)
    ax_s.set_ylim(-5, 5)
    ax_s.set_xlim(-0.1, max_Rvoid + 0.1)
    ax_s.set_xlabel(r'$r\,/\,R_v$')
    ax_s.set_ylabel(r'$\dfrac{\kappa_{\rm PR4}-\kappa_{\rm ACT}}{\sqrt{\sigma_{\rm PR4}^2+\sigma_{\rm ACT}^2}}$')
    ax_s.grid(True, alpha=0.25)

    ax_m = fig.add_subplot(gs[:, 1])
    if footprint_mask is not None:
        plt.sca(ax_m)
        try:
            hp.mollview(footprint_mask, coord=[footprint_coord, 'G'], hold=True,
                        cbar=False, cmap='Greys', min=0, max=1,
                        title='ACT footprint + voids')
            hp.graticule(dpar=30, dmer=30, alpha=0.3)
            vl = ds.get('void_l'); vb = ds.get('void_b')
            if vl is not None and vb is not None:
                hp.projscatter(np.asarray(vl), np.asarray(vb), lonlat=True,
                               s=4, color='red', alpha=0.6)
        except Exception as e:
            ax_m.text(0.5, 0.5, f'mollview failed:\n{e}', ha='center', va='center',
                      transform=ax_m.transAxes, fontsize=9)
    else:
        ax_m.set_axis_off()

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Plots] ACT vs PR4 panel saved to {output_path}")