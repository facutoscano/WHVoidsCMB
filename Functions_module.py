##### Function module for the CMB lensing voids profiles #####

#%% Imports
import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import healpy as hp
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord

plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.5, 'lines.markersize': 5,
    'legend.fontsize': 11, 'legend.frameon': False
})


#%% Geometry and cosmology
def get_angularsize_comoving(z, size_mpch):
    """Angle [deg] subtended by a comoving size [Mpc/h] at redshift z (fm.Planck18 cosmology)."""
    h = Planck18.h
    d_c = Planck18.comoving_distance(z).value * h
    return np.degrees(size_mpch / d_c)


def stamp_radius(npix, extent):
    """
    Distance of every stamp pixel to the stamp centre, in the units of `extent`
    (full side of the stamp). healpy's projectors put the projection centre at
    pixel (npix-1)/2, not at npix//2: for even npix the latter would shift the
    centre by half a pixel (a miscentring of 0.7 px along the diagonal).
    """
    c = 0.5 * (npix - 1)
    idx = np.arange(npix) - c
    yy, xx = np.meshgrid(idx, idx, indexing='ij')
    return np.sqrt(xx * xx + yy * yy) * (extent / npix)


def to_map_frame(l_gal, b_gal, frame):
    """(l,b) galactic [deg] -> lon/lat of the map frame ('galactic' or 'equatorial')."""
    if frame in ('galactic', 'gal', 'G'):
        return np.asarray(l_gal, float), np.asarray(b_gal, float)
    if frame in ('equatorial', 'icrs', 'celestial', 'C'):
        c = SkyCoord(l=np.asarray(l_gal) * u.degree,
                     b=np.asarray(b_gal) * u.degree, frame='galactic').icrs
        return c.ra.degree, c.dec.degree
    raise ValueError(f'unknown frame: {frame}')


def frame_code(frame):
    return 'C' if frame in ('equatorial', 'icrs', 'celestial', 'C') else 'G'


#%% Filters
def apply_wiener_filter(cmb_alm, nlkk_file, lmax=2048):
    """W_L = C_L / (C_L + N_L) from a (L, N_L, C_L+N_L) file."""
    nlkk_data = np.loadtxt(nlkk_file)
    ell_nlkk = nlkk_data[:, 0].astype(int)
    nl_kk = nlkk_data[:, 1]
    sn_kk = nlkk_data[:, 2]
    cl_kk = np.maximum(sn_kk - nl_kk, 0)

    W = np.zeros(lmax + 1)
    for i, ell in enumerate(ell_nlkk):
        if ell > lmax:
            break
        denom = cl_kk[i] + nl_kk[i]
        W[ell] = cl_kk[i] / denom if denom > 0 else 0.0
    alm_filtered = hp.almxfl(cmb_alm.copy(), W)
    print('[Data] Wiener filter applied.')
    return alm_filtered, W


#%% Masks and footprints
def binary_mask(path, nside=None, thr=0.9):
    m = hp.read_map(path, dtype=np.float64)
    if nside is not None and hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)
    return np.where(m >= thr, 1.0, 0.0)


def mask_fraction_map(path, nside_out, frame_in='G', frame_out='G', thr=0.9):
    """
    Fraction of each nside_out pixel inside the (binary, >=thr) mask, rotated to
    frame_out. Used for coverage: mean over a disc = fraction of the disc inside.
    """
    m = hp.read_map(path, dtype=np.float64)
    b = np.where(m >= thr, 1.0, 0.0)
    if hp.get_nside(b) != nside_out:
        b = hp.ud_grade(b, nside_out=nside_out)
    if frame_in != frame_out:
        b = hp.Rotator(coord=[frame_in, frame_out]).rotate_map_pixel(b)
    return np.clip(b, 0.0, 1.0)


def footprint_mask(l, b, output_nside, footprint_nside=32):
    """Occupancy mask of positions (l,b) at footprint_nside, upgraded to output_nside."""
    m = np.zeros(hp.nside2npix(footprint_nside), dtype=np.float64)
    m[hp.ang2pix(footprint_nside, l, b, lonlat=True)] = 1.0
    out = hp.ud_grade(m, nside_out=output_nside)
    out[out > 0] = 1.0
    return out


def fill_small_holes(m, n_neigh_min=6):
    """One pass: a pixel off the mask with >= n_neigh_min of its 8 neighbours on is switched on."""
    nside = hp.get_nside(m)
    off = np.flatnonzero(m < 0.5)
    if off.size == 0:
        return m
    nb = hp.get_all_neighbours(nside, off)            # (8, n)
    on = np.where(nb >= 0, m[np.clip(nb, 0, None)] > 0.5, False).sum(axis=0)
    out = m.copy()
    out[off[on >= n_neigh_min]] = 1.0
    return out


def survey_footprint(src, data_folder, nside_out, cache_dir=None, label='survey', force=False):
    """
    Angular footprint of the tracer survey in GALACTIC coordinates at nside_out.
      src = None                                 -> no footprint (whole sky)
      src = {'kind':'points', 'path', 'cols':(lon,lat), 'frame':'icrs'|'galactic',
             'skiprows', 'n_files', 'nside'}      -> occupancy of the tracer positions
      src = {'kind':'healpix', 'path', 'coord':'G'|'C'} -> an existing mask
    """
    if src is None:
        return None
    path = None
    if cache_dir is not None:
        tag = hashlib.md5(json.dumps(src, sort_keys=True, default=str).encode()).hexdigest()[:8]
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'footprint_{label}_ns{nside_out}_{tag}.fits')
        if os.path.exists(path) and not force:
            return hp.read_map(path, dtype=np.float64)

    if src['kind'] == 'healpix':
        p = src['path'] if os.path.isabs(src['path']) else os.path.join(data_folder, src['path'])
        F = mask_fraction_map(p, nside_out, frame_in=src.get('coord', 'G'), frame_out='G', thr=0.5)
        F = np.where(F >= 0.5, 1.0, 0.0)
    elif src['kind'] == 'points':
        files = ([src['path'].format(k + 1) for k in range(src['n_files'])]
                 if src.get('n_files') else [src['path']])
        lon, lat = [], []
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(data_folder, f)
            d = pd.read_csv(p, sep=r'\s+', header=None, skiprows=src.get('skiprows', 0),
                            usecols=list(src['cols']), comment='#')
            lon.append(d[src['cols'][0]].values.astype(float))
            lat.append(d[src['cols'][1]].values.astype(float))
        lon, lat = np.concatenate(lon), np.concatenate(lat)
        if src.get('frame', 'galactic') in ('icrs', 'equatorial'):
            c = SkyCoord(ra=lon * u.degree, dec=lat * u.degree, frame='icrs').galactic
            lon, lat = c.l.degree, c.b.degree
        ns = src.get('nside', 64)
        occ = np.zeros(hp.nside2npix(ns))
        occ[hp.ang2pix(ns, lon, lat, lonlat=True)] = 1.0
        occ = fill_small_holes(occ)
        F = hp.ud_grade(occ, nside_out=nside_out)
        F = np.where(F > 0, 1.0, 0.0)
        print(f'[footprint] {label}: {len(lon)} tracer positions -> occupancy at nside {ns}')
    else:
        raise ValueError(f"unknown footprint kind: {src['kind']}")
    print(f'[footprint] {label}: fsky={F.mean():.3f} ({F.mean() * 41253:.0f} deg2)')
    if path is not None:
        hp.write_map(path, F.astype(np.float32), overwrite=True, dtype=np.float32)
    return F


def disc_coverage(lon, lat, z, r_void, frac_map, rad_frac):
    """Fraction of the disc of radius rad_frac*R_v (in the frame of frac_map) inside frac_map."""
    nside = hp.get_nside(frac_map)
    lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
    z, r_void = np.atleast_1d(z), np.atleast_1d(r_void)
    theta = np.radians(get_angularsize_comoving(z, rad_frac * r_void))
    vec = hp.ang2vec(lon, lat, lonlat=True).reshape(-1, 3)
    cov = np.empty(len(lon))
    for i in range(len(lon)):
        pix = hp.query_disc(nside, vec[i], theta[i])
        if len(pix) == 0:
            pix = [hp.vec2pix(nside, *vec[i])]
        cov[i] = frac_map[pix].mean()
    return cov


def footprint_coverage(l_gal, b_gal, z, r_void, mask, nside, frame, max_Rvoid):
    """disc_coverage for galactic (l,b) on a mask that lives in `frame`."""
    lon, lat = to_map_frame(l_gal, b_gal, frame)
    return disc_coverage(lon, lat, z, r_void, mask, max_Rvoid)


def exclusion_mask(base_mask, l, b, redshifts, r_voids, excl_factor, silent=False):
    """base_mask (galactic) with discs of excl_factor*R_v around the real voids set to 0."""
    if excl_factor is None or excl_factor <= 0:
        return base_mask
    nside = hp.get_nside(base_mask)
    excl = base_mask.copy()
    for i in range(len(l)):
        theta = np.radians(get_angularsize_comoving(redshifts[i], excl_factor * r_voids[i]))
        excl[hp.query_disc(nside, hp.ang2vec(l[i], b[i], lonlat=True), theta)] = 0.0
    if not silent:
        print(f'[Mask] random pool: {excl.sum() / max(base_mask.sum(), 1) * 100:.1f}% of the region '
              f'remains after excluding {excl_factor}*Rv around {len(l)} voids.')
    return excl


def uniform_points_in_mask(center_mask, n, rng):
    """n points uniform on the sphere restricted to center_mask > 0.5 (galactic l,b in deg)."""
    nside = hp.get_nside(center_mask)
    frac = max(center_mask.mean(), 1e-4)
    L, B = [], []
    got = 0
    while got < n:
        m = int(1.3 * (n - got) / frac) + 16
        l = rng.uniform(0.0, 360.0, m)
        b = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, m)))
        ok = center_mask[hp.ang2pix(nside, l, b, lonlat=True)] > 0.5
        L.append(l[ok]); B.append(b[ok]); got += ok.sum()
    return np.concatenate(L)[:n], np.concatenate(B)[:n]


def matched_randoms(z, r_void, center_mask, cov_map, rad_frac, min_cov, rng, max_tries=500):
    """
    Random positions (galactic) for objects with the given (z, R_v):
      - centre uniform inside center_mask (map mask x survey footprint [x exclusion]);
      - the disc of rad_frac*R_v must have coverage >= min_cov on cov_map, the SAME
        criterion applied to the real voids.
    If an object cannot be placed after max_tries candidates, its (z, R_v) is swapped
    for another one of the sample (counted in n_swap).
    """
    n = len(z)
    z, r_void = np.asarray(z, float).copy(), np.asarray(r_void, float).copy()
    out_l, out_b = np.empty(n), np.empty(n)
    n_swap = 0
    if not min_cov or min_cov <= 0:
        out_l, out_b = uniform_points_in_mask(center_mask, n, rng)
        return out_l, out_b, z, r_void, 0
    pool_l, pool_b = uniform_points_in_mask(center_mask, 4 * n, rng)
    k = 0
    for i in range(n):
        tries = 0
        while True:
            if k >= len(pool_l):
                pool_l, pool_b = uniform_points_in_mask(center_mask, 4 * n, rng)
                k = 0
            li, bi = pool_l[k], pool_b[k]
            k += 1
            tries += 1
            if disc_coverage(li, bi, z[i], r_void[i], cov_map, rad_frac)[0] >= min_cov:
                out_l[i], out_b[i] = li, bi
                break
            if tries >= max_tries:
                j = rng.integers(n)
                z[i], r_void[i] = z[j], r_void[j]
                n_swap += 1
                tries = 0
    return out_l, out_b, z, r_void, n_swap


def rotate_map(map_data, rot_angles):
    nside = hp.npix2nside(len(map_data))
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    r = hp.rotator.Rotator(rot=rot_angles, deg=True, inv=True)
    theta_rot, phi_rot = r(theta, phi)
    return map_data[hp.ang2pix(nside, theta_rot, phi_rot)]


#%% Stacking and profiles
def stacking_gnomonic(l, b, redshifts, r_voids, cmb_map, mask, max_Rvoid, npix_stamp,
                      stacked_range, silent=False):
    if not silent:
        print(f'[Profiles] Stacking {len(stacked_range)} maps...')
    sum_map = np.zeros((npix_stamp, npix_stamp))
    count_map = np.zeros((npix_stamp, npix_stamp))
    nside = hp.npix2nside(len(cmb_map))
    vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    for i, idx in enumerate(stacked_range):
        box_size_deg = get_angularsize_comoving(redshifts[idx], 2 * max_Rvoid * r_voids[idx])
        reso_arcmin = (box_size_deg * 60.) / npix_stamp
        proj = hp.projector.GnomonicProj(rot=[l[idx], b[idx], 0], xsize=npix_stamp,
                                         ysize=npix_stamp, reso=reso_arcmin)
        stamp_data = proj.projmap(cmb_map, vec2pix_func=vec2pix_func)
        stamp_mask = proj.projmap(mask, vec2pix_func=vec2pix_func)
        valid = (stamp_mask > 0.9) & (~np.isnan(stamp_data))
        sum_map[valid] += stamp_data[valid]
        count_map[valid] += 1
        if not silent and (i + 1) % 150 == 0:
            print(f'Stacked {i + 1} / {len(stacked_range)}')
    return sum_map, count_map


def radial_sums_healpix(lon, lat, redshifts, r_voids, cmb_map, mask, bins_frac):
    """
    Radial profile measured directly on the HEALPix pixels (no projection, no resampling).

    For every object, all map pixels within bins_frac[-1]*R_v are taken and assigned to
    rings of r/R_v by their exact angular distance to the centre, r = angle/theta(R_v).
    Each object enters every ring with weight f_i = (unmasked pixels)/(all pixels) of its
    ring, i.e. ring mean m_i weighted by f_i:
        SUM_k   += s_ik / N_ik      (s = sum of unmasked values, N = all pixels in the ring)
        COUNT_k += n_ik / N_ik      (n = unmasked pixels)
    and the stacked profile is SUM/COUNT. This is the continuum limit of the stamp
    estimator (equal weight per object in units of R_v, masked area down-weighted), but
    uses every map pixel exactly once per ring: no aliasing of small-scale power through
    the stamp grid and no gnomonic distortion for large angular sizes.
    Returns SUM, COUNT (arrays of len(bins_frac)-1), additive over objects (jackknife).
    """
    nside = hp.get_nside(cmb_map)
    edges = np.asarray(bins_frac, float)
    nb = len(edges) - 1
    S, C = np.zeros(nb), np.zeros(nb)
    lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
    theta_rv = np.radians(get_angularsize_comoving(np.atleast_1d(redshifts), np.atleast_1d(r_voids)))
    vecs = np.atleast_2d(hp.ang2vec(lon, lat, lonlat=True))
    for i in range(len(lon)):
        pix = hp.query_disc(nside, vecs[i], edges[-1] * theta_rv[i])
        if len(pix) == 0:
            continue
        pv = np.column_stack(hp.pix2vec(nside, pix))
        x = np.arccos(np.clip(pv @ vecs[i], -1.0, 1.0)) / theta_rv[i]
        k = np.searchsorted(edges, x, side='right') - 1
        inb = (k >= 0) & (k < nb)
        vals = cmb_map[pix]
        good = inb & (mask[pix] > 0.9) & np.isfinite(vals) & (vals != hp.UNSEEN)
        n_all = np.bincount(k[inb], minlength=nb)[:nb]
        n_ok = np.bincount(k[good], minlength=nb)[:nb]
        s_ok = np.bincount(k[good], weights=vals[good], minlength=nb)[:nb]
        has = n_all > 0
        S[has] += s_ok[has] / n_all[has]
        C[has] += n_ok[has] / n_all[has]
    return S, C


def profile_from_sums(S, C):
    S, C = np.asarray(S, float), np.asarray(C, float)
    out = np.full(S.shape, np.nan)
    np.divide(S, C, out=out, where=C > 0)
    return out


def stack_mean_map(sum_map, count_map):
    out = np.full_like(sum_map, np.nan)
    good = count_map > 0
    out[good] = sum_map[good] / count_map[good]
    return out


def radial_profile_weighted(sum_map, count_map, max_Rvoid, bins_frac, silent=False):
    """Count-weighted radial profile in units of r/R_v (stamp side = 2*max_Rvoid)."""
    r_units = stamp_radius(sum_map.shape[0], 2 * max_Rvoid)
    profile, r_centers = [], []
    if not silent:
        print('[Profiles] Computing count-weighted radial profile...')
    for i in range(len(bins_frac) - 1):
        ring = (r_units >= bins_frac[i]) & (r_units < bins_frac[i + 1])
        den = np.nansum(count_map[ring])
        profile.append(np.nansum(sum_map[ring]) / den if den > 0 else np.nan)
        r_centers.append((bins_frac[i] + bins_frac[i + 1]) / 2.)
    return np.array(profile), np.array(r_centers)


def sample_covariance(profiles, kind='jk'):
    profiles = np.asarray(profiles, float)
    if profiles.ndim != 2 or len(profiles) < 2:
        return None, None
    good = np.isfinite(profiles).all(axis=0)
    p = profiles.shape[1]
    C = np.full((p, p), np.nan)
    if kind == 'jk':
        delta = profiles[:, good] - profiles[:, good].mean(axis=0)
        n = profiles.shape[0]
        C[np.ix_(good, good)] = (n - 1) / n * np.dot(delta.T, delta)
    elif kind == 'cmb':
        C[np.ix_(good, good)] = np.atleast_2d(np.cov(profiles[:, good], rowvar=False))
    else:
        raise ValueError(f'Unknown kind: {kind}')
    return C, np.sqrt(np.diag(C))


#%% Background subtraction and net signal
BKG_MODES = ('randoms', 'outer', 'randoms+outer')


def bkg_apply(X, r, mode, outer):
    """
    Apply the 'outer' part of the background subtraction to profile(s) X (..., nbins):
    X - mean(X over r/R_v in outer). NaN bins stay NaN and do not contaminate the others.
    """
    X = np.array(X, dtype=float, copy=True)
    if 'outer' not in mode:
        return X
    w = (np.asarray(r) >= outer[0]) & (np.asarray(r) <= outer[1])
    if not np.any(w):
        raise ValueError(f'no radial bin inside bkg_outer={outer}')
    with np.errstate(invalid='ignore'):
        m = np.nanmean(X[..., w], axis=-1)
    return X - np.expand_dims(m, -1)


def net_signal(res, mode=None, outer=None):
    """
    Net profile of a stacking result and its uncertainties.

    mode  'randoms'       : net = P - <P_rand>   (randoms drawn in the same footprint)
          'outer'         : net = P - mean(P over r/R_v in `outer`)
          'randoms+outer' : both (randoms first)
    The same linear operation is applied to every jackknife and random profile, so the
    covariances are those of the net profile.

    Returns dict:
      net, err   : net profile and its error (JK + uncertainty of <P_rand>)
      cov_jk     : JK covariance of the net profile
      cov_rand   : covariance of the net profile from randoms, inflated by (1+1/N_rand)
                   when <P_rand> is subtracted (use for chi2)
      band       : 1 sigma of a single random stack (cosmic variance + noise)
      n_rand, mode
    """
    mode = mode or res.get('bkg_mode', 'randoms')
    outer = outer or res.get('bkg_outer', (2.0, 2.5))
    if mode not in BKG_MODES:
        raise ValueError(f'unknown bkg_mode {mode}; options {BKG_MODES}')
    r = np.asarray(res['r_frac'])
    P = np.asarray(res['profile'], float)

    R = res.get('null_profiles_rand')
    R = np.asarray(R, float) if R is not None and len(R) > 1 else None
    nr = 0 if R is None else len(R)
    use_r = ('randoms' in mode) and R is not None
    with np.errstate(invalid='ignore'):
        Rbar = np.nanmean(R, axis=0) if use_r else np.zeros_like(P)
    net = bkg_apply(P - Rbar, r, mode, outer)

    cov_jk = None
    if res.get('jk_profiles') is not None and len(res['jk_profiles']) > 1:
        cov_jk, _ = sample_covariance(bkg_apply(res['jk_profiles'], r, mode, outer), kind='jk')

    cov_single = None
    if R is not None:
        cov_single, _ = sample_covariance(bkg_apply(R, r, mode, outer), kind='cmb')

    var = np.zeros_like(P)
    if cov_jk is not None:
        var = var + np.diag(cov_jk)
    if use_r and cov_single is not None:
        var = var + np.diag(cov_single) / nr
    cov_chi2 = None
    if cov_single is not None:
        cov_chi2 = cov_single * (1.0 + 1.0 / nr) if use_r else cov_single
    band = np.sqrt(np.diag(cov_single)) if cov_single is not None else None
    return {'net': net, 'err': np.sqrt(var), 'cov_jk': cov_jk, 'cov_rand': cov_chi2,
            'band': band, 'n_rand': nr, 'mode': mode, 'outer': tuple(outer)}


def _net_and_err(d):
    """Backwards-compatible wrapper: (net, err, band)."""
    ns = net_signal(d)
    return ns['net'], ns['err'], ns['band']


def bkg_text(res):
    mode = res.get('bkg_mode', 'randoms')
    if mode == 'randoms':
        return 'net = P - <randoms>'
    o = res.get('bkg_outer', (2.0, 2.5))
    s = f'{o[0]:g}-{o[1]:g}' + r'$R_v$'
    return f'net = P - mean({s})' if mode == 'outer' else f'net = P - <randoms> - mean({s})'


#%% chi2
def chi2_profile(res_a, res_b=None, hartlap=True):
    """
    chi2 of (net_a - net_b) [or net_a against 0] with C = C_a + C_b from randoms.
    Empirical distribution: the same statistic evaluated on the random stacks of a
    (minus those of b), which are stored in the results.
    """
    from scipy.stats import chi2 as _chi2
    A = net_signal(res_a)
    if A['cov_rand'] is None:
        raise ValueError('no random profiles stored (n_rand_factor=0?)')
    C, N, delta = A['cov_rand'], A['n_rand'], A['net']
    B = None
    if res_b is not None:
        B = net_signal(res_b)
        delta = A['net'] - B['net']
        if B['cov_rand'] is not None:
            C = C + B['cov_rand']
            N = min(N, B['n_rand'])
    good = np.isfinite(delta) & np.isfinite(np.diag(C))
    p = int(good.sum())
    if p == 0:
        raise ValueError('chi2: no radial bin with a finite profile and covariance')
    h = (N - p - 2) / (N - 1) if hartlap else 1.0
    if h <= 0:
        raise ValueError(f'chi2 with {p} bins needs N_rand > p+2 = {p + 2} random realizations (have {N})')
    Cg, dg = C[np.ix_(good, good)], delta[good]
    w, V = np.linalg.eigh(Cg)
    if w.min() <= 0:
        print(f'[chi2] (!) covariance not positive definite (min eigenvalue={w.min():.2e}); using pinv')
    Cinv = np.linalg.pinv(Cg)
    chi2 = float(h * dg @ Cinv @ dg)
    pte = float(_chi2.sf(chi2, p))
    order = np.argsort(w)[::-1]
    contrib = h * (V[:, order].T @ dg) ** 2 / w[order]

    # Empirical null: each random stack is treated like the data, i.e. compared with the
    # mean and covariance of the OTHER randoms (leave-one-out). Evaluating the randoms
    # with the covariance they were used to estimate (in-sample) compresses the null
    # distribution and biases the empirical PTE low.
    emp = None
    Ra = res_a.get('null_profiles_rand')
    if Ra is not None and len(Ra) > p + 4:
        Xa = bkg_apply(Ra, res_a['r_frac'], A['mode'], A['outer'])[:, good]
        Rb = None if res_b is None else res_b.get('null_profiles_rand')
        Xb = None
        if Rb is not None and len(Rb) > p + 4:
            Xb = bkg_apply(Rb, res_b['r_frac'], B['mode'], B['outer'])[:, good]
        n = len(Xa) if Xb is None else min(len(Xa), len(Xb))
        Xa = Xa[:n]
        Xb = None if Xb is None else Xb[:n]
        ok = np.all(np.isfinite(Xa), axis=1) & (True if Xb is None else np.all(np.isfinite(Xb), axis=1))
        Xa, Xb = Xa[ok], (None if Xb is None else Xb[ok])
        n = len(Xa)
        Cb_fixed = None
        if res_b is not None and Xb is None and B['cov_rand'] is not None:
            Cb_fixed = B['cov_rand'][np.ix_(good, good)]
        hl = (n - 1 - p - 2) / (n - 2) if hartlap else 1.0
        emp = np.empty(n)
        for k in range(n):
            m = np.arange(n) != k
            d = Xa[k] - Xa[m].mean(0)
            Ck = np.cov(Xa[m], rowvar=False) * (1 + 1.0 / (n - 1))
            if Xb is not None:
                d = d - (Xb[k] - Xb[m].mean(0))
                Ck = Ck + np.cov(Xb[m], rowvar=False) * (1 + 1.0 / (n - 1))
            elif Cb_fixed is not None:
                Ck = Ck + Cb_fixed
            emp[k] = hl * d @ np.linalg.pinv(np.atleast_2d(Ck)) @ d
    out = {'chi2': chi2, 'p': p, 'h': h, 'N_rand': N, 'pte': pte, 'eig': w[order],
           'contrib': contrib, 'emp': emp,
           'pte_emp': None if emp is None else float((np.sum(emp >= chi2) + 1) / (len(emp) + 1))}
    msg = f'chi2={chi2:.1f} / {p} bins (cov randoms, N_rand={N}, h={h:.3f}) PTE={pte:.3f}'
    if emp is not None:
        msg += f" | empirical PTE={out['pte_emp']:.3f} ({len(emp)} randoms)"
    out['msg'] = msg
    return out


def plot_chi2(ch, out_path, title):
    from scipy.stats import chi2 as _chi2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'wspace': 0.28})
    p = ch['p']
    xmax = max(ch['chi2'] * 1.3, _chi2.ppf(0.999, p))
    x = np.linspace(0, xmax, 400)
    if ch['emp'] is not None:
        a1.hist(ch['emp'], bins=30, range=(0, xmax), density=True, histtype='step',
                color='xkcd:grey', lw=1.6, label=f"randoms ({len(ch['emp'])})")
    a1.plot(x, _chi2.pdf(x, p), color='xkcd:steel blue', lw=1.8, label=rf'$\chi^2_{{{p}}}$')
    xs = x[x >= ch['chi2']]
    a1.fill_between(xs, _chi2.pdf(xs, p), color='xkcd:steel blue', alpha=0.25)
    a1.axvline(ch['chi2'], color='xkcd:crimson', lw=2, label=rf"observed = {ch['chi2']:.1f}")
    ttl = f"PTE = {ch['pte']:.3f}" + (f"   (empirical {ch['pte_emp']:.3f})" if ch['pte_emp'] is not None else '')
    a1.set_title(ttl, fontsize=11)
    a1.set_xlabel(r'$\chi^2$'); a1.set_ylabel('density'); a1.legend(fontsize=9); a1.grid(alpha=0.25)
    k = np.arange(1, len(ch['contrib']) + 1)
    a2.bar(k, ch['contrib'], color='xkcd:crimson', alpha=0.75)
    a2.axhline(1, color='k', ls='--', lw=1, alpha=0.7)
    a2.set_xlabel('eigenvector of C (decreasing eigenvalue)')
    a2.set_ylabel(r'$h\,(v_i\cdot\Delta)^2/\lambda_i$')
    a2b = a2.twinx()
    a2b.semilogy(k, ch['eig'] / ch['eig'][0], 'o-', color='xkcd:steel blue', ms=3, lw=1)
    a2b.set_ylabel(r'$\lambda_i/\lambda_1$', color='xkcd:steel blue')
    a2.set_title(f"h = {ch['h']:.3f} (N_rand={ch['N_rand']}, p={p})", fontsize=11)
    a2.grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[chi2] figure: {out_path}')


#%% Bookkeeping
def build_rvoid_summary(bin_results, print_table=True):
    """Median radius of the stacked voids per bin and catalogue."""
    summary = []
    for i, res in enumerate(bin_results or []):
        entry = {'bin_id': int(res.get('bin_id', i)), 'catalog': res.get('catalog', 'single'),
                 'z_range': res.get('key'), 'n_voids': int(res.get('n_voids', 0)),
                 'R_void_median': float(res.get('R_void_median', np.nan))}
        r_seeds = [float(s['R_void_median']) for s in res.get('seed_results', [])
                   if 'R_void_median' in s]
        if r_seeds:
            entry['R_void_median_seeds_mean'] = float(np.mean(r_seeds))
            entry['R_void_median_seeds_std'] = float(np.std(r_seeds))
        summary.append(entry)
    summary.sort(key=lambda e: (e['bin_id'], e['catalog']))
    if print_table:
        print('\n######### MEDIAN VOID RADIUS PER SAMPLE [Mpc/h] #########')
        print(f"{'Bin':>4} {'Catalog':>8} {'z_range':>12} {'N':>8} {'med(Rv)':>8}   med(Rv) per seed")
        for e in summary:
            seeds_txt = ''
            if 'R_void_median_seeds_mean' in e:
                seeds_txt = (f"   {e['R_void_median_seeds_mean']:.2f} "
                             f"+/- {e['R_void_median_seeds_std']:.2f}")
            print(f"{e['bin_id'] + 1:>4} {e['catalog']:>8} {str(e['z_range']):>12} "
                  f"{e['n_voids']:>8} {e['R_void_median']:>8.2f}{seeds_txt}")
        print('')
    return summary


#%% Plots
def plot_stacked_maps_and_profiles(data_list, output_path, max_Rvoid):
    n_bins = len(data_list)
    fig = plt.figure(figsize=(6 * n_bins, 12))
    gs = gridspec.GridSpec(3, n_bins + 1, height_ratios=[1.2, 1.2, 0.6],
                           width_ratios=[1] * n_bins + [0.05], hspace=0.35, wspace=0.15)
    all_maps = np.array([d['map'] for d in data_list])
    v_max, v_min = np.nanpercentile(all_maps, 99.5), np.nanpercentile(all_maps, 0.5)
    extent = [-max_Rvoid, max_Rvoid, -max_Rvoid, max_Rvoid]

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(data['map'], origin='lower', cmap='viridis', extent=extent, vmin=v_min, vmax=v_max)
        ax.set_title(f"Bin {data.get('key', i + 1)} [{data.get('catalog', '')}]\n"
                     f"z={data['z_mean']:.3f}, N={data.get('n_voids', '?')}")
        ax.tick_params(labelbottom=False)
        if i == 0:
            ax.set_ylabel(r'$r\,/\,R_v$')
        else:
            ax.tick_params(labelleft=False)
    plt.colorbar(im, cax=fig.add_subplot(gs[0, -1]), label=r'$\kappa$')

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(gs[1, i])
        r = data['r_frac']
        ns = net_signal(data)
        if ns['band'] is not None:
            ax.fill_between(r, -ns['band'] * 1e3, ns['band'] * 1e3, color='xkcd:grey', alpha=0.5,
                            zorder=1, label=r'$1\sigma$ randoms')
        rot = data.get('null_profiles_rot')
        if rot is not None and len(rot) > 1:
            Rbar = (np.nanmean(data['null_profiles_rand'], 0) if 'randoms' in ns['mode']
                    and data.get('null_profiles_rand') is not None and len(data['null_profiles_rand']) > 1 else 0.0)
            rr = bkg_apply(np.asarray(rot, float) - Rbar, r, ns['mode'], ns['outer'])
            ax.fill_between(r, (np.nanmean(rr, 0) - np.nanstd(rr, 0)) * 1e3,
                            (np.nanmean(rr, 0) + np.nanstd(rr, 0)) * 1e3,
                            color='xkcd:salmon', alpha=0.5, zorder=2, label=r'$1\sigma$ rotations')
        ax.axhline(0, color='k', linestyle=':', alpha=0.6, zorder=3)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.8, zorder=3)
        ax.plot(r, data['profile'] * 1e3, '-', color='xkcd:dark grey', alpha=0.8, zorder=3,
                label='Raw profile')
        ax.errorbar(r, ns['net'] * 1e3, yerr=ns['err'] * 1e3, fmt='o-', color='xkcd:steel blue',
                    capsize=3, linewidth=1.8, zorder=4, label='Net signal')
        ax.set_xlim(-0.1, max_Rvoid + 0.1)
        ax.tick_params(labelbottom=False)
        ax.grid(True, alpha=0.25)
        ax.set_title(bkg_text(data), fontsize=9)
        if i == 0:
            ax.set_ylabel(r'$\kappa\;[10^{-3}]$')
            ax.legend(loc='lower right', frameon=True, fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    for i, data in enumerate(data_list):
        ax_sig = fig.add_subplot(gs[2, i])
        ns = net_signal(data)
        with np.errstate(divide='ignore', invalid='ignore'):
            significance = ns['net'] / ns['err']
        ax_sig.plot(data['r_frac'], significance, 'o-', color='xkcd:crimson', markersize=5)
        ax_sig.axhline(0, color='black', alpha=0.5)
        for y in (-2, 2):
            ax_sig.axhline(y, color='gray', linestyle='--', alpha=0.8)
        ax_sig.axhspan(-2, 2, color='gray', alpha=0.1, zorder=0)
        ax_sig.set_ylim(-5, 5); ax_sig.set_xlim(-0.1, max_Rvoid + 0.1)
        ax_sig.set_xlabel(r'$r\,/\,R_v$'); ax_sig.grid(True, alpha=0.25)
        if i == 0:
            ax_sig.set_ylabel(r'$S/N\;(\sigma)$')
        else:
            ax_sig.tick_params(labelleft=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'[Plots] Stacked maps + profiles + significance plot saved to {output_path}')
    plt.close()


def plot_jackknife_and_correlation(bin_results, output_path, max_Rvoid):
    n_bins = len(bin_results)
    fig, axes = plt.subplots(n_bins, 3, figsize=(16, 4.2 * n_bins),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.3}, squeeze=False)
    for row, data in enumerate(bin_results):
        r_frac = data['r_frac']
        ns = net_signal(data)
        label, z_mean, n_voids = data.get('key', f'Bin {row + 1}'), data['z_mean'], data.get('n_voids', '?')
        ax_p = axes[row, 0]
        ax_p.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        err_jk = np.sqrt(np.diag(ns['cov_jk'])) if ns['cov_jk'] is not None else np.zeros_like(ns['net'])
        ax_p.errorbar(r_frac, ns['net'] * 1e3, yerr=err_jk * 1e3, fmt='o-', color='xkcd:steel blue',
                      capsize=3, linewidth=1.8, label='JK error')
        if ns['cov_rand'] is not None:
            ax_p.errorbar(r_frac + 0.05, ns['net'] * 1e3, yerr=np.sqrt(np.diag(ns['cov_rand'])) * 1e3,
                          fmt='none', ecolor='xkcd:dark red', alpha=0.4, capsize=3, label='randoms error')
        ax_p.set_xlim(-0.05, max_Rvoid + 0.05)
        ax_p.set_xlabel(r'$r\,/\,R_v$')
        ax_p.set_title(f'Bin {label}  (z={z_mean:.3f}, N={n_voids})')
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc='lower right', frameon=True, fontsize=9)
        ax_p.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')

        def _plot_corr(ax, cov_mat, title, first_col):
            if cov_mat is None:
                ax.text(0.5, 0.5, 'No covariance', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                ax.set_axis_off()
                return
            std = np.sqrt(np.diag(cov_mat))
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.nan_to_num(cov_mat / np.outer(std, std))
            im = ax.imshow(corr, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                           extent=[r_frac[0], r_frac[-1], r_frac[0], r_frac[-1]], aspect='auto')
            plt.colorbar(im, ax=ax, label='Correlation', fraction=0.046, pad=0.04)
            ax.set_xlabel(r'$r\,/\,R_v$'); ax.set_title(title)
            if first_col:
                ax.set_ylabel(r'$r\,/\,R_v$')
            else:
                ax.tick_params(labelleft=False)

        _plot_corr(axes[row, 1], ns['cov_jk'], f'JK corr. - Bin {label}', True)
        _plot_corr(axes[row, 2], ns['cov_rand'], f'Randoms corr. - Bin {label}', False)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_seed_consistency(bin_results_list, output_path, max_Rvoid):
    ms_bins = [d for d in bin_results_list if d.get('catalog') == 'concat' and d.get('seed_results')]
    if not ms_bins:
        return
    n_bins = len(ms_bins)
    fig, axes = plt.subplots(1, n_bins, figsize=(6 * n_bins, 5), squeeze=False)
    for col, data in enumerate(ms_bins):
        ax, seed_results, r_frac = axes[0, col], data['seed_results'], data['r_frac']
        n_seeds = len(seed_results)
        colors = cm.plasma(np.linspace(0.05, 0.85, n_seeds))
        for j, s_res in enumerate(seed_results):
            ax.plot(r_frac, net_signal(s_res)['net'] * 1e3, color=colors[j], alpha=0.4, linewidth=1.0,
                    label=f'Seed {j + 1}' if n_seeds <= 10 else None)
        ns = net_signal(data)
        ax.errorbar(r_frac, ns['net'] * 1e3, yerr=ns['err'] * 1e3, fmt='o-', color='black',
                    linewidth=2.0, capsize=3, zorder=5, label='Combined')
        ax.axhline(0, color='k', linestyle=':', alpha=0.4)
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title(f"Bin {data.get('key')}  (z={data['z_mean']:.3f}, N={data.get('n_voids', '?')})\n{n_seeds} seeds")
        ax.set_xlabel(r'$r\,/\,R_v$'); ax.set_xlim(-0.05, max_Rvoid + 0.05); ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
        ax.legend(loc='lower right', frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_merge_vs_concat(pair_list, output_path, max_Rvoid):
    n = len(pair_list)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8), gridspec_kw={'hspace': 0.28, 'wspace': 0.2},
                             squeeze=False)
    for col, pair in enumerate(pair_list):
        c, m = net_signal(pair['concat']), net_signal(pair['merge'])
        r = pair['concat']['r_frac']
        ax_p = axes[0, col]
        ax_p.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        ax_p.errorbar(r, c['net'] * 1e3, yerr=c['err'] * 1e3, fmt='o-', color='xkcd:steel blue', capsize=3,
                      label=f"Concat (N={pair['concat'].get('n_voids', '?')})")
        ax_p.errorbar(r, m['net'] * 1e3, yerr=m['err'] * 1e3, fmt='s-', color='xkcd:crimson', capsize=3,
                      label=f"Merge (N={pair['merge'].get('n_voids', '?')})")
        ax_p.set_title(f"Bin {pair.get('bin_id', col) + 1}")
        ax_p.set_xlim(-0.05, max_Rvoid + 0.05); ax_p.tick_params(labelbottom=False)
        ax_p.grid(True, alpha=0.25); ax_p.legend(loc='lower right', frameon=True, fontsize=9)
        if col == 0:
            ax_p.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
        ax_d = axes[1, col]
        ax_d.axhline(0, color='k', linestyle=':', alpha=0.6)
        ax_d.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
        ax_d.errorbar(r, (c['net'] - m['net']) * 1e3, yerr=np.sqrt(c['err'] ** 2 + m['err'] ** 2) * 1e3,
                      fmt='o-', color='xkcd:dark grey', capsize=3)
        ax_d.set_xlim(-0.05, max_Rvoid + 0.05); ax_d.set_xlabel(r'$r\,/\,R_v$'); ax_d.grid(True, alpha=0.25)
        if col == 0:
            ax_d.set_ylabel(r'concat $-$ merge $\;[10^{-3}]$')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Plots] Merge-vs-concat comparison plot saved to {output_path}')


def _draw_map_panel(ax, collected, labels, ref_label, max_Rvoid):
    """Net profiles of several maps on one axis (ref with error bars and randoms band)."""
    ax.axhline(0, color='k', linestyle=':', alpha=0.6, zorder=2)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7, zorder=2)
    others = [l for l in labels if l != ref_label]
    colors = cm.viridis(np.linspace(0.05, 0.9, max(len(others), 1)))
    for lab, col in zip(others, colors):
        d = collected[lab]
        ax.plot(d['r_frac'], net_signal(d)['net'] * 1e3, '-', color=col, linewidth=1.7, alpha=0.9,
                zorder=3, label=f"{lab} (N={d.get('n_voids', '?')})")
    if ref_label in labels:
        d = collected[ref_label]
        ns = net_signal(d)
        if ns['band'] is not None:
            ax.fill_between(d['r_frac'], -ns['band'] * 1e3, ns['band'] * 1e3, color='xkcd:grey',
                            alpha=0.35, zorder=1, label=f'{ref_label} ' + r'$1\sigma$ randoms')
        ax.errorbar(d['r_frac'], ns['net'] * 1e3, yerr=ns['err'] * 1e3, fmt='o-', color='k',
                    capsize=3, linewidth=2.0, zorder=5, label=f"{ref_label} (N={d.get('n_voids', '?')})")
    ax.set_xlim(-0.1, max_Rvoid + 0.1)
    ax.grid(True, alpha=0.25)


def plot_map_comparison(collected, output_path, max_Rvoid, success_label='PLANCK_PR4', order=None,
                        title=None):
    labels = [l for l in (order or list(collected.keys())) if l in collected]
    if not labels:
        print('[Plots] Nothing to plot')
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_map_panel(ax, collected, labels, success_label, max_Rvoid)
    ax.set_xlabel(r'$r/R_v$'); ax.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    first = collected[labels[0]]
    ax.set_title(title or bkg_text(first), fontsize=10)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Plots] Map-comparison panel saved to {output_path}')


def plot_catalog_columns(by_cat, output_path, max_Rvoid, order=None, ref_label='PLANCK_PR4',
                         suptitle=None):
    """
    One column per void catalogue (e.g. WH | BOSS). Top: net profiles of every map
    run for that catalogue. Bottom: S/N of each map.
    by_cat = {cat: {map_label: result}}
    """
    cats = [c for c in by_cat if by_cat[c]]
    if not cats:
        print('[Plots] Nothing to plot in columns')
        return
    fig = plt.figure(figsize=(6.2 * len(cats), 7.5))
    gs = gridspec.GridSpec(2, len(cats), height_ratios=[3, 1], hspace=0.05, wspace=0.08)
    ax_top0 = None
    for j, cat in enumerate(cats):
        col = by_cat[cat]
        labels = [l for l in (order or list(col.keys())) if l in col]
        ref = ref_label if ref_label in labels else labels[0]
        ax = fig.add_subplot(gs[0, j], sharey=ax_top0)
        ax_top0 = ax_top0 or ax
        _draw_map_panel(ax, col, labels, ref, max_Rvoid)
        ax.set_title(cat, fontsize=13)
        ax.tick_params(labelbottom=False)
        ax.legend(loc='lower right', frameon=True, fontsize=8)
        axs = fig.add_subplot(gs[1, j], sharex=ax)
        colors = dict(zip([l for l in labels if l != ref],
                          cm.viridis(np.linspace(0.05, 0.9, max(len(labels) - 1, 1)))))
        colors[ref] = 'k'
        for lab in labels:
            ns = net_signal(col[lab])
            with np.errstate(divide='ignore', invalid='ignore'):
                axs.plot(col[lab]['r_frac'], ns['net'] / ns['err'], 'o-', ms=3,
                         color=colors[lab], lw=1.4 if lab == ref else 1.0)
        axs.axhline(0, color='k', alpha=0.5)
        for y in (-2, 2):
            axs.axhline(y, color='gray', ls='--', alpha=0.7)
        axs.set_ylim(-6, 6); axs.set_xlabel(r'$r/R_v$'); axs.grid(alpha=0.25)
        axs.axvline(1.0, color='gray', ls='--', alpha=0.6)
        if j == 0:
            ax.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
            axs.set_ylabel('S/N')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(axs.get_yticklabels(), visible=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plots] Catalogue columns saved to {output_path}')


def plot_act_vs_pr4(collected, output_path, max_Rvoid, success_label, other_label,
                    footprint_mask=None, footprint_coord='C'):
    if success_label not in collected or other_label not in collected:
        print('[Plots] Missing profiles for ACT vs PR4')
        return
    ds, do = collected[success_label], collected[other_label]
    r = np.asarray(ds['r_frac'])
    S, O = net_signal(ds), net_signal(do)
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[3, 1], hspace=0.06, wspace=0.22)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_p)
    ax_p.axhline(0, color='k', linestyle=':', alpha=0.6)
    ax_p.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    if S['band'] is not None:
        ax_p.fill_between(r, -S['band'] * 1e3, S['band'] * 1e3, color='xkcd:grey', alpha=0.30, zorder=1,
                          label=f'{success_label} ' + r'$1\sigma$ randoms')
    ax_p.errorbar(r, O['net'] * 1e3, yerr=O['err'] * 1e3, fmt='s-', color='xkcd:teal', capsize=3,
                  zorder=4, label=f"{other_label} (N={do.get('n_voids', '?')})")
    ax_p.errorbar(r, S['net'] * 1e3, yerr=S['err'] * 1e3, fmt='o-', color='k', capsize=3,
                  linewidth=2.0, zorder=5, label=f"{success_label} (N={ds.get('n_voids', '?')})")
    ax_p.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$')
    ax_p.grid(True, alpha=0.25); ax_p.legend(loc='lower right', frameon=True, fontsize=9)
    ax_p.tick_params(labelbottom=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = (S['net'] - O['net']) / np.sqrt(S['err'] ** 2 + O['err'] ** 2)
    ax_s.axhline(0, color='k', alpha=0.5)
    for y in (-2, 2):
        ax_s.axhline(y, color='gray', linestyle='--', alpha=0.7)
    ax_s.axhspan(-2, 2, color='gray', alpha=0.10, zorder=0)
    ax_s.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax_s.plot(r, sig, 'o-', color='xkcd:crimson', markersize=4)
    ax_s.set_ylim(-5, 5); ax_s.set_xlim(-0.1, max_Rvoid + 0.1)
    ax_s.set_xlabel(r'$r/R_v$'); ax_s.set_ylabel(r'$\Delta/\sigma$'); ax_s.grid(True, alpha=0.25)

    ax_m = fig.add_subplot(gs[:, 1])
    if footprint_mask is not None:
        plt.sca(ax_m)
        try:
            hp.mollview(footprint_mask, coord=[footprint_coord, 'G'], hold=True, cbar=False, cmap='Greys',
                        min=0, max=1, title='ACT footprint (galactic) + voids')
            hp.graticule(dpar=30, dmer=30, alpha=0.3)
            vla, vba = do.get('void_l'), do.get('void_b')
            vlp, vbp = ds.get('void_l'), ds.get('void_b')
            if vla is not None:
                hp.projscatter(np.asarray(vla), np.asarray(vba), lonlat=True, s=10, color='red', alpha=0.7)
            if vlp is not None:
                hp.projscatter(np.asarray(vlp), np.asarray(vbp), lonlat=True, s=32, facecolors='none',
                               edgecolors='blue', linewidths=0.7, alpha=0.9)
        except Exception as e:
            ax_m.text(0.5, 0.5, f'mollview failed:\n{e}', ha='center', va='center', transform=ax_m.transAxes)
    else:
        ax_m.set_axis_off()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[Plots] ACT vs PR4 panel saved to {output_path}')
