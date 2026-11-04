"""
diagnostic_plots.py
===================
Diagnostic figures of the inputs of a lensing run. The selection, map, masks,
coverage region and survey footprint are built with the SAME functions as
S1_voids_parallel and from the config of Pipeline_voids.py, so what is drawn is
what is stacked.

Output in {output_folder}/Plots/ (PNG):
  1. histograms of R_void and z of the selected sample(s);
  2. sky: voids (discs of their real size), coverage region (intersection of masks)
     and the region where random centres are drawn (coverage x survey footprint);
  3. the filter kernel actually applied (Wiener W_L or Gaussian b_L);
  4. the filtered convergence map, masked.

Only one map and one catalogue per call (groups/BOTH: set them here).
    python diagnostic_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp

import Functions_module as fm
import S1_voids_parallel as s1
from Pipeline_voids import config as PIPELINE_CONFIG

config = dict(PIPELINE_CONFIG)
config.update({
    'cmb_map': 'PLANCK_PR4',        # one map
    'void_catalog': 'WH',           # 'WH' | 'BOSS'
    'coverage_map': None,           # e.g. ['ACT'] to reproduce an ACT x PR4 selection
    # plot-only settings
    'size_factor': 1.0,             # disc radius in units of R_void
    'nside_disc': 512,
    'xsize': 1600,
    'nside_cmb_plot': 1024,
    'run_hist': True, 'run_sky': True, 'run_filter': True, 'run_cmb': True,
})


def _suffix(cfg, cat_label, map_label):
    return (f"{cat_label}_{map_label}_z{cfg['zmin']:g}-{cfg['zmax']:g}_r{cfg['rmin']:g}-{cfg['rmax']:g}_"
            f"{s1._delta_label(cfg.get('delta_value'))}")


def _proj(xsize):
    return hp.projector.MollweideProj(xsize=xsize)


def _img(m, proj):
    nside = hp.npix2nside(len(m))
    return np.asarray(proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z)), float)


def _graticule(ax, proj):
    for b0 in np.arange(-60, 61, 30):
        ll = np.linspace(-179.9, 179.9, 500)
        x, y = proj.ang2xy(ll, np.full_like(ll, float(b0)), lonlat=True)
        ax.plot(x, y, color='grey', lw=0.4, alpha=0.5)
    for l0 in np.arange(-180, 181, 30):
        bb = np.linspace(-89.9, 89.9, 500)
        x, y = proj.ang2xy(np.full_like(bb, float(l0)), bb, lonlat=True)
        ax.plot(x, y, color='grey', lw=0.4, alpha=0.5)


def plot_histograms(samples, cfg, out):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'hspace': 0.28})
    colors = {'single': 'xkcd:steel blue', 'concat': 'xkcd:steel blue', 'merged': 'xkcd:crimson'}
    for ax, col, lo, hi, lab in ((axes[0], 'R_void', cfg['rmin'], cfg['rmax'], r'$R_{\rm void}$ [Mpc/h]'),
                                 (axes[1], 'z', cfg['zmin'], cfg['zmax'], r'$z$')):
        edges = np.linspace(lo, hi, 26)
        for kind, df in samples.items():
            ax.hist(df[col].values, bins=edges, histtype='step', density=True, color=colors[kind], lw=1.6,
                    label=f"{kind} (N={len(df)}, med={np.median(df[col]):.3g})")
        ax.set_xlabel(lab); ax.set_ylabel('density'); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[plots] {out}')


def plot_sky(samples, cov_frac, region, cfg, out, title):
    proj = _proj(cfg['xsize'])
    ext = proj.get_extent()
    cov_img = _img(np.where(cov_frac >= 0.5, 1.0, 0.0), proj)
    reg_img = _img(region, proj)
    fig, axes = plt.subplots(1, len(samples), figsize=(9 * len(samples), 5.2), squeeze=False)
    for ax, (kind, df) in zip(axes[0], samples.items()):
        ax.imshow(np.where(np.isfinite(cov_img) & (cov_img < 0.5), 1.0, np.nan), extent=ext, origin='lower',
                  cmap=ListedColormap(['xkcd:grey']), vmin=0, vmax=1, alpha=0.45)
        ax.imshow(np.where(np.isfinite(reg_img) & (reg_img > 0.5), 1.0, np.nan), extent=ext, origin='lower',
                  cmap=ListedColormap(['xkcd:pale green']), vmin=0, vmax=1, alpha=0.6)
        m = np.zeros(hp.nside2npix(cfg['nside_disc']))
        for li, bi, zi, ri in zip(df['l'].values, df['b'].values, df['z'].values, df['R_void'].values):
            th = np.radians(fm.get_angularsize_comoving(zi, cfg['size_factor'] * ri))
            m[hp.query_disc(cfg['nside_disc'], hp.ang2vec(li, bi, lonlat=True), max(th, 1e-4))] += 1.0
        vimg = _img(m, proj)
        im = ax.imshow(np.where(np.isfinite(vimg) & (vimg > 0), vimg, np.nan), extent=ext, origin='lower',
                       cmap='turbo', vmin=1)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='overlapping discs')
        _graticule(ax, proj)
        ax.set_title(f'{kind}: N={len(df)}  | grey: outside coverage, green: random-centre region', fontsize=10)
        ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[plots] {out}')


def plot_filter(mspec, cfg, out):
    flabel, action = s1._effective_filter(mspec, cfg)
    if action == 'wiener':
        d = np.loadtxt(mspec['nlkk'])
        L, N_L, SN = d[:, 0].astype(int), d[:, 1], d[:, 2]
        Cl = np.maximum(SN - N_L, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.where(SN > 0, Cl / SN, 0.0)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(L, W, color='xkcd:crimson'); a1.set_xlabel(r'$L$'); a1.set_ylabel(r'$W_L$')
        a1.set_title(f"Wiener ({os.path.basename(mspec['nlkk'])})"); a1.grid(alpha=0.25)
        good = (L > 1) & (Cl > 0) & (N_L > 0)
        a2.loglog(L[good], Cl[good], label=r'$C_L^{\kappa\kappa}$'); a2.loglog(L[good], N_L[good], label=r'$N_L$')
        a2.legend(); a2.grid(alpha=0.25, which='both'); a2.set_xlabel(r'$L$')
    elif action == 'gaussian':
        fwhm = cfg['smooth_value_arcmin']
        L = np.arange(2049)
        fig, a1 = plt.subplots(figsize=(6.5, 4.5))
        a1.plot(L, hp.gauss_beam(np.radians(fwhm / 60.0), lmax=2048), color='xkcd:crimson')
        a1.set_xlabel(r'$L$'); a1.set_ylabel(r'$b_L$'); a1.set_title(f"Gaussian FWHM={fwhm:g}'")
    else:
        print(f'[plots] filter "{flabel}": no kernel applied by the pipeline.')
        return
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[plots] {out}')


def plot_cmb(lensing_map, stamp_mask, mspec, flabel, cfg, out):
    ns = cfg.get('nside_cmb_plot') or hp.get_nside(lensing_map)
    m = hp.ud_grade(lensing_map, ns) if ns != hp.get_nside(lensing_map) else lensing_map.copy()
    msk = hp.ud_grade(stamp_mask, ns) if ns != hp.get_nside(stamp_mask) else stamp_mask
    m[msk < 0.5] = np.nan
    proj = _proj(cfg['xsize'])
    img = _img(m, proj)
    vlim = np.nanpercentile(np.abs(img[np.isfinite(img)]), 99)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(img, extent=proj.get_extent(), origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r'$\kappa$')
    _graticule(ax, proj)
    ax.set_title(f"{mspec['label']} ({flabel}), native frame {mspec['frame']}", fontsize=12)
    ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[plots] {out}')


def main(cfg=config):
    out_dir = os.path.join(cfg['output_folder'], 'Plots')
    os.makedirs(out_dir, exist_ok=True)
    df_ = cfg['data_folder']
    cat = s1._catalog_spec(cfg['void_catalog'], df_)
    mspec = s1._map_spec(cfg['cmb_map'], df_, cfg.get('act_smooth_arcmin', 0.0))
    suf = _suffix(cfg, cat['label'], mspec['label'])

    keys = s1._coverage_keys(cfg, cfg['cmb_map'])
    cov_frac, cov_labels = s1._coverage_region(keys, df_, cfg['coverage_nside'], cfg.get('act_smooth_arcmin', 0.0))
    region = np.where(cov_frac >= 0.5, 1.0, 0.0)
    if cfg.get('random_pool', 'footprint') == 'footprint':
        F = fm.survey_footprint(cat.get('footprint'), df_, cfg['coverage_nside'],
                                os.path.join(cfg['output_folder'], 'lensing', 'Cache_Stacks', 'footprints'),
                                label=cat['label'])
        if F is not None:
            region = region * F

    samples = None
    if cfg['run_hist'] or cfg['run_sky']:
        S = s1.load_void_samples(cfg, cat, cov_frac)
        samples = {}
        if S['single'] is not None:
            samples['single'] = S['single']
        if S['per_seed'] is not None and cfg.get('seed_mode') in ('concat', 'both'):
            import pandas as pd
            samples['concat'] = pd.concat(S['per_seed'].values(), ignore_index=True)
        if S['merged'] is not None:
            samples['merged'] = S['merged']
    if cfg['run_hist']:
        plot_histograms(samples, cfg, os.path.join(out_dir, f'hist_R_z_{suf}.png'))
    if cfg['run_sky']:
        plot_sky(samples, cov_frac, region, cfg, os.path.join(out_dir, f'sky_{suf}.png'),
                 f"{cat['label']} voids | coverage: {' x '.join(cov_labels)} | "
                 f"randoms: {cfg.get('random_pool', 'footprint')}")
    if cfg['run_filter']:
        plot_filter(mspec, cfg, os.path.join(out_dir, f'filter_{suf}.png'))
    if cfg['run_cmb']:
        m, nside, flabel = s1._load_map(mspec, cfg)
        plot_cmb(m, fm.binary_mask(mspec['mask'], nside), mspec, flabel, cfg,
                 os.path.join(out_dir, f'cmb_map_{suf}.png'))
    print('\n[plots] === DONE ===')


if __name__ == '__main__':
    main()
