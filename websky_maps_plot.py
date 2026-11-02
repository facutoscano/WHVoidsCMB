"""
websky_maps_plot.py
===================
Figura 2x2 de Websky:
  (1) voids de las N semillas FUSIONADOS con el mismo DBSCAN de la pipeline
      (void_seed_merge.merge_seeds; cortes z, R, completeness por semilla, igual
      que S1) -> discos de su tamaño angular real, o centros coloreados por z.
  (2) kappa    (3) tSZ (Compton-y, escala log)    (4) ISW
Los mapas se llevan a nside_plot (2048) con ud_grade SÓLO si vienen más finos
(kap.fits suele estar en 4096); si vienen en <= nside_plot se dejan como están.

Todo está en el frame cartesiano de la simulación (el mismo de (l,b) de los voids):
no es galáctico, la grilla es sólo de referencia.

Proyección Mollweide hecha a mano (healpy.projector.MollweideProj), igual que en
diagnostic_plots.py, para no depender de que hp.mollview abra su propia figura.

    python websky_maps_plot.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

config = {
    'websky_folder': '/home/ftoscano/Doctorado/Data/MOCKS/Websky_data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/websky/Plots/',

    # ---- voids ----
    'seed_glob': 'websky_voids/*.dat',  # None -> usa 'single_file'
    'single_file': 'voids_d-0.9.dat',
    'N_seeds': 100,
    'zmin': 0.1, 'zmax': 0.5,
    'rmin': 35.0, 'rmax': 70.0,
    'completeness_min': 1.9,            # None -> sin corte
    'cols': {'R_void': 0, 'l': 1, 'b': 2, 'z': 3, 'x': 4, 'y': 5, 'z_cart': 6,
             'delta_23': 8, 'completeness': 11},                       # índices 0-based

    # ---- merge (mismos valores que en la pipeline) ----
    'merge_eps_mpch': 8.0,
    'merge_min_frac': 0.4,
    'merge_use_catalog_xyz': True,      # xyz del catálogo (exactos en la cosmología de Websky).
                                        # False recomputaría con Planck18 de vsm (dif. ~1e-3 Mpc/h)

    'void_mode': 'discs',               # 'discs' (tamaño real, color = nº de discos superpuestos) |
                                        # 'centers' (centros coloreados por z)
    'size_factor': 1.0,                 # radio del disco en unidades de R_void
    'nside_disc': 1024,

    # ---- mapas ----
    'maps': {'kappa': 'CMB/kap.fits', 'tSZ': 'CMB/tsz_2048.fits', 'ISW': 'CMB/isw.fits'},
    'nside_plot': 2048,
    'smooth_arcmin': {'kappa': 0.0, 'tSZ': 0.0, 'ISW': 0.0},   # sólo para mostrar

    'xsize': 2000,                      # ancho de la proyección [px]
}


# ----------------------------------------------------------------------
# Voids
# ----------------------------------------------------------------------
def load_voids_merged(cfg):
    wf, c = cfg['websky_folder'], cfg['cols']
    if cfg.get('seed_glob'):
        files = sorted(glob.glob(os.path.join(wf, cfg['seed_glob'])))
        if len(files) < cfg['N_seeds']:
            raise FileNotFoundError(f"{len(files)} archivos con '{cfg['seed_glob']}' "
                                    f"< N_seeds={cfg['N_seeds']}")
        files = files[:cfg['N_seeds']]
    else:
        files = [os.path.join(wf, cfg['single_file'])]

    frames = []
    for k, f in enumerate(files):
        raw = pd.read_csv(f, sep=r'\s+', header=None, comment='#')
        df = pd.DataFrame({name: raw[j].values.astype(float) for name, j in c.items()})
        m = ((df['z'] >= cfg['zmin']) & (df['z'] < cfg['zmax']) &
             (df['R_void'] >= cfg['rmin']) & (df['R_void'] <= cfg['rmax']))
        if cfg.get('completeness_min') is not None and 'completeness' in df:
            m &= df['completeness'] > cfg['completeness_min']
        frames.append(df[m].assign(seed=k))
    cat = pd.concat(frames, ignore_index=True)
    per_seed = cat.groupby('seed').size().reindex(range(len(files)), fill_value=0)
    print(f"[voids] {len(files)} catálogos -> {len(cat)} detecciones "
          f"(por semilla: media={per_seed.mean():.1f}, min={per_seed.min()}, max={per_seed.max()})")
    if len(files) == 1:
        print("[voids] una sola semilla: no hay nada que fusionar.")
        return cat, 1, per_seed.mean()

    import void_seed_merge as vsm
    merged, _ = vsm.merge_seeds(cat, eps_mpch=cfg['merge_eps_mpch'],
                                min_frac=cfg['merge_min_frac'], n_seeds=len(files),
                                use_catalog_xyz=cfg['merge_use_catalog_xyz'])
    vsm.sanity_report(merged)
    print(f"[voids] merge: {len(merged)} voids; med R={merged['R_void'].median():.1f} Mpc/h, "
          f"med z={merged['z'].median():.3f}")
    return merged, len(files), per_seed.mean()


# ----------------------------------------------------------------------
# Mapas
# ----------------------------------------------------------------------
def load_map(path, nside_plot, smooth_arcmin=0.0):
    import healpy as hp
    m = hp.read_map(path, field=0, dtype=np.float64)
    ns = hp.get_nside(m)
    print(f"[map] {os.path.basename(path)}: nside={ns}, mean={np.mean(m):.3e}, std={np.std(m):.3e}")
    if ns > nside_plot:
        m = hp.ud_grade(m, nside_out=nside_plot)
        print(f"[map]   ud_grade {ns} -> {nside_plot}")
    if smooth_arcmin and smooth_arcmin > 0:
        m = hp.smoothing(m, fwhm=np.radians(smooth_arcmin / 60.0))
    return m


def project(m, xsize):
    import healpy as hp
    nside = hp.npix2nside(len(m))
    proj = hp.projector.MollweideProj(xsize=xsize)
    img = np.asarray(proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z)), float)
    img[~np.isfinite(img) | (img == hp.UNSEEN)] = np.nan
    return img, proj


def void_image(cat, cfg, h=0.68, Om=0.31):
    """Discos de radio size_factor*R_void (ángulo con chi(z) de Websky), contando
    cuántos discos cubren cada píxel."""
    import healpy as hp
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=Om, Tcmb0=0.0)
    ns = cfg['nside_disc']
    m = np.zeros(hp.nside2npix(ns))
    chi = cosmo.comoving_distance(cat['z'].values).value * h          # Mpc/h
    theta = cfg['size_factor'] * cat['R_void'].values / chi           # rad
    vec = hp.ang2vec(cat['l'].values, cat['b'].values, lonlat=True)
    for v, t in zip(vec, theta):
        m[hp.query_disc(ns, v, max(t, 1e-4))] += 1.0
    img, proj = project(m, cfg['xsize'])
    inside = np.isfinite(img)
    img[inside & (img <= 0)] = np.nan          # sólo donde hay voids
    return img, inside, proj


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def _graticule(ax, proj):
    for b0 in np.arange(-60, 61, 30):
        ll = np.linspace(-179.9, 179.9, 400)
        x, y = proj.ang2xy(ll, np.full_like(ll, float(b0)), lonlat=True)
        ax.plot(x, y, color='grey', lw=0.3, alpha=0.5)
    for l0 in np.arange(-180, 181, 60):
        bb = np.linspace(-89.9, 89.9, 400)
        x, y = proj.ang2xy(np.full_like(bb, float(l0)), bb, lonlat=True)
        ax.plot(x, y, color='grey', lw=0.3, alpha=0.5)


def _sym_limits(img, pct=99.0):
    v = np.nanpercentile(np.abs(img), pct)
    return -v, v


def _log_limits(img):
    pos = img[np.isfinite(img) & (img > 0)]
    if pos.size == 0:
        return 1e-10, 1e-9
    return np.percentile(pos, 50), np.percentile(pos, 99.9)


def make_figure(cfg, panels, out_path, title):
    """
    panels: lista de 4 dicts {'img','extent','kind','label','title','proj', ...}
    kind in {'voids','sym','log'}.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.2))
    for ax, p in zip(axes.ravel(), panels):
        img, ext = p['img'], p['extent']
        if p['kind'] == 'voids':
            ax.imshow(np.where(p['inside'], 1.0, np.nan), extent=ext, origin='lower',
                      cmap='Greys', vmin=0, vmax=6)                    # fondo gris claro
            if p.get('centers') is not None:
                x, y, zc = p['centers']
                im = ax.scatter(x, y, s=1.5, c=zc, cmap='viridis', lw=0)
            else:
                vmax = max(2, np.nanmax(img)) if np.any(np.isfinite(img)) else 2
                im = ax.imshow(img, extent=ext, origin='lower', cmap='turbo', vmin=1, vmax=vmax)
        elif p['kind'] == 'log':
            lo, hi = _log_limits(img)
            im = ax.imshow(np.clip(img, lo, None), extent=ext, origin='lower',
                           cmap='inferno', norm=LogNorm(vmin=lo, vmax=hi))
        else:
            lo, hi = _sym_limits(img)
            im = ax.imshow(img, extent=ext, origin='lower', cmap='RdBu_r', vmin=lo, vmax=hi)
        if im is not None:
            plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03,
                         label=p['label'])
        _graticule(ax, p['proj'])
        ax.set_title(p['title'], fontsize=12)
        ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] guardado {out_path}")


def main(cfg=config):
    os.makedirs(cfg['output_folder'], exist_ok=True)
    cat, nfiles, per_seed = load_voids_merged(cfg)

    title_v = (f"Voids merge ({nfiles} semillas, eps={cfg['merge_eps_mpch']:g}, "
               f"min_frac={cfg['merge_min_frac']:g}): N={len(cat)}  [{per_seed:.0f}/semilla]")
    if cfg['void_mode'] == 'centers':
        import healpy as hp
        proj = hp.projector.MollweideProj(xsize=cfg['xsize'])
        xs, ys = proj.ang2xy(cat['l'].values, cat['b'].values, lonlat=True)
        bg, _ = project(np.ones(hp.nside2npix(8)), cfg['xsize'])
        vp = {'img': None, 'inside': np.isfinite(bg), 'centers': (xs, ys, cat['z'].values),
              'label': 'z'}
    else:
        img, inside, proj = void_image(cat, cfg)
        vp = {'img': img, 'inside': inside,
              'label': f"discos superpuestos ({cfg['size_factor']:g}" + r"$\times R_v$)"}
    vp.update({'extent': proj.get_extent(), 'proj': proj, 'kind': 'voids', 'title': title_v})

    panels = [vp]
    styles = {'kappa': ('sym', r'$\kappa$'),
              'tSZ':   ('log', r'Compton-$y$'),
              'ISW':   ('sym', 'ISW (unidades del archivo)')}
    for name, rel in cfg['maps'].items():
        m = load_map(os.path.join(cfg['websky_folder'], rel), cfg['nside_plot'],
                     cfg['smooth_arcmin'].get(name, 0.0))
        mimg, mproj = project(m, cfg['xsize'])
        del m
        kind, lab = styles.get(name, ('sym', name))
        sm = cfg['smooth_arcmin'].get(name, 0.0)
        panels.append({'img': mimg, 'extent': mproj.get_extent(), 'proj': mproj,
                       'kind': kind, 'label': lab,
                       'title': f"{name}  (nside {cfg['nside_plot']}"
                                + (f", FWHM {sm:g}')" if sm else ")")})

    tag = (f"z{cfg['zmin']:g}-{cfg['zmax']:g}_r{cfg['rmin']:g}-{cfg['rmax']:g}_"
           f"{nfiles}seeds_merge_eps{cfg['merge_eps_mpch']:g}_mf{cfg['merge_min_frac']:g}_"
           f"{cfg['void_mode']}")
    title = (rf"Websky  |  ${cfg['zmin']:g}\leq z<{cfg['zmax']:g}$,  "
             rf"${cfg['rmin']:g}\leq R_v\leq{cfg['rmax']:g}$ Mpc/h  |  frame de la simulación")
    make_figure(cfg, panels, os.path.join(cfg['output_folder'], f"websky_voids_maps_{tag}.png"), title)


if __name__ == "__main__":
    main()
