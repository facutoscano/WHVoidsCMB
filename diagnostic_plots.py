"""
diagnostic_plots.py
===================
Plots diagnósticos (no de ciencia fina, sólo para mostrar la selección/insumos)
de la corrida de lensing de Pipeline_voids.py. Reutiliza EXACTAMENTE la misma
carga/filtrado/merge que S1_voids_parallel.run_pipeline, así lo que se dibuja es
la misma muestra y el mismo mapa que efectivamente se apila.

Genera, dentro de {output_folder}/Plots/  (todo en PNG, con sufijo de config
en el nombre para que distintas corridas no se pisen):
  1. Dos filas de histogramas: R_void (rmin-rmax) y z (zmin-zmax), con el
     catálogo y el corte en delta_23 en el título.
  2. Distribución en el cielo (Mollweide, l,b) con el tamaño angular real:
       - seed_mode con 'merge'  -> voids finales como DISCOS de su tamaño real (un color).
       - seed_mode con 'concat' -> mapa de DENSIDAD de centros (sólo donde >0).
     La máscara del mapa de CMB se sombrea con alpha (región excluida en gris).
  3. Si hay filtro: el kernel aplicado (W_ell del Wiener, o b_ell del gaussiano).
  4. El mapa de CMB usado (ya filtrado) enmascarado, en el nside del config.

El render del cielo/CMB se hace proyectando a mano con healpy.projector.MollweideProj
(NO con hp.mollview) para controlar capas y transparencia y evitar el bug de que
hp.mollview abre su propia figura.

Corre con el MISMO numpy/healpy con el que corre el pipeline:
    python diagnostic_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless: sólo escribe PNGs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

import Functions_module as fm              # apply_wiener_filter, get_angularsize_comoving
import void_seed_merge as vsm             # merge_seeds


# ======================================================================
#  CONFIG  (copiado de Pipeline_voids.py, sólo lo relevante para plotear)
# ======================================================================
config = {
    'data_folder':   '/home/ftoscano/Doctorado/Data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/New_Results/',

    # --- selección de datos (igual a Pipeline_voids) ---
    'release':       'PR4',        # 'PR3' o 'PR4'
    'void_catalog':  'BOSS',       # 'WH' (l,b galácticas) o 'BOSS' (ra,dec -> l,b)
    'N_seeds':       100,          # None -> catálogo único; si no, N semillas
    'delta_value':   None,        # None: sin corte | >0: delta_23 > v | <0: delta_23 < v

    # --- manejo multi-seed (idéntico al pipeline) ---
    'seed_mode':     'both',      # 'concat' | 'merge' | 'both'
    'merge_eps_mpch':        8.0,
    'merge_min_frac':        0.4,
    'merge_use_catalog_xyz': False,

    'zmin': 0.1, 'zmax': 0.5,
    'rmin': 30.0, 'rmax': 70.0,    # Mpc/h

    # --- geometría / filtro (igual al pipeline) ---
    'max_Rvoid':          2.5,
    'npix_stamp':         400,
    'filter_mode':        'wiener',   # 'none' | 'gaussian' | 'wiener'
    'smooth_value_arcmin': 0.0,       # sólo si filter_mode == 'gaussian'

    # --- SÓLO del plotter (no afectan la ciencia) ---
    'nside':          2048,   # el pipeline lo hardcodea a 2048; acá se respeta el config
    'size_factor':    1.0,    # radio de los discos en unidades de R_void (1.0 = tamaño real del void)
    'nside_disc':     1024,   # nside para pintar los discos (barato; 2048 es innecesario)
    'nside_weight':   128,    # nside GRUESO para el mapa de densidad de centros (concat)
    'xsize':          1600,   # ancho en px de la proyección Mollweide
    'mask_alpha':     0.35,   # transparencia del sombreado de la región enmascarada
    'nside_cmb_plot': None,   # None -> usa 'nside'; bajalo (p.ej. 1024) para acelerar

    # --- qué plots correr ---
    'run_hist':   True,
    'run_sky':    True,
    'run_filter': True,
    'run_cmb':    True,
}


# ======================================================================
#  Etiqueta de config para los nombres de archivo
# ======================================================================
def _suffix(config):
    dv = config.get('delta_value')
    if dv is None:
        dtag = 'd23all'
    elif dv >= 0:
        dtag = f'd23gt{dv:g}'
    else:
        dtag = f'd23lt{abs(dv):g}'
    filt = config.get('filter_mode', 'none')
    if filt == 'gaussian':
        filt = f'gauss{config.get("smooth_value_arcmin", 0):g}am'
    return (f"{config['void_catalog']}_{config['release']}_"
            f"z{config['zmin']:g}-{config['zmax']:g}_"
            f"r{config['rmin']:g}-{config['rmax']:g}_"
            f"{dtag}_{filt}_{config.get('seed_mode', 'na')}")


# ======================================================================
#  Helpers de catálogo  (COPIADOS de S1_voids_parallel para consistencia)
# ======================================================================
def _catalog_spec(catalog, data_folder):
    label = (catalog or 'WH').upper()
    if label == 'BOSS':
        return {
            'label': 'BOSS',
            'seed_template': f'{data_folder}/CATALOGOS/BOSS_voids_merge/voids_boss_{{:03d}}.dat',
            'single_file':   f'{data_folder}/CATALOGOS/BOSS_voids.dat',
            'lon_col': 'ra', 'lat_col': 'dec', 'frame': 'icrs',
        }
    return {
        'label': 'WH',
        'seed_template': f'{data_folder}/CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{{:03d}}.dat',
        'single_file':   f'{data_folder}/CATALOGOS/WenHan_voids.dat',
        'lon_col': 'l', 'lat_col': 'b', 'frame': 'galactic',
    }


def _ensure_galactic(df, cat):
    if cat['frame'] == 'galactic':
        return df
    c = SkyCoord(ra=df[cat['lon_col']].values * u.degree,
                 dec=df[cat['lat_col']].values * u.degree, frame='icrs').galactic
    df = df.copy()
    df['l'] = c.l.degree
    df['b'] = c.b.degree
    return df


def _apply_delta_23_filter(df, delta_value):
    if delta_value is None:
        return df
    elif delta_value >= 0:                  # ojo: == 0 cae en la rama '>'
        return df[df['delta_23'] > delta_value]
    else:
        return df[df['delta_23'] < delta_value]


def _delta_text(delta_value):
    if delta_value is None:
        return r'$\delta_{23}$: sin corte'
    if delta_value >= 0:
        return rf'$\delta_{{23}} > {delta_value:g}$'
    return rf'$\delta_{{23}} < {delta_value:g}$'


# ======================================================================
#  Construcción de las muestras  (misma lógica que run_pipeline)
# ======================================================================
def build_samples(config, cat):
    """Devuelve {kind: DataFrame} con kind in {'single','concat','merged'}."""
    zmin, zmax = config['zmin'], config['zmax']
    rmin, rmax = config['rmin'], config['rmax']
    n_seeds = config.get('N_seeds', None)
    delta_value = config.get('delta_value', None)

    if n_seeds is None:
        col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z']
        raw = pd.read_csv(cat['single_file'], sep=r'\s+', names=col_names, header=None)
        raw = _ensure_galactic(raw, cat)
        sample = raw[(raw['z'] >= zmin) & (raw['z'] < zmax) &
                     (raw['R_void'] >= rmin) & (raw['R_void'] <= rmax)].copy()
        print(f'[plots] catálogo único: {len(sample)} voids.')
        return {'single': sample}

    col_names = ['R_void', cat['lon_col'], cat['lat_col'], 'z', 'x', 'y', 'z_cart',
                 'delta_int', 'delta_23', 'completeness', 'delta_LOS']
    frames = []
    for seed in range(n_seeds):
        df = pd.read_csv(cat['seed_template'].format(seed + 1), sep=r'\s+',
                         names=col_names, header=None)
        df = _ensure_galactic(df, cat)
        base = ((df['z'] >= zmin) & (df['z'] < zmax) &
                (df['R_void'] >= rmin) & (df['R_void'] <= rmax) &
                (df['completeness'] > 1.9))
        df = _apply_delta_23_filter(df[base].copy(), delta_value)
        df['seed'] = seed
        frames.append(df)
    concat_all = pd.concat(frames, ignore_index=True)
    print(f'[plots] concat multi-seed: {len(concat_all)} detecciones '
          f'(con duplicados, pesadas por multiplicidad).')

    samples = {}
    seed_mode = config.get('seed_mode', 'concat')
    if seed_mode in ('concat', 'both'):
        samples['concat'] = concat_all
    if seed_mode in ('merge', 'both'):
        merged, _ = vsm.merge_seeds(concat_all,
                                    eps_mpch=config['merge_eps_mpch'],
                                    min_frac=config['merge_min_frac'],
                                    n_seeds=n_seeds,
                                    use_catalog_xyz=config['merge_use_catalog_xyz'])
        samples['merged'] = merged
        print(f'[plots] merge DBSCAN: {len(merged)} voids únicos.')
    return samples


# ======================================================================
#  Mapa de CMB filtrado + máscara  (misma construcción que S1)
# ======================================================================
def load_cmb_map_and_mask(config):
    data_folder = config['data_folder']
    release = config['release']
    nside = config['nside']

    klm_file = f'{data_folder}CMB/Lensing/KAPPA_{release}klm_MV.fits'
    common_mask_file = f'{data_folder}CMB/Lensing/Common_mask_PR4Lensing_2048.fits'  # hardcodeado PR4 (igual que S1)

    print(f'[plots] leyendo alm de CMB: {klm_file}')
    cmb_alm = hp.fitsfunc.read_alm(klm_file, hdu=1, return_mmax=False)
    common_mask = hp.read_map(common_mask_file)

    filter_mode = config.get('filter_mode', 'none')
    W = None
    if filter_mode == 'wiener':
        nlkk_file = f'{data_folder}CMB/Lensing/nlkk_PR4_MV.dat'                       # hardcodeado PR4 (igual que S1)
        cmb_alm_f, W = fm.apply_wiener_filter(cmb_alm, nlkk_file, lmax=2048)
        lensing_map = hp.alm2map(cmb_alm_f, nside=nside)
        print('[plots] mapa filtrado con Wiener.')
    elif filter_mode == 'gaussian' and config.get('smooth_value_arcmin', 0.0) > 0:
        fwhm_deg = config['smooth_value_arcmin'] / 60.0
        lensing_map = hp.smoothing(hp.alm2map(cmb_alm, nside=nside),
                                   fwhm=np.radians(fwhm_deg))
        print(f'[plots] mapa suavizado con gaussiana FWHM={fwhm_deg:.3f} deg.')
    else:
        lensing_map = hp.alm2map(cmb_alm, nside=nside)
        print('[plots] mapa sin filtro adicional.')
    return lensing_map, common_mask, W


# ======================================================================
#  Proyección Mollweide manual (control total de capas / alpha)
# ======================================================================
def _make_proj(xsize):
    return hp.projector.MollweideProj(xsize=xsize)


def _sky_image(m, nside, proj):
    """Proyecta un mapa healpix a la grilla 2D de Mollweide (float; nan/-inf fuera)."""
    vec2pix = lambda x, y, z: hp.vec2pix(nside, x, y, z)
    return np.asarray(proj.projmap(m, vec2pix), dtype=float)


def _draw_graticule(ax, proj, dmer=30, dpar=30):
    for b0 in np.arange(-60, 61, dpar):
        ll = np.linspace(-179.9, 179.9, 500)
        x, y = proj.ang2xy(ll, np.full_like(ll, float(b0)), lonlat=True)
        ax.plot(x, y, color='grey', lw=0.4, alpha=0.5)
    for l0 in np.arange(-180, 181, dmer):
        bb = np.linspace(-89.9, 89.9, 500)
        x, y = proj.ang2xy(np.full_like(bb, float(l0)), bb, lonlat=True)
        ax.plot(x, y, color='grey', lw=0.4, alpha=0.5)


def _mask_excluded_image(common_mask, proj):
    """Capa 2D = 1 en la región EXCLUIDA por la máscara (dentro de la elipse), nan si no."""
    nside_mask = hp.npix2nside(len(common_mask))
    raw = _sky_image(common_mask, nside_mask, proj)
    return np.where(np.isfinite(raw) & (raw < 0.5), 1.0, np.nan)


def _paint_disc_map(l, b, z, rv, nside, size_factor):
    """Mapa healpix con cada void pintado como disco de su tamaño angular real."""
    m = np.zeros(hp.nside2npix(nside))
    for i in range(len(l)):
        theta_deg = fm.get_angularsize_comoving(z[i], size_factor * rv[i])
        vec = hp.ang2vec(l[i], b[i], lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(max(theta_deg, 1e-3)))
        m[pix] += 1.0
    return m


def _paint_centers_map(l, b, nside):
    """Densidad de centros: cuántos caen en cada píxel (colapsa la LOS)."""
    m = np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, l, b, lonlat=True)
    np.add.at(m, pix, 1.0)
    return m


# ======================================================================
#  PLOT 1 — histogramas de R_void y z
# ======================================================================
def plot_histograms(samples, config, out_dir, cat_label):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'hspace': 0.28})
    colors = {'single': 'xkcd:steel blue', 'concat': 'xkcd:steel blue',
              'merged': 'xkcd:crimson'}
    labels = {'single': 'catálogo único', 'concat': 'concat (multi-seed)',
              'merged': 'merged (DBSCAN)'}

    ax = axes[0]
    r_edges = np.linspace(config['rmin'], config['rmax'], 26)
    for kind, df in samples.items():
        ax.hist(df['R_void'].values, bins=r_edges, histtype='step', density=True,
                color=colors.get(kind, 'k'), lw=1.6,
                label=f"{labels.get(kind, kind)}  (N={len(df)}, med={np.median(df['R_void']):.1f})")
    ax.axvline(config['rmin'], color='grey', ls='--', alpha=0.6)
    ax.axvline(config['rmax'], color='grey', ls='--', alpha=0.6)
    ax.set_xlabel(r'$R_{\rm void}$  [Mpc/h]'); ax.set_ylabel('densidad')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[1]
    z_edges = np.linspace(config['zmin'], config['zmax'], 26)
    for kind, df in samples.items():
        ax.hist(df['z'].values, bins=z_edges, histtype='step', density=True,
                color=colors.get(kind, 'k'), lw=1.6,
                label=f"{labels.get(kind, kind)}  (N={len(df)}, med={np.median(df['z']):.3f})")
    ax.axvline(config['zmin'], color='grey', ls='--', alpha=0.6)
    ax.axvline(config['zmax'], color='grey', ls='--', alpha=0.6)
    ax.set_xlabel(r'$z$'); ax.set_ylabel('densidad')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    fig.suptitle(f"{cat_label} voids   |   {_delta_text(config.get('delta_value'))}",
                 y=0.94, fontsize=13)
    out = os.path.join(out_dir, f'hist_R_z_{_suffix(config)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  PLOT 2 — distribución en el cielo con tamaño real + máscara con alpha
# ======================================================================
def _draw_sky_panel(ax, void_img, mask_img, proj, title, mode, cbar_label, mask_alpha):
    ext = proj.get_extent()
    if mode == 'concat':                       # densidad de centros (sólo >0)
        im = ax.imshow(void_img, extent=ext, origin='lower', cmap='turbo', vmin=1)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    else:                                      # merged/single: disco plano, un color
        ax.imshow(void_img, extent=ext, origin='lower',
                  cmap=ListedColormap(['xkcd:crimson']), vmin=0, vmax=1)
    # máscara: región excluida en gris translúcido (por encima, para ver solapes)
    ax.imshow(mask_img, extent=ext, origin='lower',
              cmap=ListedColormap(['grey']), vmin=0, vmax=1, alpha=mask_alpha)
    _draw_graticule(ax, proj)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal'); ax.axis('off')


def plot_sky_distribution(samples, common_mask, config, out_dir, cat_label):
    proj = _make_proj(config['xsize'])
    mask_img = _mask_excluded_image(common_mask, proj)

    panels = []
    for kind, df in samples.items():
        if kind == 'concat':
            cm = _paint_centers_map(df['l'].values, df['b'].values, config['nside_weight'])
            raw = _sky_image(cm, config['nside_weight'], proj)
            vimg = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
            panels.append((vimg, 'concat',
                           f'{cat_label} — densidad de centros', 'detecciones/píxel'))
        else:
            dm = _paint_disc_map(df['l'].values, df['b'].values,
                                 df['z'].values, df['R_void'].values,
                                 config['nside_disc'], config['size_factor'])
            raw = _sky_image(dm, config['nside_disc'], proj)
            vimg = np.where(np.isfinite(raw) & (raw > 0), 1.0, np.nan)
            sf = config['size_factor']
            panels.append((vimg, 'merged',
                           f'{cat_label} — voids ({sf:g}' + r'$\times R_v$)', None))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5.2), squeeze=False)
    for ax, (vimg, mode, title, cbar) in zip(axes[0], panels):
        _draw_sky_panel(ax, vimg, mask_img, proj, title, mode, cbar, config['mask_alpha'])
    fig.suptitle(_delta_text(config.get('delta_value')), y=1.02, fontsize=12)
    out = os.path.join(out_dir, f'sky_distribution_{_suffix(config)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  PLOT 3 — kernel del filtro (Wiener o gaussiano)
# ======================================================================
def plot_filter_kernel(config, out_dir):
    filter_mode = config.get('filter_mode', 'none')
    data_folder = config['data_folder']

    if filter_mode == 'wiener':
        nlkk_file = f'{data_folder}CMB/Lensing/nlkk_PR4_MV.dat'
        d = np.loadtxt(nlkk_file)
        L = d[:, 0].astype(int); N_L = d[:, 1]; SN = d[:, 2]
        Cl = np.maximum(SN - N_L, 0.0)          # igual que fm.apply_wiener_filter
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.where(SN > 0, Cl / SN, 0.0)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(L, W, color='xkcd:crimson', lw=1.8)
        a1.set_xlabel(r'$\ell$'); a1.set_ylabel(r'$W_\ell$')
        a1.set_title(r'Wiener  $W_\ell = C_\ell^{\kappa\kappa}/(C_\ell^{\kappa\kappa}+N_\ell)$')
        a1.set_xlim(0, len(L)); a1.set_ylim(-0.02, 1.02); a1.grid(alpha=0.25)

        good = (L > 1) & (Cl > 0) & (N_L > 0)
        a2.loglog(L[good], Cl[good], color='xkcd:steel blue', lw=1.6, label=r'$C_\ell^{\kappa\kappa}$ (fiducial)')
        a2.loglog(L[good], N_L[good], color='xkcd:orange', lw=1.6, label=r'$N_\ell^{\kappa\kappa}$')
        a2.set_xlabel(r'$\ell$'); a2.set_ylabel(r'$C_\ell$')
        a2.set_title('Señal vs ruido'); a2.legend(); a2.grid(alpha=0.25, which='both')
        out = os.path.join(out_dir, f'filter_kernel_wiener_{_suffix(config)}.png')

    elif filter_mode == 'gaussian' and config.get('smooth_value_arcmin', 0.0) > 0:
        fwhm_deg = config['smooth_value_arcmin'] / 60.0
        lmax = 2048
        b_ell = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax)
        L = np.arange(lmax + 1)
        fig, a1 = plt.subplots(1, 1, figsize=(6.5, 4.5))
        a1.plot(L, b_ell, color='xkcd:crimson', lw=1.8)
        a1.set_xlabel(r'$\ell$'); a1.set_ylabel(r'$b_\ell$')
        a1.set_title(rf'Gaussiano  FWHM$={config["smooth_value_arcmin"]:g}\prime$ ($={fwhm_deg:.3f}^\circ$)')
        a1.set_xlim(0, lmax); a1.grid(alpha=0.25)
        out = os.path.join(out_dir, f'filter_kernel_gaussian_{_suffix(config)}.png')

    else:
        print('[plots] filter_mode sin filtro efectivo: no hay kernel que plotear.')
        return

    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  PLOT 4 — mapa de CMB (ya filtrado) enmascarado, en el nside del config
# ======================================================================
def plot_cmb_map(lensing_map, common_mask, config, out_dir):
    nside_plot = config.get('nside_cmb_plot') or config['nside']
    m, mask = lensing_map, common_mask
    if nside_plot != config['nside']:
        m = hp.ud_grade(m, nside_out=nside_plot)
        mask = hp.ud_grade(mask, nside_out=nside_plot)

    disp = m.copy()
    disp[mask < 0.5] = np.nan                    # enmascara ANTES de proyectar
    proj = _make_proj(config['xsize'])
    img = _sky_image(disp, nside_plot, proj)
    good = np.isfinite(img)
    vlim = np.nanpercentile(np.abs(img[good]), 99) if good.any() else 1.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ext = proj.get_extent()
    im = ax.imshow(img, extent=ext, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r'$\kappa$')
    _draw_graticule(ax, proj)
    label = {'wiener': 'Wiener', 'gaussian': 'gaussiano',
             'none': 'sin filtro'}.get(config.get('filter_mode', 'none'), config.get('filter_mode'))
    ax.set_title(rf"$\kappa$ ({config['release']}, {label}) — nside {nside_plot}", fontsize=12)
    ax.set_aspect('equal'); ax.axis('off')
    out = os.path.join(out_dir, f'cmb_map_{_suffix(config)}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  MAIN
# ======================================================================
def main():
    out_dir = os.path.join(config['output_folder'], 'Plots')
    os.makedirs(out_dir, exist_ok=True)
    cat = _catalog_spec(config['void_catalog'], config['data_folder'])
    cat_label = cat['label']
    print(f'[plots] catálogo = {cat_label} | seed_mode = {config.get("seed_mode")} '
          f'| filtro = {config.get("filter_mode")}')

    samples = None
    if config['run_hist'] or config['run_sky']:
        samples = build_samples(config, cat)

    if config['run_hist']:
        plot_histograms(samples, config, out_dir, cat_label)

    lensing_map = common_mask = None
    if config['run_sky'] or config['run_cmb']:
        lensing_map, common_mask, _ = load_cmb_map_and_mask(config)

    if config['run_sky']:
        plot_sky_distribution(samples, common_mask, config, out_dir, cat_label)

    if config['run_filter']:
        plot_filter_kernel(config, out_dir)

    if config['run_cmb']:
        plot_cmb_map(lensing_map, common_mask, config, out_dir)

    print('\n[plots] === LISTO ===')


if __name__ == '__main__':
    main()