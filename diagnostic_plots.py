"""
diagnostic_plots.py
===================
Plots diagnósticos (no de ciencia fina, sólo para mostrar la selección/insumos)
de la corrida de lensing de Pipeline_voids.py. Reutiliza EXACTAMENTE la misma
carga/filtrado/merge que S1_voids_parallel.run_pipeline, así lo que se dibuja es
la misma muestra y el mismo mapa que efectivamente se apila.

Genera, dentro de {output_folder}/Plots/ :
  1. Dos filas de histogramas: R_void (rmin-rmax) y z (zmin-zmax), con el
     catálogo y el corte en delta_23 en el título.
  2. Distribución en el cielo (mollview, l,b) con el tamaño angular real de los
     voids: si seed_mode incluye 'concat' -> mapa de pesos (densidad de centros);
     si incluye 'merge' -> voids finales pintados como discos de su tamaño real.
     En ambos se sobre-imprime el contorno de la máscara del mapa de CMB.
  3. Si hay filtro: el kernel aplicado (W_ell del Wiener, o b_ell del gaussiano).
  4. El mapa de CMB usado (ya filtrado) enmascarado, en el nside del config.

Corre con el MISMO numpy/healpy con el que corre el pipeline:
    python diagnostic_plots.py

NOTA: este script LEE los FITS/catálogos reales del disco (rutas del config).
No se puede probar sin esos datos.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless: sólo escribe PDFs
import matplotlib.pyplot as plt
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
    'rmin': 25.0, 'rmax': 70.0,    # Mpc/h

    # --- geometría / filtro (igual al pipeline) ---
    'max_Rvoid':          2.5,
    'npix_stamp':         400,
    'filter_mode':        'wiener',   # 'none' | 'gaussian' | 'wiener'
    'smooth_value_arcmin': 0.0,       # sólo si filter_mode == 'gaussian'

    # --- SÓLO del plotter (no afectan la ciencia) ---
    'nside':          2048,   # el pipeline lo hardcodea a 2048; acá se respeta el config
    'size_factor':    1.0,    # radio de los discos en unidades de R_void (1.0 = tamaño real del void)
    'nside_disc':     1024,   # nside para pintar los discos del cielo (barato; 2048 es innecesario)
    'nside_weight':   2048,   # nside para el mapa de pesos (densidad de centros)
    'nside_edge':     512,    # nside para trazar el contorno de la máscara
    'nside_cmb_plot': None,   # None -> usa 'nside'; bajalo (p.ej. 1024) para acelerar el mollview

    # --- qué plots correr ---
    'run_hist':   True,
    'run_sky':    True,
    'run_filter': True,
    'run_cmb':    True,
}


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
    """Garantiza columnas galácticas 'l','b' (el mapa de Planck está en galácticas)."""
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
    """Devuelve {kind: DataFrame} con kind in {'single','concat','merged'} según
    N_seeds y seed_mode. Cada DataFrame tiene al menos l,b,z,R_void ya filtrados."""
    data_folder = config['data_folder']
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

    # --- multi-seed: concat de todas las semillas filtradas ---
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
    """Reproduce EXACTO lo que hace S1_voids_parallel: lee alm, aplica el filtro
    y devuelve (lensing_map, common_mask, W_o_None). Ojo: la máscara y el nlkk
    están hardcodeados a PR4 en el pipeline (ver notas)."""
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
#  Utilidades de dibujo en el cielo
# ======================================================================
def _paint_disc_map(l, b, z, rv, nside, size_factor):
    """Mapa healpix con cada void pintado como un disco de su tamaño angular real
    (valor = nº de voids que solapan ese píxel)."""
    m = np.zeros(hp.nside2npix(nside))
    for i in range(len(l)):
        theta_deg = fm.get_angularsize_comoving(z[i], size_factor * rv[i])
        vec = hp.ang2vec(l[i], b[i], lonlat=True)
        pix = hp.query_disc(nside, vec, np.radians(max(theta_deg, 1e-3)))
        m[pix] += 1.0
    return m


def _paint_centers_map(l, b, nside):
    """Mapa de pesos: cuántos centros de void caen en cada píxel (colapsa LOS)."""
    m = np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, l, b, lonlat=True)
    np.add.at(m, pix, 1.0)
    return m


def _mask_boundary_lb(mask, nside_edge=512):
    """(l,b) de los píxeles frontera de la máscara binarizada, para trazar el contorno."""
    m = hp.ud_grade(mask, nside_out=nside_edge)
    binm = (m > 0.5).astype(np.int8)
    inside = np.where(binm == 1)[0]
    if len(inside) == 0:
        return np.array([]), np.array([])
    neigh = hp.get_all_neighbours(nside_edge, inside)   # (8, Ninside); -1 = sin vecino
    is_edge = np.zeros(len(inside), dtype=bool)
    for k in range(neigh.shape[0]):
        nb = neigh[k]
        out = np.ones(len(inside), dtype=bool)          # -1 cuenta como "afuera"
        valid = nb >= 0
        out[valid] = (binm[nb[valid]] == 0)
        is_edge |= out
    bpix = inside[is_edge]
    theta, phi = hp.pix2ang(nside_edge, bpix)
    return np.degrees(phi), 90.0 - np.degrees(theta)    # l, b


def _mollview_with_mask(sky_map, edge_l, edge_b, title, cbar_label,
                        sub=None, cmap='turbo'):
    disp = sky_map.copy()
    disp[disp == 0] = hp.UNSEEN
    hp.mollview(disp, title=title, cmap=cmap, min=1, sub=sub,
                unit=cbar_label, badcolor='white')
    if len(edge_l):
        hp.projscatter(edge_l, edge_b, lonlat=True, s=0.15,
                       color='black', alpha=0.6)
    hp.graticule(dpar=30, dmer=30, color='grey', alpha=0.4)


# ======================================================================
#  PLOT 1 — histogramas de R_void y z
# ======================================================================
def plot_histograms(samples, config, out_dir, cat_label):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8),
                             gridspec_kw={'hspace': 0.28})

    colors = {'single': 'xkcd:steel blue', 'concat': 'xkcd:steel blue',
              'merged': 'xkcd:crimson'}
    labels = {'single': 'catálogo único', 'concat': 'concat (multi-seed)',
              'merged': 'merged (DBSCAN)'}

    # --- R_void ---
    ax = axes[0]
    r_edges = np.linspace(config['rmin'], config['rmax'], 26)
    for kind, df in samples.items():
        ax.hist(df['R_void'].values, bins=r_edges, histtype='step', density=True,
                color=colors.get(kind, 'k'), lw=1.6,
                label=f"{labels.get(kind, kind)}  (N={len(df)}, "
                      f"med={np.median(df['R_void']):.1f})")
    ax.axvline(config['rmin'], color='grey', ls='--', alpha=0.6)
    ax.axvline(config['rmax'], color='grey', ls='--', alpha=0.6)
    ax.set_xlabel(r'$R_{\rm void}$  [Mpc/h]')
    ax.set_ylabel('densidad')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # --- z ---
    ax = axes[1]
    z_edges = np.linspace(config['zmin'], config['zmax'], 26)
    for kind, df in samples.items():
        ax.hist(df['z'].values, bins=z_edges, histtype='step', density=True,
                color=colors.get(kind, 'k'), lw=1.6,
                label=f"{labels.get(kind, kind)}  (N={len(df)}, "
                      f"med={np.median(df['z']):.3f})")
    ax.axvline(config['zmin'], color='grey', ls='--', alpha=0.6)
    ax.axvline(config['zmax'], color='grey', ls='--', alpha=0.6)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('densidad')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.suptitle(f"{cat_label} voids   |   {_delta_text(config.get('delta_value'))}",
                 y=0.94, fontsize=13)
    out = os.path.join(out_dir, f'hist_R_z_{cat_label}.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  PLOT 2 — distribución en el cielo con tamaño real + contorno de máscara
# ======================================================================
def plot_sky_distribution(samples, common_mask, config, out_dir, cat_label):
    edge_l, edge_b = _mask_boundary_lb(common_mask, config['nside_edge'])

    # una entrada por muestra dibujable (concat -> pesos; merged/single -> discos)
    panels = []
    for kind, df in samples.items():
        if kind == 'concat':
            m = _paint_centers_map(df['l'].values, df['b'].values, config['nside_weight'])
            panels.append((m, f'{cat_label} — mapa de pesos (centros/píxel)',
                           'nº de detecciones'))
        else:   # 'merged' o 'single'
            m = _paint_disc_map(df['l'].values, df['b'].values,
                                df['z'].values, df['R_void'].values,
                                config['nside_disc'], config['size_factor'])
            sf = config['size_factor']
            panels.append((m, f'{cat_label} — voids ({sf:g}$\\times R_v$)',
                           'nº de voids solapados'))

    n = len(panels)
    fig = plt.figure(figsize=(9 * n, 6))
    for i, (m, title, cbar) in enumerate(panels):
        _mollview_with_mask(m, edge_l, edge_b, title, cbar,
                            sub=(1, n, i + 1) if n > 1 else None)
    fig.suptitle(_delta_text(config.get('delta_value')), y=1.02, fontsize=12)
    out = os.path.join(out_dir, f'sky_distribution_{cat_label}.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] guardado {out}')


# ======================================================================
#  PLOT 3 — kernel del filtro (Wiener o gaussiano)
# ======================================================================
def plot_filter_kernel(config, out_dir, W_wiener=None):
    filter_mode = config.get('filter_mode', 'none')
    data_folder = config['data_folder']

    if filter_mode == 'wiener':
        nlkk_file = f'{data_folder}CMB/Lensing/nlkk_PR4_MV.dat'
        d = np.loadtxt(nlkk_file)
        L = d[:, 0].astype(int)
        N_L = d[:, 1]
        SN = d[:, 2]
        Cl = np.maximum(SN - N_L, 0.0)          # igual que fm.apply_wiener_filter
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.where(SN > 0, Cl / SN, 0.0)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(L, W, color='xkcd:crimson', lw=1.8)
        a1.set_xlabel(r'$\ell$'); a1.set_ylabel(r'$W_\ell$')
        a1.set_title(r'Wiener  $W_\ell = C_\ell^{\kappa\kappa}/(C_\ell^{\kappa\kappa}+N_\ell)$')
        a1.set_xlim(0, len(L)); a1.set_ylim(-0.02, 1.02); a1.grid(alpha=0.25)

        good = (L > 1) & (Cl > 0) & (N_L > 0)
        a2.loglog(L[good], Cl[good], color='xkcd:steel blue', lw=1.6,
                  label=r'$C_\ell^{\kappa\kappa}$ (fiducial)')
        a2.loglog(L[good], N_L[good], color='xkcd:orange', lw=1.6,
                  label=r'$N_\ell^{\kappa\kappa}$')
        a2.set_xlabel(r'$\ell$'); a2.set_ylabel(r'$C_\ell$')
        a2.set_title('Señal vs ruido'); a2.legend(); a2.grid(alpha=0.25, which='both')
        out = os.path.join(out_dir, 'filter_kernel_wiener.pdf')

    elif filter_mode == 'gaussian' and config.get('smooth_value_arcmin', 0.0) > 0:
        fwhm_deg = config['smooth_value_arcmin'] / 60.0
        lmax = 2048
        b_ell = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax)
        L = np.arange(lmax + 1)
        fig, a1 = plt.subplots(1, 1, figsize=(6.5, 4.5))
        a1.plot(L, b_ell, color='xkcd:crimson', lw=1.8)
        a1.set_xlabel(r'$\ell$'); a1.set_ylabel(r'$b_\ell$')
        a1.set_title(rf'Gaussiano  FWHM$={config["smooth_value_arcmin"]:g}\prime$'
                     rf'  ($={fwhm_deg:.3f}^\circ$)')
        a1.set_xlim(0, lmax); a1.grid(alpha=0.25)
        out = os.path.join(out_dir, 'filter_kernel_gaussian.pdf')

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
    m = lensing_map
    mask = common_mask
    if nside_plot != config['nside']:
        m = hp.ud_grade(m, nside_out=nside_plot)
        mask = hp.ud_grade(mask, nside_out=nside_plot)

    disp = m.copy()
    disp[mask < 0.5] = hp.UNSEEN
    vlim = np.nanpercentile(np.abs(disp[np.isfinite(disp) & (disp != hp.UNSEEN)]), 99)

    fig = plt.figure(figsize=(11, 7))
    label = {'wiener': 'Wiener', 'gaussian': 'gaussiano',
             'none': 'sin filtro'}.get(config.get('filter_mode', 'none'), config.get('filter_mode'))
    hp.mollview(disp, title=rf"$\kappa$ ({config['release']}, {label}) — nside {nside_plot}",
                cmap='RdBu_r', min=-vlim, max=vlim, unit=r'$\kappa$', badcolor='white')
    hp.graticule(dpar=30, dmer=30, color='grey', alpha=0.4)
    out = os.path.join(out_dir, f"cmb_map_{config['release']}_{config.get('filter_mode')}.pdf")
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

    # --- muestras (se necesitan para hist y cielo) ---
    samples = None
    if config['run_hist'] or config['run_sky']:
        samples = build_samples(config, cat)

    if config['run_hist']:
        plot_histograms(samples, config, out_dir, cat_label)

    # --- mapa/máscara (se necesitan para cielo, filtro-cmb) ---
    lensing_map = common_mask = W = None
    need_maps = config['run_sky'] or config['run_cmb']
    if need_maps:
        lensing_map, common_mask, W = load_cmb_map_and_mask(config)

    if config['run_sky']:
        plot_sky_distribution(samples, common_mask, config, out_dir, cat_label)

    if config['run_filter']:
        plot_filter_kernel(config, out_dir, W_wiener=W)

    if config['run_cmb']:
        plot_cmb_map(lensing_map, common_mask, config, out_dir)

    print('\n[plots] === LISTO ===')


if __name__ == '__main__':
    main()