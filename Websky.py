"""
Websky.py
=========
Voids identificados en Websky  x  kappa (CMB lensing) de Websky, con EXACTAMENTE
la misma pipeline que los datos:

    S1_voids_parallel._run_single_map  ->  Parallel_module  ->  Functions_module

No se reimplementa nada del stacking: se registra 'WEBSKY' como catálogo y como
mapa dentro de S1_voids_parallel y se llama a su _run_single_map. Selección,
cobertura, merge DBSCAN, binneado por cuantiles, stacking, JK, nulls (randoms +
rotaciones), caché y plots son el mismo código que corre sobre los datos.

Lo único específico de Websky está en este archivo:

 1. COSMOLOGÍA. fm/vsm usan Planck18 para pasar Mpc/h -> ángulo (y xyz en el
    merge). Para Websky se reemplaza por su cosmología (h=0.68, Om=0.31,
    Ob=0.049, plana, sin radiación): con ella chi(z) reproduce |xyz| del
    catálogo con precisión ~1e-6 (con Planck18 ~3e-5: despreciable, pero
    lo correcto es la de la simulación).

 2. CATÁLOGO. Los .dat de Websky tienen 12 columnas (las de los datos son 11):
        R_void l b z x y z_cart | 5 columnas extra
    Si se le pasan crudos a S1, pandas usa la 1ra columna como ÍNDICE y corre
    todo un lugar (R_void <- l, l <- b, ...) SIN dar error. Por eso se
    reescriben al formato de 11 columnas de la pipeline (o de 4 en modo
    catálogo único), mapeando las columnas extra con 'websky_cols'.
    (l,b) de Websky son los ángulos del frame cartesiano de la simulación, el
    mismo frame del HEALPix de kap.fits -> no hay rotación de coordenadas
    (se declara frame='galactic' sólo porque en S1 eso significa "identidad").

 3. MAPA. kap.fits (nside_in, típicamente 4096) -> alm hasta lmax (=nside de
    la pipeline, 2048, igual que en datos) [+ ruido gaussiano con N_L de PR4,
    opcional] -> se guarda como alm. El FILTRO lo aplica S1 con el mismo
    código que en datos (fm.apply_wiener_filter con el nlkk de PR4): el
    perfil de Websky filtrado con el MISMO W_L es la predicción del valor
    medio del perfil de datos (el ruido promedia a cero en el stack lineal).

 4. (Opcional) Comparación contra un Data_FullRun_*.pkl de los datos.

Uso:  python Websky.py      (desde la carpeta de los módulos de la pipeline)
"""

#%% Imports
import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, Planck18

import Functions_module as fm
import void_seed_merge as vsm
import S1_voids_parallel as s1


#%% Config
config = {
    'data_folder':   '/home/ftoscano/Doctorado/Data/',
    'websky_folder': '/home/ftoscano/Doctorado/Data/MOCKS/Websky_data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/websky/',

    # ---------------- Websky: insumos ----------------
    'cosmology':   'websky',              # 'websky' (h=0.68, Om=0.31) | 'planck18' (test)
    'kappa_file':  'CMB/kap.fits',        # relativo a websky_folder
    'single_file': 'voids_d-0.9.dat',     # usado si N_seeds = None
    'seed_glob':   'websky_voids/*.dat',  # usado si N_seeds = int (orden lexicográfico)

    # Rol de cada columna de los .dat de Websky (índice 0-based; None = no existe).
    # Las 7 primeras están verificadas (l,b <-> xyz y chi(z) <-> |xyz|).
    # !!! Las extra NO están verificadas: mirar el resumen por columna que se
    # imprime al correr. delta_23 sólo importa si delta_value != None;
    # completeness entra en el corte > 1.9 de S1 (si es None se rellena con 2).
    'websky_cols': {'R_void': 0, 'l': 1, 'b': 2, 'z': 3,
                    'x': 4, 'y': 5, 'z_cart': 6,
                    'delta_int': 7, 'delta_23': 8,
                    'completeness': 11, 'delta_LOS': None},

    # ---------------- Websky: mapa ----------------
    'nside':        2048,                 # nside de la pipeline (igual que PR4)
    'lmax':         2048,                 # = nside, como en datos (apply_wiener_filter(lmax=nside))
    'map2alm_iter': 1,                    # iteraciones de map2alm (lmax << 3*nside_in: 1 sobra)
    'deconvolve_pixwin': False,           # pixwin(4096) a L<=2048 es >~0.98 y W_L allí ~0
    'add_noise':    False,                # True: + realización gaussiana de N_L^PR4
    'noise_seed':   0,
    'mask_mode':    'full',               # 'full' (todo el cielo) |
                                          # 'planck_pr4' (máscara PR4 como geometría; el frame
                                          #  de Websky NO es galáctico, sólo copia fsky/bordes)
    'nlkk_file':     'CMB/PLANCK/Lensing/nlkk_PR4_MV.dat',                 # relativo a data_folder
    'pr4_mask_file': 'CMB/PLANCK/Lensing/Common_mask_PR4Lensing_2048.fits',
    'force_map':     False,               # rehacer el alm cacheado

    # ---------------- Pipeline (mismas claves que Pipeline_voids.py) ----------------
    'min_footprint_coverage': 0.9,
    'N_seeds': 100,                      # None -> single_file | int -> seed_glob
    'delta_value': None,
    'seed_mode': 'merge',
    'merge_eps_mpch': 8.0,
    'merge_min_frac': 0.4,
    'merge_use_catalog_xyz': False,       # con la cosmología de Websky recomputar == catálogo

    'zmin': 0.1, 'zmax': 0.5,
    'rmin': 35.0, 'rmax': 70.0,

    'max_Rvoid': 2.5,
    'Rvoid_bin': 0.14,
    'npix_stamp': 400,
    'filter_mode': ['wiener', 'none'],    # str o lista: 'wiener' (= datos) | 'none' (kappa
                                          # sin ruido, banda L<=lmax) | 'gaussian'
    'smooth_value_arcmin': 0.0,
    'act_smooth_arcmin': 0.0,

    'binning_mode': 'redshift',
    'n_bins': 1,

    'exec_mode': 'errors',
    'n_subsamples': 60,
    'n_rand_factor': 300,
    'n_rotations': 30,
    'random_pool': 'full',
    'random_excl_factor': None,

    'n_workers': 120,
    'force_rerun': True,

    # ---------------- Comparación con datos (opcional) ----------------
    # Data_FullRun_*.pkl de la corrida de datos, o la CARPETA que lo contiene
    # (mismo max_Rvoid y Rvoid_bin).
    'compare_data_pkl': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/lensing/WH_PLANCK_PR4_redshift_1bins_errors_0.1_0.5_35.0_70.0_maxRv2.5_0.02Rvperpix_d23_all_wiener',
    # True -> NO corre nada: lee los Data_FullRun_*.pkl de Websky ya guardados para
    # esta selección (cada filter_mode pedido) y sólo hace la comparación.
    'compare_only': True,
}

PIPELINE_COLS = ['R_void', 'l', 'b', 'z', 'x', 'y', 'z_cart',
                 'delta_int', 'delta_23', 'completeness', 'delta_LOS']

# Websky (Stein et al. 2020): Om=0.31, Ob=0.049, h=0.68, ns=0.965, sigma8=0.81.
# Tcmb0=0 (sin radiación): reproduce chi(z)=|xyz| del catálogo a ~1e-6.
WEBSKY_COSMO = FlatLambdaCDM(H0=68.0, Om0=0.31, Ob0=0.049, Tcmb0=0.0, name='Websky')

_ORIG_SPECS = (s1._catalog_spec, s1._map_spec)     # originales, antes de cualquier parche


#%% Logger
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


#%% Cosmología
def set_cosmology(kind):
    """fm y vsm leen el nombre global 'Planck18' de su módulo: se lo reemplaza.
    Se hace ANTES de crear el pool (fork) -> los workers heredan el parche."""
    cosmo = WEBSKY_COSMO if kind == 'websky' else Planck18
    fm.Planck18 = cosmo
    vsm.Planck18 = cosmo
    print(f"[cosmo] {cosmo.name}: h={cosmo.h:.4f}, Om0={cosmo.Om0:.4f}")
    return cosmo


#%% Catálogos
def _read_websky_raw(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#')
    df.columns = list(range(df.shape[1]))
    return df


def _to_pipeline(raw, cols):
    """DataFrame crudo de Websky -> columnas de la pipeline (11), por rol."""
    ncol = raw.shape[1]
    out = pd.DataFrame(index=raw.index)
    for name in PIPELINE_COLS:
        j = cols.get(name)
        if j is None:
            out[name] = 2.0 if name == 'completeness' else np.nan
        else:
            if j >= ncol:
                raise ValueError(f"websky_cols['{name}']={j} pero el archivo tiene {ncol} columnas")
            out[name] = raw[j].astype(float).values
    return out


def column_summary(raw, label):
    print(f"[catalog] Resumen por columna de {label} (N={len(raw)}):")
    for j in raw.columns:
        v = raw[j].values.astype(float)
        nu = len(np.unique(v[:20000]))
        print(f"   col {j:2d}: min={np.nanmin(v):12.5g}  med={np.nanmedian(v):12.5g}  "
              f"max={np.nanmax(v):12.5g}  n_unique(<=2e4 filas)={nu}")


def geometry_check(df, cosmo, label, tol_deg=1e-3, tol_chi=1e-3):
    """(l,b) vs dirección de (x,y,z_cart) y chi(z) vs |xyz|: detecta columnas mal
    asignadas y cosmología equivocada."""
    xyz = df[['x', 'y', 'z_cart']].values
    r = np.linalg.norm(xyz, axis=1)
    u_cat = hp_free_ang2vec(df['l'].values, df['b'].values)
    cosang = np.clip(np.sum(u_cat * xyz / r[:, None], axis=1), -1, 1)
    dang = np.degrees(np.arccos(cosang))
    chi = cosmo.comoving_distance(df['z'].values).value * cosmo.h
    rel = r / chi - 1.0
    with np.errstate(invalid='ignore'):
        ok_a, ok_c = np.nanmax(dang) < tol_deg, np.nanmax(np.abs(rel)) < tol_chi
    print(f"[geom] {label}: max sep((l,b),xyz)={np.nanmax(dang):.2e} deg | "
          f"|xyz|/chi(z)-1: med={np.nanmedian(rel):+.2e}, max|.|={np.nanmax(np.abs(rel)):.2e}"
          f"  -> {'OK' if (ok_a and ok_c) else 'FALLA'}")
    if not (ok_a and ok_c):
        raise ValueError(f"[geom] {label}: (l,b,z) no son consistentes con (x,y,z_cart) en la "
                         f"cosmología elegida -> revisar 'websky_cols' / 'cosmology'.")
    return True


def hp_free_ang2vec(l_deg, b_deg):
    lo, la = np.radians(l_deg), np.radians(b_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _delta_cut(df, delta_value):
    """Mismo criterio que S1.apply_delta_23_filter."""
    if delta_value is None:
        return df
    if df['delta_23'].isna().all():
        raise ValueError("delta_value != None pero websky_cols['delta_23'] es None")
    return df[df['delta_23'] > delta_value] if delta_value >= 0 else df[df['delta_23'] < delta_value]


def prepare_catalogs(cfg, cosmo, out_dir):
    """
    Escribe los catálogos en el formato que lee S1:
      - N_seeds=None -> 1 archivo de 4 columnas (R_void l b z). S1 en ese modo NO
        aplica completeness ni delta_23: se aplican acá para que la etiqueta
        d23_* de la carpeta sea verdadera.
      - N_seeds=int  -> N archivos de 11 columnas websky_{001..N}.dat (S1 aplica
        completeness>1.9 y delta_23 por semilla, igual que en datos).
    Se regeneran siempre (son rápidos): así nunca queda un caché viejo.
    """
    os.makedirs(out_dir, exist_ok=True)
    wf, cols, n_seeds = cfg['websky_folder'], cfg['websky_cols'], cfg.get('N_seeds')

    if n_seeds is None:
        src = os.path.join(wf, cfg['single_file'])
        raw = _read_websky_raw(src)
        column_summary(raw, os.path.basename(src))
        df = _to_pipeline(raw, cols)
        geometry_check(df, cosmo, os.path.basename(src))
        n0 = len(df)
        if cols.get('completeness') is not None:
            df = df[df['completeness'] > 1.9]
        df = _delta_cut(df, cfg.get('delta_value'))
        print(f"[catalog] {os.path.basename(src)}: {n0} -> {len(df)} voids tras completeness/delta_23")
        single = os.path.join(out_dir, 'websky_single.dat')
        np.savetxt(single, df[['R_void', 'l', 'b', 'z']].values, fmt='%.6f')
        template = os.path.join(out_dir, 'websky_{:03d}.dat')     # no usado
        return single, template

    files = sorted(glob.glob(os.path.join(wf, cfg['seed_glob'])))
    if len(files) < n_seeds:
        raise FileNotFoundError(f"N_seeds={n_seeds} pero '{cfg['seed_glob']}' encuentra "
                                f"{len(files)} archivos en {wf}")
    files = files[:n_seeds]
    print(f"[catalog] {n_seeds} semillas: {os.path.basename(files[0])} ... "
          f"{os.path.basename(files[-1])}")
    template = os.path.join(out_dir, 'websky_{:03d}.dat')
    for k, src in enumerate(files):
        raw = _read_websky_raw(src)
        if k == 0:
            column_summary(raw, os.path.basename(src))
        df = _to_pipeline(raw, cols)
        geometry_check(df, cosmo, os.path.basename(src))
        np.savetxt(template.format(k + 1), df[PIPELINE_COLS].values, fmt='%.6f')
    single = os.path.join(out_dir, 'websky_single.dat')           # no usado
    return single, template


#%% Mapa
def prepare_kappa_alm(cfg, out_dir):
    """kap.fits -> alm(lmax) [+ ruido N_L^PR4]; L<2 a cero. Cacheado en FITS."""
    os.makedirs(out_dir, exist_ok=True)
    lmax = cfg['lmax']
    kfile = os.path.join(cfg['websky_folder'], cfg['kappa_file'])
    ntag = f"_nPR4s{cfg['noise_seed']}" if cfg['add_noise'] else ''
    ptag = '_pwdec' if cfg['deconvolve_pixwin'] else ''
    base = os.path.splitext(os.path.basename(kfile))[0]
    path = os.path.join(out_dir, f"websky_{base}_alm_lmax{lmax}_it{cfg['map2alm_iter']}{ptag}{ntag}.fits")
    if os.path.exists(path) and not cfg.get('force_map', False):
        print(f"[map] alm cacheado: {path}")
        return path

    print(f"[map] Leyendo {kfile} ...")
    m = hp.read_map(kfile, field=0, dtype=np.float64)
    nside_in = hp.get_nside(m)
    print(f"[map] nside_in={nside_in}, <kappa>={np.mean(m):.3e}, std={np.std(m):.3e}")
    if lmax > 2 * nside_in:
        print(f"[map] (!) lmax={lmax} > 2*nside_in: map2alm poco preciso.")
    alm = hp.map2alm(m, lmax=lmax, iter=cfg['map2alm_iter'])
    del m

    fl = np.ones(lmax + 1)
    fl[:2] = 0.0                                          # sin monopolo/dipolo
    if cfg['deconvolve_pixwin']:
        fl[2:] /= hp.pixwin(nside_in, lmax=lmax)[2:]
    alm = hp.almxfl(alm, fl)

    if cfg['add_noise']:
        nlkk = os.path.join(cfg['data_folder'], cfg['nlkk_file'])
        NL = np.loadtxt(nlkk)[:, 1]
        if len(NL) < lmax + 1:
            raise ValueError(f"{nlkk} llega a L={len(NL)-1} < lmax={lmax}")
        NL = NL[:lmax + 1].copy(); NL[:2] = 0.0
        np.random.seed(cfg['noise_seed'])
        alm = alm + hp.synalm(NL, lmax=lmax)
        print(f"[map] + ruido gaussiano N_L^PR4 (seed={cfg['noise_seed']}) de {nlkk}")

    hp.write_alm(path, alm, overwrite=True)
    print(f"[map] alm guardado: {path}")
    return path


def prepare_mask(cfg, out_dir):
    if cfg['mask_mode'] == 'planck_pr4':
        return os.path.join(cfg['data_folder'], cfg['pr4_mask_file'])
    if cfg['mask_mode'] != 'full':
        raise ValueError(f"mask_mode desconocido: {cfg['mask_mode']}")
    path = os.path.join(out_dir, 'websky_fullsky_mask_ns16.fits')
    if not os.path.exists(path):                           # S1 la lleva a nside con ud_grade
        hp.write_map(path, np.ones(hp.nside2npix(16)), overwrite=True, dtype=np.float32)
    return path


#%% Registro de WEBSKY en S1
def install_websky(cat_spec, map_spec):
    orig_cat, orig_map = _ORIG_SPECS

    def _cat(catalog, data_folder):
        if str(catalog).upper() == 'WEBSKY':
            return dict(cat_spec)
        return orig_cat(catalog, data_folder)

    def _map(cmb_map, data_folder, act_smooth_arcmin=0.0):
        if str(cmb_map).upper() == 'WEBSKY':
            return dict(map_spec)
        return orig_map(cmb_map, data_folder, act_smooth_arcmin)

    s1._catalog_spec, s1._map_spec = _cat, _map


def map_label(cfg):
    lab = 'WSkap'
    if cfg['add_noise']:
        lab += f"_nPR4s{cfg['noise_seed']}"
    if cfg['mask_mode'] == 'planck_pr4':
        lab += '_mPR4'
    if cfg['cosmology'] != 'websky':
        lab += f"_cosmo{cfg['cosmology']}"
    return lab


#%% Comparación con datos
def _pick(results, prefer=('merged', 'concat', 'single')):
    for kind in prefer:
        for r in results:
            if r.get('catalog') == kind and int(r.get('bin_id', 0)) == 0:
                return r
    return results[0]


def _resolve_pkl(path):
    """Acepta el .pkl o la carpeta de la corrida (busca Data_FullRun_*.pkl adentro)."""
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, 'Data_FullRun_*.pkl')))
        if len(found) != 1:
            raise FileNotFoundError(f"{path}: {len(found)} archivos Data_FullRun_*.pkl (se espera 1)")
        return found[0]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _filter_from_folder(name):
    """Sufijo de filtro que pone S1 en el nombre de la carpeta -> filter_mode."""
    if name.endswith('_wiener'):
        return 'wiener'
    if name.endswith('_no_filter'):
        return 'none'
    if '_gaussian_' in name:
        return 'gaussian'
    return None


def load_saved_websky(cfg):
    """Resultados de Websky ya guardados por S1 para la selección del config."""
    lens = os.path.join(cfg['output_folder'], 'lensing')
    wanted = cfg['filter_mode'] if isinstance(cfg['filter_mode'], (list, tuple)) else [cfg['filter_mode']]
    sel = (f"WEBSKY_{map_label(cfg)}_{cfg['binning_mode']}_{cfg['n_bins']}bins_{cfg['exec_mode']}_"
           f"{cfg['zmin']}_{cfg['zmax']}_{cfg['rmin']}_{cfg['rmax']}_maxRv{cfg['max_Rvoid']:.1f}_")
    results = {}
    for d in sorted(glob.glob(os.path.join(lens, sel + '*'))):
        fmode = _filter_from_folder(os.path.basename(d))
        if fmode not in wanted or not os.path.isdir(d):
            continue
        if fmode in results:
            print(f"[compare] (!) más de una corrida '{fmode}' para esta selección; uso {os.path.basename(d)}")
        with open(_resolve_pkl(d), 'rb') as f:
            results[fmode] = _pick(pickle.load(f)['results'])
        print(f"[compare] Websky [{fmode}] <- {os.path.basename(d)}")
    if not results:
        raise FileNotFoundError(f"No hay corridas de Websky guardadas que empiecen con\n  {os.path.join(lens, sel)}")
    return results


def compare_with_data(ws_results, data_pkl, out_path, max_Rvoid):
    data_pkl = _resolve_pkl(data_pkl)
    print(f"[compare] datos <- {data_pkl}")
    with open(data_pkl, 'rb') as f:
        dd = pickle.load(f)
    dres, dpar = _pick(dd['results']), dd.get('parameters', {})
    d_filter = dpar.get('filter_mode')
    r = np.asarray(dres['r_frac'])
    net_d, err_d, _ = fm._net_and_err(dres)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    ax.axhline(0, color='k', ls=':', alpha=0.6); ax.axvline(1, color='gray', ls='--', alpha=0.6)
    ax.errorbar(r, net_d * 1e3, yerr=err_d * 1e3, fmt='o-', color='k', capsize=3, zorder=5,
                label=f"Datos {dpar.get('cmb_map', '?')} [{d_filter}] (N={dres.get('n_voids', '?')}, "
                      f"med Rv={dres.get('R_void_median', np.nan):.1f})")
    colors = {'wiener': 'xkcd:crimson', 'none': 'xkcd:steel blue', 'gaussian': 'xkcd:teal'}

    for fmode, res in ws_results.items():
        if res is None:
            continue
        rw = np.asarray(res['r_frac'])
        if rw.shape != r.shape or not np.allclose(rw, r):
            print(f"[compare] (!) r_frac distinto ({fmode}): revisar max_Rvoid/Rvoid_bin. Salteo.")
            continue
        net_w, err_w, _ = fm._net_and_err(res)
        col = colors.get(fmode, 'xkcd:grey')
        ax.fill_between(r, (net_w - err_w) * 1e3, (net_w + err_w) * 1e3, color=col, alpha=0.25)
        ax.plot(r, net_w * 1e3, '-', color=col, lw=1.8,
                label=f"Websky [{fmode}] (N={res.get('n_voids', '?')}, "
                      f"med Rv={res.get('R_void_median', np.nan):.1f})")
        if fmode != d_filter:
            continue
        # residuo y chi2 sólo contra el MISMO filtro
        with np.errstate(divide='ignore', invalid='ignore'):
            axr.plot(r, (net_d - net_w) / np.sqrt(err_d ** 2 + err_w ** 2), 'o-', color=col)
        C = _total_cov(dres) + _total_cov(res)
        good = np.isfinite(net_d) & np.isfinite(net_w) & np.isfinite(np.diag(C))
        n_jk = int(dpar.get('n_subsamples', 0) or 0)
        p = int(good.sum())
        hart = (n_jk - p - 2) / (n_jk - 1) if n_jk > p + 2 else np.nan
        dvec = (net_d - net_w)[good]
        chi2 = float(dvec @ np.linalg.pinv(C[np.ix_(good, good)]) @ dvec) * hart
        try:
            from scipy.stats import chi2 as _chi2
            pte = _chi2.sf(chi2, p)
        except Exception:
            pte = np.nan
        print(f"[compare] datos vs Websky [{fmode}]: chi2={chi2:.1f} / {p} bins "
              f"(Hartlap={hart:.2f}, PTE={pte:.3f})  -- indicativo: cov JK + randoms")

    for y in (-2, 2):
        axr.axhline(y, color='gray', ls='--', alpha=0.7)
    axr.axhline(0, color='k', alpha=0.5); axr.set_ylim(-5, 5)
    axr.set_xlabel(r'$r/R_v$'); axr.set_ylabel(r'$\Delta/\sigma$')
    ax.set_ylabel(r'$\kappa_{\rm net}\;[10^{-3}]$'); ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25); axr.grid(alpha=0.25); ax.set_xlim(-0.1, max_Rvoid + 0.1)
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[compare] figura: {out_path}")


def _total_cov(res):
    """cov JK + varianza de la media de randoms (diagonal), igual que el error de fm."""
    nb = len(res['profile'])
    C = res.get('cov_jk')
    C = np.zeros((nb, nb)) if C is None else np.nan_to_num(np.asarray(C, float))
    nr = res.get('n_randoms_done', 0) or 0
    if nr > 0 and res.get('null_rand_std') is not None:
        C = C + np.diag(np.asarray(res['null_rand_std']) ** 2 / nr)
    return C


#%% Main
def main(cfg=config):
    cfg = dict(cfg)
    cfg['void_catalog'] = 'WEBSKY'
    cfg['cmb_map'] = 'WEBSKY'
    cfg['coverage_map'] = None

    lens_out = os.path.join(cfg['output_folder'], 'lensing')
    os.makedirs(lens_out, exist_ok=True)
    out_cmp = os.path.join(lens_out, f"WebskyVsData_{map_label(cfg)}_{cfg['zmin']}_{cfg['zmax']}_"
                                     f"{cfg['rmin']}_{cfg['rmax']}.pdf")
    if cfg.get('compare_only'):
        if not cfg.get('compare_data_pkl'):
            raise ValueError("compare_only=True necesita 'compare_data_pkl'")
        compare_with_data(load_saved_websky(cfg), cfg['compare_data_pkl'], out_cmp, cfg['max_Rvoid'])
        return None

    sys.stdout = DualLogger(os.path.join(lens_out, 'websky_run.log'))
    print("### PIPELINE RUN: WEBSKY VOIDS x WEBSKY KAPPA ###")
    with open(os.path.join(lens_out, 'Websky_pipeline_config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    cosmo = set_cosmology(cfg['cosmology'])

    print("\n>>> Catálogos")
    single, template = prepare_catalogs(cfg, cosmo, os.path.join(cfg['output_folder'], 'catalogs'))

    print("\n>>> Mapa")
    maps_dir = os.path.join(cfg['output_folder'], 'maps')
    alm_path = prepare_kappa_alm(cfg, maps_dir)
    mask_path = prepare_mask(cfg, maps_dir)

    cat_spec = {'label': 'WEBSKY', 'seed_template': template, 'single_file': single,
                'lon_col': 'l', 'lat_col': 'b', 'frame': 'galactic'}   # 'galactic' == identidad en S1
    map_spec = {'label': map_label(cfg), 'kind': 'alm', 'frame': 'galactic',
                'nside': cfg['nside'], 'full_sky': True, 'apply_wiener': True,
                'klm': alm_path, 'mask': mask_path,
                'nlkk': os.path.join(cfg['data_folder'], cfg['nlkk_file'])}
    install_websky(cat_spec, map_spec)

    filters = cfg['filter_mode'] if isinstance(cfg['filter_mode'], (list, tuple)) else [cfg['filter_mode']]
    results = {}
    for fmode in filters:
        print(f"\n================= WEBSKY | filter_mode = {fmode} =================")
        c = dict(cfg); c['filter_mode'] = fmode
        results[fmode] = s1._run_single_map(c)

    if cfg.get('compare_data_pkl'):
        print("\n>>> Comparación con datos")
        compare_with_data(results, cfg['compare_data_pkl'], out_cmp, cfg['max_Rvoid'])

    with open(os.path.join(lens_out, 'SUCCESS_websky'), 'w') as f:
        f.write('Done.')
    print("\n=== FINISHED ===")
    return results


if __name__ == "__main__":
    main()
