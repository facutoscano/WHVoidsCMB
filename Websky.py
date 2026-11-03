"""
Websky.py
=========
Voids identificados en Websky  x  kappa de Websky, con EXACTAMENTE la misma pipeline
que los datos (S1_voids_parallel._run_single_map -> Parallel_module -> Functions_module).
Se registra 'WEBSKY' como catálogo y como mapa dentro de S1 y se llama a su
_run_single_map: selección, cobertura, merge, binneado, stacking, JK, nulls, caché y
plots son el mismo código que corre sobre los datos.

CONVENCIÓN DE COORDENADAS
    Voids y kap.fits de Websky comparten el frame nativo de la simulación (mismos ejes
    cartesianos): entre ellos NUNCA se transforma nada. Ese frame se usa como galáctico.
      FULL, PR4     -> todo nativo: voids, kappa y máscara PR4 (G) sin transformar;
                       footprint WH/BOSS en galácticas.
      ACT, ACT-PR4  -> frame ecuatorial (C): máscara ACT y mapa de Niall tal cual, máscara
                       PR4 rotada G->C, footprint WH/BOSS en ecuatoriales. Voids y kappa:
                       'act_native_mode' = 'identity' -> sin rotar (Niall usó el nativo como C)
                                           'rotate'   -> rotados G->C (Niall rotó el nativo)
                                           'auto'     -> se decide correlacionando el mapa
                                                         de Niall con kappa y tSZ de Websky
                                                         bajo ambas hipótesis.
    Voids y mapas se rotan con la MISMA matriz (healpy Rotator G->C).

VOIDS ('void_catalogs': 'WH' | 'BOSS' | 'WH-BOSS')
    Siempre los mismos voids de Websky; el catálogo elige qué footprint se copia:
    se conservan los voids con >= footprint_min_cov de su disco de footprint_rv_frac*Rv
    dentro del footprint (equivalente a la completeness de Sparkling en los datos).
    'WH-BOSS' corre ambos y los compara.

CORRIDAS ('surveys': cualquier subconjunto de 'FULL', 'PR4', 'ACT', 'ACT-PR4')
    FULL     raw_FULL       kappa puro (sin ruido ni filtro), cielo completo, todos los voids
    PR4      raw_PR4        kappa puro, máscara PR4
             obsPR4_PR4     kappa + ruido N_L^PR4 + Wiener PR4, máscara PR4   (= datos WH x PR4)
    ACT      raw_ACT        kappa puro, máscara ACT (cobertura de voids en ACT)
             noiseACT_ACT   mapa de Niall: foregrounds + reconstrucción ACT, SIN lensing
                            -> sesgo de foregrounds y nivel real de ruido de ACT
    ACT-PR4  obsPR4_ACTPR4  kappa + ruido PR4 + Wiener PR4, voids con cobertura en ACT
                            (= datos en modo 'act-pr4' de S1)
    Las corridas en ACT son de cielo parcial (sin rotaciones), como ACT en los datos.

COMPARACIÓN CON DATOS ('compare_data'[catálogo][survey] = Data_FullRun_*.pkl o su carpeta)
    PR4, ACT-PR4: datos vs obsPR4 -> perfiles + figura aparte de chi2 (cov de randoms).
    ACT:          datos ACT vs noiseACT (+ raw_ACT de referencia) + cociente de ruidos
                  sigma_rand(Websky ACT sin lensing)/sigma_rand(datos ACT) + chi2 de
                  noiseACT contra 0 (¿el sesgo de foregrounds es compatible con nulo?).
    Las barras de estas figuras son sigma de randoms (la misma cov que el chi2).

Uso:  python Websky.py      (desde la carpeta de los módulos de la pipeline)
"""

#%% Imports
import os
import re
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
from astropy.coordinates import SkyCoord
from astropy import units as u

import Functions_module as fm
import void_seed_merge as vsm
import S1_voids_parallel as s1


#%% Config
config = {
    'data_folder':   '/home/ftoscano/Doctorado/Data/',
    'websky_folder': '/home/ftoscano/Doctorado/Data/MOCKS/Websky_data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/websky/',

    # ---------------- Qué correr ----------------
    'void_catalogs': 'WH',                     # 'WH' | 'BOSS' | 'WH-BOSS'
    'surveys': ['FULL', 'PR4', 'ACT', 'ACT-PR4'],

    # ---------------- Websky: catálogo ----------------
    'cosmology':   'websky',                   # 'websky' (h=0.68, Om=0.31) | 'planck18' (test)
    'single_file': 'voids_d-0.9.dat',          # usado si N_seeds = None
    'seed_glob':   'websky_voids/*.dat',       # usado si N_seeds = int
    'websky_cols': {'R_void': 0, 'l': 1, 'b': 2, 'z': 3,
                    'x': 4, 'y': 5, 'z_cart': 6,
                    'delta_int': 7, 'delta_23': 8,
                    'completeness': 11, 'delta_LOS': None},

    # ---------------- Websky: mapas (relativos a websky_folder) ----------------
    'kappa_file':     'CMB/kap.fits',
    'act_noise_map':  'kappa_niall_foreground_wsky_WF.fits',   # ACT-like sin lensing (ya filtrado, en C)
    'act_native_mode': 'auto',                 # 'auto' | 'identity' | 'rotate' (ver docstring)
    'tsz_check_map':  'CMB/tsz_2048.fits',     # usado por 'auto' (además del kappa de Websky)
    # ---------------- Planck: ruido y filtro (relativo a data_folder) ----------------
    # 3 columnas (L, N_L, C_L+N_L): N_L para el ruido gaussiano y el MISMO archivo para el
    # Wiener que aplica S1 (idéntico a los datos).
    'pr4_noise_file': 'CMB/PLANCK/Lensing/nlkk_PR4_MV.dat',
    'noise_seed':     0,
    'nside':          2048,                    # nside de las corridas de alm (PR4 y raw)
    'lmax':           2048,
    'map2alm_iter':   1,
    'deconvolve_pixwin': False,
    'force_map':      False,                   # rehacer alm / máscaras / footprints cacheados

    # ---------------- Máscaras (relativas a data_folder) ----------------
    'masks': {
        'PR4': {'path': 'CMB/PLANCK/Lensing/Common_mask_PR4Lensing_2048.fits', 'coord': 'G'},
        'ACT': {'path': 'CMB/ACT/mask_act_dr6_baseline_ns512.fits',           'coord': 'C'},
    },
    'footprint_sources': {
        # 'points': ocupación HEALPix (nside) de posiciones (cols lon/lat, 'frame' icrs|galactic)
        # 'healpix': máscara ya hecha ('coord': 'G' | 'C')
        'WH':   {'kind': 'points', 'path': 'CATALOGOS/WenHan_Clusters.dat',
                 'cols': (2, 3), 'frame': 'icrs', 'skiprows': 56, 'n_files': None, 'nside': 64},
        'BOSS': {'kind': 'points', 'path': 'CATALOGOS/BOSS_voids_merge/voids_boss_{:03d}.dat',
                 'cols': (1, 2), 'frame': 'icrs', 'skiprows': 0, 'n_files': 100, 'nside': 16},
    },
    'footprint_nside':   256,
    'footprint_rv_frac': 1.0,
    'footprint_min_cov': 0.95,

    # ---------------- Pipeline (mismas claves que Pipeline_voids.py) ----------------
    'min_footprint_coverage': 0.9,
    'N_seeds': 100,
    'delta_value': None,
    'seed_mode': 'merge',
    'merge_eps_mpch': 8.0,
    'merge_min_frac': 0.4,
    'merge_use_catalog_xyz': False,

    'zmin': 0.1, 'zmax': 0.5,
    'rmin': 35.0, 'rmax': 70.0,

    'max_Rvoid': 2.5,
    'Rvoid_bin': 0.14,
    'npix_stamp': 400,
    'smooth_value_arcmin': 0.0,
    'act_smooth_arcmin': 0.0,

    'binning_mode': 'redshift',
    'n_bins': 1,

    'exec_mode': 'errors',
    'n_subsamples': 60,
    'n_rand_factor': 300,
    'n_rotations': 10,
    'random_pool': 'full',
    'random_excl_factor': None,

    'n_workers': 120,
    'force_rerun': True,

    # ---------------- Comparación con datos ----------------
    # Data_FullRun_*.pkl o la carpeta que lo contiene; None = no comparar ese caso.
    # 'ACT-PR4' = corrida de datos PR4 en el footprint de ACT (carpeta ..._wiener_covACT).
    'compare_data': {
        'WH':   {'PR4': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/lensing/'
                        'WH_PLANCK_PR4_redshift_1bins_errors_0.1_0.5_35.0_70.0_maxRv2.5_0.02Rvperpix_d23_all_wiener',
                 'ACT': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/lensing/WH_ACT_redshift_1bins_errors_0.1_0.5_35.0_70.0_maxRv2.5_0.08Rvperpix_d23_all_prefiltered', 'ACT-PR4': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/lensing/WH_PLANCK_PR4_redshift_1bins_errors_0.1_0.5_35.0_70.0_maxRv2.5_0.02Rvperpix_d23_all_wiener_covACT'},
        'BOSS': {'PR4': None, 'ACT': None, 'ACT-PR4': None},
    },
    'compare_only': False,                     # True: no corre nada, lee lo guardado y compara
}

NATIVE_FRAME = 'G'                             # Websky: cielo completo en galácticas
SURVEY_FRAME = {'FULL': 'G', 'PR4': 'G', 'ACT': 'C', 'ACT-PR4': 'C'}
COV_KEY = 'WEBSKY_COV'                         # clave de coverage_map para S1
PIPELINE_COLS = ['R_void', 'l', 'b', 'z', 'x', 'y', 'z_cart',
                 'delta_int', 'delta_23', 'completeness', 'delta_LOS']
WEBSKY_COSMO = FlatLambdaCDM(H0=68.0, Om0=0.31, Ob0=0.049, Tcmb0=0.0, name='Websky')
_ORIG_SPECS = (s1._catalog_spec, s1._map_spec)

RUN_NAMES = {
    'raw_FULL':      'κ puro, cielo completo',
    'raw_PR4':       'κ puro, cobertura PR4',
    'raw_ACT':       'κ puro, cobertura ACT',
    'obsPR4_PR4':    'κ como PR4 (ruido + Wiener)',
    'obsPR4_ACTPR4': 'κ como PR4 en cobertura ACT',
    'noiseACT_ACT':  'ACT sin lensing (fg + ruido)',
}
RUN_COLORS = {
    'raw_FULL': 'xkcd:black', 'raw_PR4': 'xkcd:steel blue', 'raw_ACT': 'xkcd:teal',
    'obsPR4_PR4': 'xkcd:crimson', 'obsPR4_ACTPR4': 'xkcd:orange', 'noiseACT_ACT': 'xkcd:purple',
}
CAT_COLORS = {'WH': 'xkcd:blue', 'BOSS': 'xkcd:orange'}


#%% Logger
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


#%% Cosmología
def set_cosmology(kind):
    cosmo = WEBSKY_COSMO if kind == 'websky' else Planck18
    fm.Planck18 = cosmo
    vsm.Planck18 = cosmo
    print(f"[cosmo] {cosmo.name}: h={cosmo.h:.4f}, Om0={cosmo.Om0:.4f}")
    return cosmo


#%% Definición de corridas
def catalogs_of(cfg):
    vc = cfg['void_catalogs'].upper()
    if vc == 'WH-BOSS':
        return ['WH', 'BOSS']
    if vc not in ('WH', 'BOSS'):
        raise ValueError(f"void_catalogs desconocido: {cfg['void_catalogs']}")
    return [vc]


def build_runs(cfg):
    S = [s.upper() for s in cfg['surveys']]
    bad = set(S) - {'FULL', 'PR4', 'ACT', 'ACT-PR4'}
    if bad:
        raise ValueError(f"surveys desconocidos: {bad}")
    runs = []
    if 'FULL' in S:
        runs.append(dict(key='raw_FULL', survey='FULL', field='raw', mask='full', cov=None, footprint=False))
    if 'PR4' in S:
        runs += [dict(key='raw_PR4', survey='PR4', field='raw', mask='PR4', cov=None, footprint=True),
                 dict(key='obsPR4_PR4', survey='PR4', field='obsPR4', mask='PR4', cov=None, footprint=True)]
    if 'ACT' in S:
        runs += [dict(key='raw_ACT', survey='ACT', field='raw', mask='ACT', cov='ACT', footprint=True),
                 dict(key='noiseACT_ACT', survey='ACT', field='noiseACT', mask='ACT', cov='ACT', footprint=True)]
    if 'ACT-PR4' in S:
        runs.append(dict(key='obsPR4_ACTPR4', survey='ACT-PR4', field='obsPR4', mask='PR4', cov='ACT',
                         footprint=True))
    for r in runs:
        r['full_sky'] = r['mask'] != 'ACT'          # cielo parcial ACT: sin rotaciones (como en datos)
        r['frame'] = SURVEY_FRAME[r['survey']]
    return runs


def cat_label(cat):
    return f"WS-{cat}" if cat else "WS-ALL"


def map_label(cfg, run):
    lab = f"WS{run['field']}"
    if run['field'] == 'obsPR4':
        lab += f"s{cfg['noise_seed']}"
    lab += '_' + run['survey'].replace('-', '')
    if cfg['cosmology'] != 'websky':
        lab += f"_cosmo{cfg['cosmology']}"
    return lab


def filter_label(run):
    """Sufijo de filtro que S1 pone en la carpeta (+ _cov<label> si hay coverage_map)."""
    f = {'raw': 'no_filter', 'obsPR4': 'wiener', 'noiseACT': 'prefiltered'}[run['field']]
    return f + ('_covACTmask' if run['cov'] == 'ACT' else '')


#%% Máscaras y footprints en el frame de Websky
def _rotator(frame_in, frame_out):
    return hp.Rotator(coord=[frame_in, frame_out])


def _to_frame(m, coord_in, frame):
    if coord_in == frame:
        return m
    return _rotator(coord_in, frame).rotate_map_pixel(m)


def mask_path(cfg, which, frame, maps_dir):
    """Ruta a la máscara 'full' | 'PR4' | 'ACT' en el frame pedido (cacheada si se rota)."""
    if which == 'full':
        path = os.path.join(maps_dir, 'websky_fullsky_mask_ns16.fits')
        if not os.path.exists(path):
            hp.write_map(path, np.ones(hp.nside2npix(16)), overwrite=True, dtype=np.float32)
        return path
    spec = cfg['masks'][which]
    src = os.path.join(cfg['data_folder'], spec['path'])
    if not os.path.exists(src):
        raise FileNotFoundError(f"máscara {which} no encontrada: {src}")
    if spec['coord'] == frame:
        return src
    path = os.path.join(maps_dir, f"mask_{which}_in{frame}.fits")
    if os.path.exists(path) and not cfg.get('force_map', False):
        return path
    m = hp.read_map(src, dtype=np.float64)
    mr = _to_frame(m, spec['coord'], frame)
    hp.write_map(path, mr.astype(np.float32), overwrite=True, dtype=np.float32)
    print(f"[mask] {which}: {spec['coord']} -> {frame} (fsky {np.mean(m >= 0.9):.3f} -> "
          f"{np.mean(mr >= 0.9):.3f}) -> {os.path.basename(path)}")
    return path


def _footprint_points(src, cfg, frame):
    files = ([src['path'].format(k + 1) for k in range(src['n_files'])]
             if src.get('n_files') else [src['path']])
    lon, lat = [], []
    for f in files:
        d = pd.read_csv(os.path.join(cfg['data_folder'], f), sep=r'\s+', header=None,
                        skiprows=src.get('skiprows', 0), usecols=list(src['cols']), comment='#')
        lon.append(d[src['cols'][0]].values.astype(float))
        lat.append(d[src['cols'][1]].values.astype(float))
    lon, lat = np.concatenate(lon), np.concatenate(lat)
    fr = 'C' if src['frame'] in ('icrs', 'equatorial') else 'G'
    if fr != frame:
        if fr == 'C':
            c = SkyCoord(ra=lon * u.degree, dec=lat * u.degree, frame='icrs').galactic
            lon, lat = c.l.degree, c.b.degree
        else:
            c = SkyCoord(l=lon * u.degree, b=lat * u.degree, frame='galactic').icrs
            lon, lat = c.ra.degree, c.dec.degree
    print(f"[footprint]   {len(lon)} posiciones de {os.path.basename(files[0])}"
          f"{' (+%d archivos)' % (len(files) - 1) if len(files) > 1 else ''}")
    return fm.footprint_mask(lon, lat, output_nside=cfg['footprint_nside'], footprint_nside=src['nside'])


def build_footprint(cfg, cat, frame, maps_dir):
    path = os.path.join(maps_dir, f"footprint_{cat}_in{frame}_ns{cfg['footprint_nside']}.fits")
    if os.path.exists(path) and not cfg.get('force_map', False):
        print(f"[footprint] {cat} ({frame}): cacheado {os.path.basename(path)}")
        return hp.read_map(path, dtype=np.float64)
    src = cfg['footprint_sources'][cat]
    print(f"[footprint] {cat} en {frame} ({src['kind']})")
    if src['kind'] == 'points':
        F = _footprint_points(src, cfg, frame)
    else:
        m = hp.read_map(os.path.join(cfg['data_folder'], src['path']), dtype=np.float64)
        F = (hp.ud_grade(_to_frame(m, src.get('coord', 'G'), frame), cfg['footprint_nside']) > 0.5).astype(float)
    hp.write_map(path, F.astype(np.float32), overwrite=True, dtype=np.float32)
    print(f"[footprint] {cat} ({frame}): fsky={F.mean():.3f} ({F.mean() * 41253:.0f} deg2)")
    return F


def plot_geometry(F, mask_files, out_png, title):
    """Chequeo visual en el frame de Websky: footprint de voids + máscaras de mapa."""
    try:
        ns = hp.get_nside(F)
        show = 2.0 * (F > 0.5)
        for mf in mask_files:
            m = hp.ud_grade(hp.read_map(mf, dtype=np.float64), ns)
            show = show + np.where(m >= 0.9, 1.0, 0.0)
            print(f"[footprint]   footprint & {os.path.basename(mf)}: "
                  f"{np.mean((F > 0.5) & (m >= 0.9)) * 41253:.0f} deg2")
        hp.mollview(show, title=title, cmap='Greys', cbar=False)
        hp.graticule(dpar=30, dmer=60, alpha=0.4)
        plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close('all')
    except Exception as e:
        print(f"[footprint] (no se pudo graficar: {e})")


#%% Catálogos
def _read_websky_raw(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#')
    df.columns = list(range(df.shape[1]))
    return df


def _to_pipeline(raw, cols):
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
        print(f"   col {j:2d}: min={np.nanmin(v):12.5g}  med={np.nanmedian(v):12.5g}  "
              f"max={np.nanmax(v):12.5g}  n_unique(<=2e4 filas)={len(np.unique(v[:20000]))}")


def hp_free_ang2vec(l_deg, b_deg):
    lo, la = np.radians(l_deg), np.radians(b_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def geometry_check(df, cosmo, label, tol_deg=1e-3, tol_chi=1e-3):
    xyz = df[['x', 'y', 'z_cart']].values
    r = np.linalg.norm(xyz, axis=1)
    cosang = np.clip(np.sum(hp_free_ang2vec(df['l'].values, df['b'].values) * xyz / r[:, None], axis=1), -1, 1)
    dang = np.degrees(np.arccos(cosang))
    chi = cosmo.comoving_distance(df['z'].values).value * cosmo.h
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = r / chi - 1.0
        ok_a, ok_c = np.nanmax(dang) < tol_deg, np.nanmax(np.abs(rel)) < tol_chi
    print(f"[geom] {label}: max sep((l,b),xyz)={np.nanmax(dang):.2e} deg | "
          f"|xyz|/chi(z)-1: max|.|={np.nanmax(np.abs(rel)):.2e}  -> {'OK' if (ok_a and ok_c) else 'FALLA'}")
    if not (ok_a and ok_c):
        raise ValueError(f"[geom] {label}: (l,b,z) no son consistentes con (x,y,z_cart) -> revisar "
                         f"'websky_cols' / 'cosmology'.")
    return True


def _delta_cut(df, delta_value):
    if delta_value is None:
        return df
    if df['delta_23'].isna().all():
        raise ValueError("delta_value != None pero websky_cols['delta_23'] es None")
    return df[df['delta_23'] > delta_value] if delta_value >= 0 else df[df['delta_23'] < delta_value]


def rotate_voids(df, frame):
    """(l,b) y (x,y,z_cart) del frame nativo de Websky (G) al frame pedido, con la misma
    matriz que se usa para los mapas. Distancias 3D (merge DBSCAN) invariantes."""
    if frame == NATIVE_FRAME or len(df) == 0:
        return df
    rot = _rotator(NATIVE_FRAME, frame)
    lon, lat = rot(df['l'].values, df['b'].values, lonlat=True)
    df = df.copy()
    df['l'], df['b'] = np.mod(lon, 360.0), lat
    xyz = df[['x', 'y', 'z_cart']].values @ np.asarray(rot.mat).T
    df['x'], df['y'], df['z_cart'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return df


def _footprint_cut(df, F, cfg, label, cat):
    if F is None or len(df) == 0:
        return df
    cov = fm.footprint_coverage(df['l'].values, df['b'].values, df['z'].values,
                                df['R_void'].values, F, cfg['footprint_nside'], 'galactic',
                                max_Rvoid=cfg['footprint_rv_frac'])
    keep = cov >= cfg['footprint_min_cov']
    print(f"[footprint] {label}: {keep.sum()}/{len(df)} voids dentro de {cat}")
    return df[keep]


def prepare_catalogs(cfg, cosmo, out_dir, F=None, cat=None, rot_to=None):
    """Catálogos en el formato que lee S1 (se regeneran siempre).
    rot_to=None: posiciones nativas; rot_to='C': voids rotados NATIVE_FRAME -> C."""
    frame = rot_to or NATIVE_FRAME
    os.makedirs(out_dir, exist_ok=True)
    wf, cols, n_seeds = cfg['websky_folder'], cfg['websky_cols'], cfg.get('N_seeds')
    template = os.path.join(out_dir, 'websky_{:03d}.dat')
    single = os.path.join(out_dir, 'websky_single.dat')
    if n_seeds is None:
        src = os.path.join(wf, cfg['single_file'])
        raw = _read_websky_raw(src)
        column_summary(raw, os.path.basename(src))
        df = _to_pipeline(raw, cols)
        geometry_check(df, cosmo, os.path.basename(src))
        df = rotate_voids(df, frame)
        if frame != NATIVE_FRAME:
            geometry_check(df, cosmo, os.path.basename(src) + f' [rotado a {frame}]')
        n0 = len(df)
        if cols.get('completeness') is not None:
            df = df[df['completeness'] > 1.9]
        df = _delta_cut(df, cfg.get('delta_value'))
        df = _footprint_cut(df, F, cfg, os.path.basename(src), cat)
        print(f"[catalog] {os.path.basename(src)}: {n0} -> {len(df)} voids")
        np.savetxt(single, df[['R_void', 'l', 'b', 'z']].values, fmt='%.6f')
        return single, template
    files = sorted(glob.glob(os.path.join(wf, cfg['seed_glob'])))
    if len(files) < n_seeds:
        raise FileNotFoundError(f"N_seeds={n_seeds} pero '{cfg['seed_glob']}' encuentra {len(files)} archivos")
    files = files[:n_seeds]
    print(f"[catalog] {n_seeds} semillas: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    for k, src in enumerate(files):
        raw = _read_websky_raw(src)
        if k == 0:
            column_summary(raw, os.path.basename(src))
        df = _to_pipeline(raw, cols)
        geometry_check(df, cosmo, os.path.basename(src))
        df = rotate_voids(df, frame)
        if frame != NATIVE_FRAME and k == 0:
            geometry_check(df, cosmo, os.path.basename(src) + f' [rotado a {frame}]')
        df = _footprint_cut(df, F, cfg, os.path.basename(src), cat)
        np.savetxt(template.format(k + 1), df[PIPELINE_COLS].values, fmt='%.6f')
    return single, template


#%% Mapas
def _read_cols(path):
    d = np.loadtxt(path)
    return d[:, 0].astype(int), d


def kappa_raw_alm(cfg, maps_dir):
    """kap.fits -> alm(L<=lmax), L<2 a cero. Cacheado."""
    src = os.path.join(cfg['websky_folder'], cfg['kappa_file'])
    base = os.path.splitext(os.path.basename(src))[0]
    ptag = '_pwdec' if cfg['deconvolve_pixwin'] else ''
    path = os.path.join(maps_dir, f"websky_{base}_alm_lmax{cfg['lmax']}_it{cfg['map2alm_iter']}{ptag}.fits")
    if os.path.exists(path) and not cfg.get('force_map', False):
        print(f"[map] kappa raw: cacheado {os.path.basename(path)}")
        return path
    print(f"[map] kappa raw: leyendo {src} ...")
    m = hp.read_map(src, field=0, dtype=np.float64)
    nside_in = hp.get_nside(m)
    lmax = min(cfg['lmax'], 3 * nside_in - 1)
    print(f"[map] nside_in={nside_in}, mean={np.mean(m):.3e}, std={np.std(m):.3e}, lmax={lmax}")
    alm = hp.map2alm(m, lmax=lmax, iter=cfg['map2alm_iter'])
    del m
    fl = np.ones(lmax + 1); fl[:2] = 0.0
    if cfg['deconvolve_pixwin']:
        fl[2:] /= hp.pixwin(nside_in, lmax=lmax)[2:]
    hp.write_alm(path, hp.almxfl(alm, fl), overwrite=True)
    return path


def kappa_obsPR4_alm(cfg, raw_path, maps_dir):
    """raw + ruido gaussiano N_L^PR4. El Wiener lo aplica S1 con pr4_noise_file (como en datos)."""
    nfile = os.path.join(cfg['data_folder'], cfg['pr4_noise_file'])
    L, d = _read_cols(nfile)
    if d.shape[1] < 3:
        raise ValueError(f"{nfile}: se esperan 3 columnas (L, N_L, C_L+N_L) para ruido + Wiener")
    path = raw_path.replace('.fits', f"_noisePR4s{cfg['noise_seed']}.fits")
    if os.path.exists(path) and not cfg.get('force_map', False):
        print(f"[map] kappa obsPR4: cacheado {os.path.basename(path)}")
        return path, nfile
    alm = hp.read_alm(raw_path)
    lmax = hp.Alm.getlmax(len(alm))
    NL = np.zeros(lmax + 1)
    ok = L <= lmax
    NL[L[ok]] = d[ok, 1]
    NL[:2] = 0.0
    if L.max() < lmax:
        print(f"[map] (!) {os.path.basename(nfile)} llega a L={L.max()} < lmax={lmax}")
    np.random.seed(cfg['noise_seed'])
    hp.write_alm(path, alm + hp.synalm(NL, lmax=lmax), overwrite=True)
    print(f"[map] kappa obsPR4: + ruido N_L de {os.path.basename(nfile)} (seed={cfg['noise_seed']})")
    return path, nfile


def alm_in_frame(path, frame, force=False):
    """alm (frame nativo G) -> frame pedido, misma rotación que los voids. Cacheado."""
    if frame == NATIVE_FRAME:
        return path
    out = path.replace('.fits', f'_in{frame}.fits')
    if os.path.exists(out) and not force:
        print(f"[map] {os.path.basename(out)}: cacheado")
        return out
    alm = hp.read_alm(path)
    hp.write_alm(out, _rotator(NATIVE_FRAME, frame).rotate_alm(alm), overwrite=True)
    print(f"[map] {os.path.basename(path)}: alm rotado {NATIVE_FRAME} -> {frame}")
    return out


def decide_act_rotation(cfg, niall_path, act_mask_path, raw_alm_path):
    """
    ¿Cómo se relaciona el mapa de Niall (C) con el frame nativo de Websky?
      'identity': Niall usó el nativo como C sin rotar   'rotate': Niall rotó el nativo G->C
    'auto': coeficiente de correlación (L en [100, 1000], dentro de ACT) del mapa de Niall con
    kappa y tSZ de Websky, sin rotar y rotados G->C. Los foregrounds reconstruidos trazan la
    estructura de la simulación: la hipótesis correcta da |r| significativa, la otra ~0.
    Returns True si hay que rotar voids y kappa a C.
    """
    mode = cfg.get('act_native_mode', 'auto')
    if mode in ('identity', 'rotate'):
        print(f"[act-frame] act_native_mode='{mode}' (fijado en el config)")
        return mode == 'rotate'
    if mode != 'auto':
        raise ValueError(f"act_native_mode desconocido: {mode}")
    k = hp.read_map(niall_path, dtype=np.float64)
    ns = hp.get_nside(k)
    msk = (hp.ud_grade(hp.read_map(act_mask_path, dtype=np.float64), ns) >= 0.9).astype(float)
    lmax = min(1000, 3 * ns - 1)
    tracers = {}
    alm = hp.read_alm(raw_alm_path)
    l0 = hp.Alm.getlmax(len(alm))
    tracers['kappa Websky'] = hp.resize_alm(alm, l0, l0, lmax, lmax) if l0 > lmax else alm
    tpath = os.path.join(cfg['websky_folder'], cfg.get('tsz_check_map') or '')
    if cfg.get('tsz_check_map') and os.path.exists(tpath):
        tracers['tSZ Websky'] = hp.map2alm(hp.ud_grade(hp.read_map(tpath, dtype=np.float64), ns), lmax=lmax)
    L = np.arange(lmax + 1)
    sel = L >= 100
    sigma = 1.0 / np.sqrt(np.sum(2 * L[sel] + 1) * msk.mean())   # ruido de r si no hay correlación
    ckk = hp.anafast(k * msk, lmax=lmax)
    rot = _rotator(NATIVE_FRAME, 'C')
    score = {'identity': 0.0, 'rotate': 0.0}
    for name, a in tracers.items():
        r = {}
        for hyp, aa in (('identity', a), ('rotate', rot.rotate_alm(a))):
            m = hp.alm2map(aa, ns, lmax=lmax) * msk
            cxy = hp.anafast(k * msk, m, lmax=lmax)
            cyy = hp.anafast(m, lmax=lmax)
            r[hyp] = np.sum(cxy[sel]) / np.sqrt(np.sum(ckk[sel]) * np.sum(cyy[sel]))
            score[hyp] += abs(r[hyp]) / sigma
        print(f"[act-frame] Niall x {name:13s}: r(sin rotar)={r['identity']:+.4f}  "
              f"r(rotado G->C)={r['rotate']:+.4f}   (ruido ~{sigma:.4f})")
    best = max(score, key=score.get)
    other = 'rotate' if best == 'identity' else 'identity'
    if score[best] < 5 * len(tracers) or score[best] < 3 * score[other]:
        raise ValueError("[act-frame] la correlación no distingue las dos hipótesis; fijá "
                         "'act_native_mode' a mano ('identity' o 'rotate') después de consultarlo con Niall.")
    print(f"[act-frame] -> '{best}': " + ("voids y kappa ROTADOS G->C para ACT/ACT-PR4" if best == 'rotate'
                                         else "voids y kappa SIN ROTAR para ACT/ACT-PR4 (nativo = C)"))
    return best == 'rotate'


#%% Registro de WEBSKY en S1
def install_websky(cat_spec, map_spec, cov_spec=None):
    orig_cat, orig_map = _ORIG_SPECS

    def _cat(catalog, data_folder):
        if str(catalog).upper() == 'WEBSKY':
            return dict(cat_spec)
        return orig_cat(catalog, data_folder)

    def _map(cmb_map, data_folder, act_smooth_arcmin=0.0):
        key = str(cmb_map).upper()
        if key == 'WEBSKY':
            return dict(map_spec)
        if key == COV_KEY and cov_spec is not None:
            return dict(cov_spec)
        return orig_map(cmb_map, data_folder, act_smooth_arcmin)

    s1._catalog_spec, s1._map_spec = _cat, _map


#%% Resultados guardados
def _resolve_pkl(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, 'Data_FullRun_*.pkl')))
        if len(found) != 1:
            raise FileNotFoundError(f"{path}: {len(found)} archivos Data_FullRun_*.pkl (se espera 1)")
        return found[0]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _pick(results, prefer=('merged', 'concat', 'single')):
    for kind in prefer:
        for r in results:
            if r.get('catalog') == kind and int(r.get('bin_id', 0)) == 0:
                return r
    return results[0]


def _sel_prefix(cfg):
    return (f"{cfg['binning_mode']}_{cfg['n_bins']}bins_{cfg['exec_mode']}_"
            f"{cfg['zmin']}_{cfg['zmax']}_{cfg['rmin']}_{cfg['rmax']}_maxRv{cfg['max_Rvoid']:.1f}_")


def load_saved_websky(cfg, runs, cats):
    """{(cat|'ALL', run_key): {'res','pkl','label','filter','run'}}"""
    lens = os.path.join(cfg['output_folder'], 'lensing')
    out = {}
    for run in runs:
        for cat in (cats if run['footprint'] else [None]):
            lab, filt = map_label(cfg, run), filter_label(run)
            pre = f"{cat_label(cat)}_{lab}_{_sel_prefix(cfg)}"
            cands = [d for d in glob.glob(os.path.join(lens, pre + '*'))
                     if os.path.isdir(d) and d.endswith('_' + filt)]
            key = (cat or 'ALL', run['key'])
            if not cands:
                print(f"[load] (!) sin corrida guardada para {key}: {pre}*_{filt}")
                continue
            d = max(cands, key=os.path.getmtime)
            pkl = _resolve_pkl(d)
            with open(pkl, 'rb') as f:
                res = _pick(pickle.load(f)['results'])
            out[key] = {'res': res, 'pkl': pkl, 'label': lab, 'filter': filt.split('_cov')[0], 'run': run}
            print(f"[load] {key} <- {os.path.basename(d)}")
    return out


def _folder_filter(pkl):
    """Filtro de la corrida a partir del nombre de la carpeta de S1."""
    name = os.path.basename(os.path.dirname(pkl))
    m = re.search(r'_d23_(?:all|gt[\d.]+|lt[\d.]+)_(.+?)(?:_cov[^_]+)?$', name)
    return m.group(1) if m else '*'


def _null_rand_profiles(pkl_path, res, release, filt):
    """Perfiles de randoms cacheados por Parallel_module (null_tests_*.npz), si existen."""
    if not pkl_path:
        return None
    root = os.path.join(os.path.dirname(os.path.dirname(pkl_path)), 'Cache_Stacks')
    pat = os.path.join(root, '**', f"null_tests_{release}_{res.get('key')}_*_N{res.get('n_voids')}_"
                                    f"*_{filt}_maxRv*.npz")
    found = glob.glob(pat, recursive=True)
    if not found:
        return None
    f = max(found, key=os.path.getmtime)
    try:
        R = np.asarray(np.load(f)['null_profiles_rand'], float)
        print(f"[chi2] randoms cacheados: {os.path.relpath(f, root)} ({len(R)} perfiles)")
        return R if R.ndim == 2 and len(R) > 1 else None
    except Exception as e:
        print(f"[chi2] no pude leer {f}: {e}")
        return None


#%% Estadística
def _cov_rand(res):
    """cov del perfil neto según randoms: cov_cmb * (1 + 1/N_rand)."""
    C, nr = res.get('cov_cmb'), int(res.get('n_randoms_done', 0) or 0)
    if C is None or nr < 2:
        return None, nr
    return np.asarray(C, float) * (1.0 + 1.0 / nr), nr


def _net(res):
    net, _, _ = fm._net_and_err(res)
    C, _ = _cov_rand(res)
    sig = np.sqrt(np.diag(C)) if C is not None else np.full_like(net, np.nan)
    return np.asarray(net), sig


def chi2_test(a, b=None, rand_a=None, rand_b=None):
    """
    chi2 de (net_a - net_b) [o net_a contra 0 si b es None] con C = C_a + C_b (randoms).
    a, b: dict(res=..., pkl=..., release=..., filter=...).
    """
    na, _ = _net(a['res'])
    Ca, nra = _cov_rand(a['res'])
    if Ca is None:
        raise ValueError("sin cov_cmb (¿n_rand_factor=0?)")
    if b is not None:
        nb, _ = _net(b['res'])
        Cb, nrb = _cov_rand(b['res'])
        C = Ca + (Cb if Cb is not None else 0.0)
        N = min(nra, nrb) if Cb is not None else nra
        delta = na - nb
    else:
        C, N, delta = Ca, nra, na
    good = np.isfinite(delta) & np.isfinite(np.diag(C))
    p = int(good.sum())
    h = (N - p - 2) / (N - 1)
    Cg, dg = C[np.ix_(good, good)], delta[good]
    w, V = np.linalg.eigh(Cg)
    if w.min() <= 0:
        print(f"[chi2] (!) covarianza no definida positiva (min autovalor={w.min():.2e}); uso pinv")
    Cinv = np.linalg.pinv(Cg)
    chi2 = float(h * dg @ Cinv @ dg)
    from scipy.stats import chi2 as _chi2
    pte = float(_chi2.sf(chi2, p))
    order = np.argsort(w)[::-1]
    contrib = h * (V[:, order].T @ dg) ** 2 / w[order]

    emp = None
    Ra = _null_rand_profiles(a.get('pkl'), a['res'], a.get('release', '*'), a.get('filter', '*'))
    if Ra is not None:
        dd = Ra[:, good] - np.nanmean(Ra[:, good], 0)
        if b is not None:
            Rb = _null_rand_profiles(b.get('pkl'), b['res'], b.get('release', '*'), b.get('filter', '*'))
            if Rb is not None:
                n = min(len(Ra), len(Rb))
                dd = dd[:n] - (Rb[:n, good] - np.nanmean(Rb[:, good], 0))
        dd = dd[np.all(np.isfinite(dd), axis=1)]
        emp = h * np.einsum('ki,ij,kj->k', dd, Cinv, dd)
    out = {'chi2': chi2, 'p': p, 'h': h, 'N_rand': N, 'pte': pte, 'eig': w[order],
           'contrib': contrib, 'emp': emp,
           'pte_emp': None if emp is None else float((np.sum(emp >= chi2) + 1) / (len(emp) + 1))}
    msg = f"chi2={chi2:.1f} / {p} bins (cov randoms, N_rand={N}, h={h:.3f}) PTE={pte:.3f}"
    if emp is not None:
        msg += f" | PTE empírico={out['pte_emp']:.3f} ({len(emp)} randoms)"
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
    a1.plot(x, _chi2.pdf(x, p), color='xkcd:steel blue', lw=1.8, label=rf"$\chi^2_{{{p}}}$")
    xs = x[x >= ch['chi2']]
    a1.fill_between(xs, _chi2.pdf(xs, p), color='xkcd:steel blue', alpha=0.25)
    a1.axvline(ch['chi2'], color='xkcd:crimson', lw=2, label=rf"observado = {ch['chi2']:.1f}")
    ttl = f"PTE = {ch['pte']:.3f}" + (f"   (empírico {ch['pte_emp']:.3f})" if ch['pte_emp'] is not None else "")
    a1.set_title(ttl, fontsize=11)
    a1.set_xlabel(r'$\chi^2$'); a1.set_ylabel('densidad'); a1.legend(fontsize=9); a1.grid(alpha=0.25)
    k = np.arange(1, len(ch['contrib']) + 1)
    a2.bar(k, ch['contrib'], color='xkcd:crimson', alpha=0.75)
    a2.axhline(1, color='k', ls='--', lw=1, alpha=0.7)
    a2.set_xlabel('autovector de C (autovalor decreciente)')
    a2.set_ylabel(r'$h\,(v_i\cdot\Delta)^2/\lambda_i$')
    a2b = a2.twinx()
    a2b.semilogy(k, ch['eig'] / ch['eig'][0], 'o-', color='xkcd:steel blue', ms=3, lw=1)
    a2b.set_ylabel(r'$\lambda_i/\lambda_1$', color='xkcd:steel blue')
    a2.set_title(f"h = {ch['h']:.3f} (N_rand={ch['N_rand']}, p={p})", fontsize=11)
    a2.grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[chi2] figura: {out_path}")


#%% Figuras
def _band(ax, res, color, label, scale=1e3, ls='-'):
    r = np.asarray(res['r_frac'])
    net, sig = _net(res)
    ax.fill_between(r, (net - sig) * scale, (net + sig) * scale, color=color, alpha=0.22)
    ax.plot(r, net * scale, ls, color=color, lw=1.8, label=f"{label} (N={res.get('n_voids', '?')})")


def _frame_axes(ax, cfg, ylabel=r'$\kappa_{\rm net}\;[10^{-3}]$'):
    ax.axhline(0, color='k', ls=':', alpha=0.6); ax.axvline(1, color='gray', ls='--', alpha=0.6)
    ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.set_xlim(-0.1, cfg['max_Rvoid'] + 0.1)


def _sel_tag(cfg):
    return f"{cfg['zmin']}_{cfg['zmax']}_{cfg['rmin']}_{cfg['rmax']}"


def plot_overview(ws, cfg, cat, out_dir):
    """Todas las corridas de Websky de un catálogo: arriba kappa puro (geometría), abajo observados."""
    top = [k for k in ('raw_FULL', 'raw_PR4', 'raw_ACT') if (cat if k != 'raw_FULL' else 'ALL', k) in ws]
    bot = [k for k in ('obsPR4_PR4', 'obsPR4_ACTPR4', 'noiseACT_ACT') if (cat, k) in ws]
    if not top and not bot:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True, gridspec_kw={'hspace': 0.08})
    for k in top:
        _band(axes[0], ws[(cat if k != 'raw_FULL' else 'ALL', k)]['res'], RUN_COLORS[k], RUN_NAMES[k])
    for k in bot:
        _band(axes[1], ws[(cat, k)]['res'], RUN_COLORS[k], RUN_NAMES[k])
    for ax, t in zip(axes, ('κ puro (sin ruido ni filtro)', 'como se observaría')):
        _frame_axes(ax, cfg); ax.legend(fontsize=8, loc='lower right'); ax.set_title(t, fontsize=10)
    axes[1].set_xlabel(r'$r/R_v$')
    fig.suptitle(f"Websky | voids en footprint {cat} | bandas: σ de randoms", fontsize=11)
    out = os.path.join(out_dir, f"Websky_overview_{cat}_{_sel_tag(cfg)}.pdf")
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[plot] {out}")


def plot_cats(ws, cfg, cats, out_dir):
    """WH vs BOSS para cada corrida con footprint."""
    keys = [k for k in RUN_NAMES if k != 'raw_FULL' and all((c, k) in ws for c in cats)]
    if len(cats) < 2 or not keys:
        return
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.2 * n), sharex=True, squeeze=False,
                             gridspec_kw={'hspace': 0.1})
    for ax, k in zip(axes[:, 0], keys):
        for c in cats:
            _band(ax, ws[(c, k)]['res'], CAT_COLORS[c], f"{c}")
        _frame_axes(ax, cfg); ax.legend(fontsize=8, loc='lower right', title=RUN_NAMES[k], title_fontsize=8)
    axes[-1, 0].set_xlabel(r'$r/R_v$')
    out = os.path.join(out_dir, f"Websky_WHvsBOSS_{_sel_tag(cfg)}.pdf")
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[plot] {out}")


def _load_data(path):
    pkl = _resolve_pkl(path)
    with open(pkl, 'rb') as f:
        dd = pickle.load(f)
    res, par = _pick(dd['results']), dd.get('parameters', {})
    return {'res': res, 'pkl': pkl, 'release': '*', 'filter': _folder_filter(pkl), 'par': par}


def _check_r(a, b, what):
    ok = np.allclose(np.asarray(a['r_frac']), np.asarray(b['r_frac']))
    if not ok:
        print(f"[compare] (!) r_frac distinto en {what}: revisar max_Rvoid/Rvoid_bin. Salteo.")
    return ok


def compare_pr4_like(ws, cfg, cat, survey, data_path, out_dir):
    """PR4 o ACT-PR4: datos vs Websky obsPR4 (+ kappa puro de referencia) y chi2."""
    obs_key = 'obsPR4_PR4' if survey == 'PR4' else 'obsPR4_ACTPR4'
    ref_keys = ['raw_PR4', 'raw_FULL'] if survey == 'PR4' else ['raw_ACT']
    if (cat, obs_key) not in ws:
        print(f"[compare] (!) {cat}/{survey}: falta la corrida {obs_key}")
        return
    D = _load_data(data_path)
    print(f"[compare] {cat}/{survey}: datos <- {D['pkl']}")
    W = ws[(cat, obs_key)]
    if not _check_r(D['res'], W['res'], f"{cat}/{survey}"):
        return
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    r = np.asarray(D['res']['r_frac'])
    nd, sd = _net(D['res'])
    ax.errorbar(r, nd * 1e3, yerr=sd * 1e3, fmt='o-', color='k', capsize=3, zorder=5,
                label=f"Datos {cat} x {D['par'].get('cmb_map', survey)} (N={D['res'].get('n_voids', '?')})")
    for k in ref_keys:
        kk = ('ALL' if k == 'raw_FULL' else cat, k)
        if kk in ws:
            _band(ax, ws[kk]['res'], RUN_COLORS[k], RUN_NAMES[k], ls='--')
    _band(ax, W['res'], RUN_COLORS[obs_key], RUN_NAMES[obs_key])
    _frame_axes(ax, cfg); ax.legend(fontsize=8, loc='lower right')

    ch = chi2_test(D, {'res': W['res'], 'pkl': W['pkl'], 'release': W['label'], 'filter': W['filter']})
    print(f"[chi2] {cat}/{survey} datos vs {RUN_NAMES[obs_key]}: {ch['msg']}")
    nw, sw = _net(W['res'])
    with np.errstate(divide='ignore', invalid='ignore'):
        axr.plot(r, (nd - nw) / np.sqrt(sd ** 2 + sw ** 2), 'o-', color=RUN_COLORS[obs_key])
    for y in (-2, 2):
        axr.axhline(y, color='gray', ls='--', alpha=0.7)
    axr.axhline(0, color='k', alpha=0.5); axr.set_ylim(-5, 5); axr.grid(alpha=0.25)
    axr.set_xlabel(r'$r/R_v$'); axr.set_ylabel(r'$\Delta/\sigma_{\rm rand}$')
    tag = f"{cat}_{survey.replace('-', '')}_{_sel_tag(cfg)}"
    out = os.path.join(out_dir, f"WebskyVsData_{tag}.pdf")
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[compare] {out}")
    plot_chi2(ch, os.path.join(out_dir, f"Chi2_WebskyVsData_{tag}.pdf"),
              f"Datos {cat} x {survey} vs {RUN_NAMES[obs_key]}  |  cov de randoms")


def compare_act(ws, cfg, cat, data_path, out_dir):
    """ACT: datos vs Websky ACT sin lensing (fg + ruido) + kappa puro, cociente de ruidos y chi2 del fg vs 0."""
    nk = (cat, 'noiseACT_ACT')
    if nk not in ws:
        print(f"[compare] (!) {cat}/ACT: falta la corrida noiseACT_ACT")
        return
    Wn = ws[nk]
    D = _load_data(data_path) if data_path else None
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    r = np.asarray(Wn['res']['r_frac'])
    if D is not None and _check_r(D['res'], Wn['res'], f"{cat}/ACT"):
        print(f"[compare] {cat}/ACT: datos <- {D['pkl']}")
        nd, sd = _net(D['res'])
        ax.errorbar(r, nd * 1e3, yerr=sd * 1e3, fmt='o-', color='k', capsize=3, zorder=5,
                    label=f"Datos {cat} x ACT (N={D['res'].get('n_voids', '?')})")
        _, sn = _net(Wn['res'])
        axr.plot(r, sn / sd, 'o-', color=RUN_COLORS['noiseACT_ACT'])
        axr.axhline(1, color='k', alpha=0.5)
        print(f"[compare] {cat}/ACT: σ_rand(Websky sin lensing)/σ_rand(datos) por bin: "
              f"min={np.nanmin(sn / sd):.2f} med={np.nanmedian(sn / sd):.2f} max={np.nanmax(sn / sd):.2f}")
    if (cat, 'raw_ACT') in ws:
        _band(ax, ws[(cat, 'raw_ACT')]['res'], RUN_COLORS['raw_ACT'],
              RUN_NAMES['raw_ACT'] + ' [sin filtro ACT]', ls='--')
    _band(ax, Wn['res'], RUN_COLORS['noiseACT_ACT'], RUN_NAMES['noiseACT_ACT'])
    _frame_axes(ax, cfg); ax.legend(fontsize=8, loc='lower right')
    axr.set_ylim(0, 2); axr.grid(alpha=0.25)
    axr.set_xlabel(r'$r/R_v$'); axr.set_ylabel(r'$\sigma^{\rm WS}_{\rm rand}/\sigma^{\rm datos}_{\rm rand}$')
    tag = f"{cat}_ACT_{_sel_tag(cfg)}"
    out = os.path.join(out_dir, f"WebskyVsData_{tag}.pdf")
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[compare] {out}")
    ch = chi2_test({'res': Wn['res'], 'pkl': Wn['pkl'], 'release': Wn['label'], 'filter': Wn['filter']})
    print(f"[chi2] {cat}/ACT foregrounds+ruido contra 0: {ch['msg']}")
    plot_chi2(ch, os.path.join(out_dir, f"Chi2_ACTforegrounds_vs0_{tag}.pdf"),
              f"Websky ACT sin lensing (fg + ruido) contra 0 | voids {cat} | cov de randoms")


#%% Main
def run_one(cfg, run, cat, cat_spec, maps):
    lab = map_label(cfg, run)
    fr = run['frame']
    if run['field'] == 'raw':
        kind, path, nside, filt, nlkk = 'alm', maps[('raw', fr)], cfg['nside'], 'none', None
    elif run['field'] == 'obsPR4':
        kind, path, nside, filt, nlkk = 'alm', maps[('obsPR4', fr)], cfg['nside'], 'wiener', maps['pr4_nlkk']
    else:   # noiseACT
        kind, path, nside, filt, nlkk = 'map', maps['noiseACT'], maps['noiseACT_nside'], 'none', None
    cov_spec = None
    if run['cov'] == 'ACT':
        cov_spec = {'label': 'ACTmask', 'kind': 'map', 'frame': 'galactic', 'nside': 512,
                    'full_sky': False, 'apply_wiener': False, 'klm': None,
                    'mask': maps[('mask', 'ACT', fr)], 'nlkk': None}
    print(f"\n================= WEBSKY | {cat_label(cat)} | {run['key']} -> {lab} "
          f"[{filt}{', cobertura ACT' if cov_spec else ''}, frame {fr}] =================")
    install_websky(cat_spec, {'label': lab, 'kind': kind, 'frame': 'galactic', 'nside': nside,
                              'full_sky': run['full_sky'], 'apply_wiener': filt == 'wiener',
                              'klm': path, 'mask': maps[('mask', run['mask'], fr)], 'nlkk': nlkk},
                   cov_spec)
    c = dict(cfg)
    c.update(filter_mode=filt, coverage_map=COV_KEY if cov_spec else None)
    s1._run_single_map(c)


def main(cfg=config):
    cfg = dict(cfg)
    cfg.update(void_catalog='WEBSKY', cmb_map='WEBSKY', coverage_map=None)
    cats, runs = catalogs_of(cfg), build_runs(cfg)

    lens_out = os.path.join(cfg['output_folder'], 'lensing')
    maps_dir = os.path.join(cfg['output_folder'], 'maps')
    os.makedirs(lens_out, exist_ok=True); os.makedirs(maps_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(lens_out, 'websky_run.log'))
    print(f"\n### WEBSKY | voids={cats} | corridas={[r['key'] for r in runs]} | "
          f"frames={sorted({r['frame'] for r in runs})} | compare_only={cfg.get('compare_only')} ###")

    if not cfg.get('compare_only'):
        with open(os.path.join(lens_out, 'Websky_pipeline_config.json'), 'w') as f:
            json.dump(cfg, f, indent=4, default=str)
        cosmo = set_cosmology(cfg['cosmology'])

        print("\n>>> Mapas y máscaras")
        force = cfg.get('force_map', False)
        raw_G = kappa_raw_alm(cfg, maps_dir)
        maps = {}
        obs_G = None
        if any(r['field'] == 'obsPR4' for r in runs):
            obs_G, maps['pr4_nlkk'] = kappa_obsPR4_alm(cfg, raw_G, maps_dir)
        for r in runs:
            for m in {r['mask']} | ({'ACT'} if r['cov'] else set()):
                if ('mask', m, r['frame']) not in maps:
                    maps[('mask', m, r['frame'])] = mask_path(cfg, m, r['frame'], maps_dir)

        rotate_C = False
        if any(r['frame'] == 'C' for r in runs):
            maps['noiseACT'] = os.path.join(cfg['websky_folder'], cfg['act_noise_map'])
            have_niall = os.path.exists(maps['noiseACT'])
            if any(r['field'] == 'noiseACT' for r in runs) and not have_niall:
                raise FileNotFoundError(maps['noiseACT'])
            if have_niall:
                maps['noiseACT_nside'] = hp.get_nside(hp.read_map(maps['noiseACT'], dtype=np.float32))
            if cfg.get('act_native_mode', 'auto') == 'auto' and not have_niall:
                raise ValueError("act_native_mode='auto' necesita el mapa de Niall; fijalo a mano.")
            print("\n>>> Relación frame nativo de Websky <-> ACT")
            rotate_C = decide_act_rotation(cfg, maps['noiseACT'], maps[('mask', 'ACT', 'C')], raw_G)

        for r in runs:
            fr = r['frame']
            rot = fr == 'C' and rotate_C
            if r['field'] == 'raw' and ('raw', fr) not in maps:
                maps[('raw', fr)] = alm_in_frame(raw_G, 'C', force) if rot else raw_G
            if r['field'] == 'obsPR4' and ('obsPR4', fr) not in maps:
                maps[('obsPR4', fr)] = alm_in_frame(obs_G, 'C', force) if rot else obs_G

        print("\n>>> Catálogos")
        cat_specs = {}
        def _spec(single, tpl, cat):
            return {'label': cat_label(cat), 'seed_template': tpl, 'single_file': single,
                    'lon_col': 'l', 'lat_col': 'b', 'frame': 'galactic'}   # 'galactic' == identidad en S1
        if any(not r['footprint'] for r in runs):
            single, tpl = prepare_catalogs(cfg, cosmo, os.path.join(cfg['output_folder'], 'catalogs', 'fullsky_G'))
            cat_specs[(None, 'G')] = _spec(single, tpl, None)
        for fr in sorted({r['frame'] for r in runs if r['footprint']}):
            for cat in cats:
                F = build_footprint(cfg, cat, fr, maps_dir)
                plot_geometry(F, [maps[('mask', m, fr)] for m in ('PR4', 'ACT') if ('mask', m, fr) in maps],
                              os.path.join(maps_dir, f"geometry_{cat}_in{fr}.png"),
                              f"Websky en {fr}: footprint {cat} + máscaras")
                rot_to = 'C' if (fr == 'C' and rotate_C) else None
                single, tpl = prepare_catalogs(cfg, cosmo,
                                               os.path.join(cfg['output_folder'], 'catalogs',
                                                            f"{cat}_{fr}{'rot' if rot_to else ''}"),
                                               F, cat, rot_to=rot_to)
                cat_specs[(cat, fr)] = _spec(single, tpl, cat)

        for run in runs:
            for cat in (cats if run['footprint'] else [None]):
                run_one(cfg, run, cat, cat_specs[(cat, run['frame'])], maps)

    print("\n>>> Resultados guardados")
    ws = load_saved_websky(cfg, runs, cats)
    for cat in cats:
        plot_overview(ws, cfg, cat, lens_out)
    plot_cats(ws, cfg, cats, lens_out)

    print("\n>>> Comparación con datos")
    for cat in cats:
        cd = (cfg.get('compare_data') or {}).get(cat, {}) or {}
        surveys = [s.upper() for s in cfg['surveys']]
        for sv in ('PR4', 'ACT-PR4'):
            if sv in surveys and cd.get(sv):
                compare_pr4_like(ws, cfg, cat, sv, cd[sv], lens_out)
        if 'ACT' in surveys:
            compare_act(ws, cfg, cat, cd.get('ACT'), lens_out)

    if not cfg.get('compare_only'):
        with open(os.path.join(lens_out, 'SUCCESS_websky'), 'w') as f:
            f.write('Done.')
    print("\n=== FINISHED ===")
    return ws


if __name__ == "__main__":
    main()
