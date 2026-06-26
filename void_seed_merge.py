
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from astropy.cosmology import Planck18
import healpy as hp

COLS_MULTISEED = ['R_void', 'l', 'b', 'z', 'x', 'y', 'z_cart',
                  'delta_int', 'delta_23', 'completeness', 'delta_LOS']


def load_seed_catalogs(path_template, n_seeds, cols=COLS_MULTISEED,
                       base_filter=None):
    """
    path_template : str con un campo {:03d}, e.g.
        '/.../WenHan_voids_z0.6/voids_z0.6_{:03d}.dat'  (1-indexado)
    base_filter   : callable(df)->df aplicado POR semilla (cortes z, R,
                    completeness, etc.). Si None, no filtra.
    """
    frames = []
    for s in range(n_seeds):
        df = pd.read_csv(path_template.format(s + 1), sep=r'\s+',
                         names=cols, header=None)
        if base_filter is not None:
            df = base_filter(df)
        df = df.copy()
        df['seed'] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"[seed_merge] {n_seeds} semillas cargadas -> {len(out)} voids (pre-merge).")
    return out


def _radec_to_cart_comoving(l, b, z, h=None):
    """l,b [deg] galácticas, z -> (x,y,z) comóvil en Mpc/h."""
    if h is None:
        h = Planck18.h
    chi = Planck18.comoving_distance(z).value * h        # Mpc/h
    lon, lat = np.radians(l), np.radians(b)
    return np.column_stack([chi * np.cos(lat) * np.cos(lon),
                            chi * np.cos(lat) * np.sin(lon),
                            chi * np.sin(lat)])


def merge_seeds(df, eps_mpch=8.0, min_frac=0.2, n_seeds=None,
                use_catalog_xyz=True, h=None):
    """
    Parameters
    ----------
    df              : DataFrame concatenado (salida de load_seed_catalogs).
    eps_mpch        : radio de linking de DBSCAN en Mpc/h. EMPEZAR CHICO.
    min_frac        : fracción mínima de semillas para considerar el void robusto
                      (min_samples = round(min_frac * n_seeds)).
    use_catalog_xyz : usar columnas x,y,z_cart del catálogo (se asumen comóvil
                      Mpc/h). Si False, recomputa desde (l,b,z) -> cross-check útil.

    Returns
    -------
    merged : DataFrame con un void por fila (l,b,z,R_void promediados + diagnósticos).
    labels : etiqueta de grupo DBSCAN por punto original (-1 = ruido).
    """
    if n_seeds is None:
        n_seeds = df['seed'].nunique()

    if use_catalog_xyz and {'x', 'y', 'z_cart'}.issubset(df.columns):
        XYZ = df[['x', 'y', 'z_cart']].values.astype(float)
    else:
        XYZ = _radec_to_cart_comoving(df['l'].values, df['b'].values,
                                      df['z'].values, h)

    min_samples = max(2, int(round(min_frac * n_seeds)))
    labels = DBSCAN(eps=eps_mpch, min_samples=min_samples).fit_predict(XYZ)

    n_groups = labels.max() + 1
    n_noise = int(np.sum(labels == -1))
    print(f"[seed_merge] DBSCAN eps={eps_mpch} Mpc/h, min_samples={min_samples}: "
          f"{n_groups} real voids, {n_noise} points discarded (noise).")

    rows = []
    for g in range(n_groups):
        m = labels == g
        sub = df[m]
        lon, lat = np.radians(sub['l'].values), np.radians(sub['b'].values)
        ux, uy, uz = (np.mean(np.cos(lat) * np.cos(lon)),
                      np.mean(np.cos(lat) * np.sin(lon)),
                      np.mean(np.sin(lat)))
        norm = np.sqrt(ux * ux + uy * uy + uz * uz)
        row = {
            'void_id': g,
            'l': np.degrees(np.arctan2(uy, ux)) % 360.0,
            'b': np.degrees(np.arcsin(np.clip(uz / norm, -1, 1))),
            'z': sub['z'].mean(),
            'R_void': sub['R_void'].mean(),
            'R_void_std': sub['R_void'].std(),
            'n_seed_members': int(sub['seed'].nunique()),
            'n_points': int(m.sum()),
            'sep_scatter_deg': np.degrees(np.std(np.column_stack([lon, lat]), axis=0).mean()),
        }
        for extra in ('delta_23', 'completeness', 'delta_LOS'):
            if extra in sub.columns:
                row[extra] = sub[extra].mean()
        rows.append(row)

    merged = pd.DataFrame(rows)
    if len(merged):
        print(f"[seed_merge] Catálogo final: {len(merged)} voids. "
              f"Mediana semillas/void = {np.median(merged['n_seed_members']):.0f}; "
              f"R_void scatter medio = {np.nanmean(merged['R_void_std']):.2f} Mpc/h.")
    return merged, labels


def weight_map_mollview(df, nside=128, outpath=None,
                        title="Voids Centre Density"):
    """
    DIAGNÓSTICO VISUAL (no es la deduplicación).

    Pinta en un mollview cuántos centros de void (de TODAS las semillas) caen en
    cada píxel. Sirve para ver de un vistazo qué tan concentrado/repetible es el
    recentrado.

    OJO: es una proyección ANGULAR -> colapsa la dirección LOS (z). Dos voids a
    igual (l,b) y distinto z se superponen en el mismo píxel. Por eso NO sirve
    para seleccionar cúmulos en una cáscara 2.5D: para el catálogo usar
    merge_seeds() (clustering en 3D comóvil). Esto es sólo inspección.
    """
    import matplotlib.pyplot as plt
    npix = hp.nside2npix(nside)
    wmap = np.zeros(npix)
    pix = hp.ang2pix(nside, df['l'].values, df['b'].values, lonlat=True)
    np.add.at(wmap, pix, 1.0)
    n_per_pix = wmap.copy()
    wmap[wmap == 0] = hp.UNSEEN
    hp.mollview(wmap, title=title, cmap='turbo', min=1)
    hp.graticule()
    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[seed_merge] weight map guardado en {outpath}")
    occ = n_per_pix[n_per_pix > 0]
    print(f"[seed_merge] (diag) píxeles ocupados={len(occ)}, "
          f"mediana centros/píxel={np.median(occ):.1f} "
          f"(comparar con n_seeds para intuir el recentrado).")
    return n_per_pix


def sanity_report(merged):
    """Diagnósticos rápidos para auditar la elección de eps."""
    print("\n[seed_merge] === Sanity ===")
    print(f"  N voids                 : {len(merged)}")
    print(f"  semillas/void  (min/med/max): "
          f"{merged['n_seed_members'].min()}/"
          f"{int(np.median(merged['n_seed_members']))}/"
          f"{merged['n_seed_members'].max()}")
    susp = merged[merged['n_points'] > 1.5 * merged['n_seed_members']]
    if len(susp):
        print(f"  (!) {len(susp)} voids con n_points >> n_semillas: posible "
              f"fusión de voids distintos (bajá eps).")
    print("  Si la mediana semillas/void << n_seeds, eps es muy chico (fragmenta).")
