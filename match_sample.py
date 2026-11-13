"""
match_sample.py
===============
Submuestreo aleatorio de un catálogo de voids (p. ej. Websky) para que reproduzca la
distribución de un catálogo de referencia (p. ej. los datos WH) en una o más columnas
(R_void, z, delta_23, ...), con un número total de objetos fijado.

Método (el mismo que se usa para igualar n(z) en 2PCF):
  1. Celdas = cuantiles de la REFERENCIA en cada columna (producto cartesiano si son
     varias). Así cada celda tiene ~N_ref/n_cells objetos de referencia y no hay celdas
     vacías donde están los datos.
  2. Objetivo por celda: n_T(c) = N_target * n_ref(c)/N_ref (restos mayores -> suma exacta).
  3. En cada celda se eligen n_T(c) objetos de la muestra al azar, SIN reemplazo.
Dentro de cada celda la distribución sigue siendo la de la muestra: con celdas finas el
residuo es chico, y se reporta con tests KS antes/después.

Si en alguna celda la muestra tiene menos objetos que los pedidos:
  on_short='shrink'  : baja N_target al máximo que conserva la forma exacta (default)
  on_short='replace' : completa esas celdas con reemplazo (voids repetidos: evitar)
  on_short='error'   : aborta
"""
import numpy as np
from scipy.stats import ks_2samp


def quantile_edges(x, n_bins):
    x = np.asarray(x, float)
    e = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if len(e) < 2:
        e = np.array([e[0], e[0]])
    tol = 1e-9 * max(1.0, abs(e[-1]))
    e[0], e[-1] = e[0] - tol, e[-1] + tol     # soporte = rango de la referencia: lo de afuera no se sortea
    return e


def cell_index(df, cols, edges):
    """Índice de celda (producto cartesiano); -1 si cae fuera del rango de la referencia."""
    idx, mult = np.zeros(len(df), dtype=np.int64), 1
    out = np.zeros(len(df), dtype=bool)
    for c in cols:
        e = edges[c]
        k = np.searchsorted(e, df[c].values.astype(float), side='right') - 1
        bad = (k < 0) | (k > len(e) - 2)
        out |= bad
        idx += np.clip(k, 0, len(e) - 2) * mult
        mult *= len(e) - 1
    idx[out] = -1
    return idx, mult


def _alloc(n_ref_cells, total):
    """Reparte 'total' proporcional a n_ref_cells (restos mayores; suma exacta)."""
    raw = n_ref_cells * (total / n_ref_cells.sum())
    n = np.floor(raw).astype(np.int64)
    rest = int(total - n.sum())
    if rest > 0:
        n[np.argsort(-(raw - n), kind='stable')[:rest]] += 1
    return n


def _max_target(nS, nR):
    """Mayor N tal que _alloc(nR, N) <= nS en todas las celdas."""
    occ = nR > 0
    N = int(np.floor(np.min(nS[occ] / nR[occ]) * nR.sum()))
    while N > 0 and np.any(_alloc(nR, N) > nS):
        N -= 1
    return N


def ks_report(a, b, cols):
    return {c: ks_2samp(a[c].values, b[c].values) for c in cols if c in a.columns and c in b.columns}


def match_distribution(sample, ref, cols=('R_void',), n_bins=10, n_target='ref', seed=0,
                       on_short='shrink', report_cols=('R_void', 'z'), label='', verbose=True):
    """
    sample   : DataFrame a submuestrear (Websky).
    ref      : DataFrame de referencia (datos).
    cols     : columnas a igualar, p. ej. ('R_void',) o ('R_void', 'z').
    n_bins   : int o {col: int}; cuantiles de la referencia por columna.
    n_target : 'ref' (N de la referencia), 'max' (máximo N que conserva la forma) o int.
    seed     : semilla del sorteo (mismo input + misma semilla -> misma selección).
    Devuelve (submuestra, info).
    """
    cols = list(cols)
    for c in cols:
        for name, d in (('sample', sample), ('ref', ref)):
            if c not in d.columns:
                raise KeyError(f"[match] columna '{c}' no está en {name}")
            if not np.all(np.isfinite(d[c].values.astype(float))):
                raise ValueError(f"[match] '{c}' tiene NaN/inf en {name}")
    nb = n_bins if isinstance(n_bins, dict) else {c: n_bins for c in cols}
    edges = {c: quantile_edges(ref[c].values, nb[c]) for c in cols}
    cs, ncell = cell_index(sample, cols, edges)
    cr, _ = cell_index(ref, cols, edges)
    nS = np.bincount(cs[cs >= 0], minlength=ncell)
    nR = np.bincount(cr[cr >= 0], minlength=ncell)
    n_out = int((cs < 0).sum())
    N_ref, N_max = len(ref), _max_target(nS, nR)

    if n_target == 'ref':
        N_req = N_ref
    elif n_target == 'max':
        N_req = N_max
    else:
        N_req = int(n_target)
    replace = False
    if N_req > N_max:
        msg = (f"[match]{label} (!) la muestra no alcanza para N={N_req} con la forma de la referencia "
               f"(máximo exacto: N={N_max})")
        if on_short == 'shrink':
            print(msg + f" -> uso N={N_max}")
            N_req = N_max
        elif on_short == 'replace':
            print(msg + " -> completo con REEMPLAZO (voids repetidos)")
            replace = True
        else:
            raise ValueError(msg)
    if N_req <= 0:
        raise ValueError(f"[match]{label} N objetivo = {N_req}")

    nT = _alloc(nR, N_req)
    rng = np.random.default_rng(seed)
    order = np.argsort(cs, kind='stable')
    starts = np.searchsorted(cs[order], np.arange(ncell))   # cs = -1 queda al principio y no se usa
    sel = []
    for k in np.flatnonzero(nT):
        members = order[starts[k]:starts[k] + nS[k]]
        rep = replace and nT[k] > nS[k]
        sel.append(rng.choice(members, size=nT[k], replace=rep))
    sel = np.sort(np.concatenate(sel))
    out = sample.iloc[sel].copy()

    rcols = [c for c in dict.fromkeys(list(report_cols) + cols) if c in sample.columns and c in ref.columns]
    info = {'cols': cols, 'edges': edges, 'n_sample': len(sample), 'N_ref': N_ref, 'N_max': N_max,
            'N': len(out), 'seed': seed, 'replace': replace, 'n_duplicates': int(len(sel) - len(np.unique(sel))),
            'ks_before': ks_report(sample, ref, rcols), 'ks_after': ks_report(out, ref, rcols),
            'n_cells': int((nR > 0).sum()), 'n_outside_ref_range': n_out}
    if verbose:
        print(f"[match]{label} igualo {cols} ({info['n_cells']} celdas, cuantiles de la referencia): "
              f"{len(sample)} -> {len(out)} (ref N={N_ref}, máx. exacto N={N_max}, semilla {seed}; "
              f"{n_out} fuera del rango de la referencia)")
        for c in rcols:
            b, a = info['ks_before'][c], info['ks_after'][c]
            tag = '' if c in cols else '   <- NO igualada'
            print(f"[match]{label}   {c:8s}: mediana muestra {np.median(sample[c]):.3f} -> {np.median(out[c]):.3f} "
                  f"(ref {np.median(ref[c]):.3f}) | KS p: {b.pvalue:.2e} -> {a.pvalue:.3f}{tag}")
    return out, info
