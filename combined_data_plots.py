"""
combined_data_plots.py
======================
Figuras comparativas que juntan, para una MISMA selección de voids
(zmin, zmax, release, n_bins, rmin, rmax, max_Rvoid y la resolución
Rv/pix derivada de npix_stamp), todo lo que haya en Results/:

  Results/lensing/{CAT}_{release}_..._{zmin}_{zmax}_{rmin}_{rmax}_maxRv..._{reso}Rvperpix_{delta}_{filter}/
  Results/tSZ/{CAT}_{release}_z{zmin}-{zmax}_r{rmin}-{rmax}_..._sz.../

Se genera UNA figura por CASO (all, delta23>0, delta23<0). Cada figura tiene
dos columnas ('Wen&Han' | 'BOSS') y tres filas, cada una con su cajita de S/N
pegada debajo:

  Fila 1  (κ [1e-3], lensing): net signal (profile - null_rand_mean) con error
          JK+randoms. Debajo, S/N = net/err (eje y en [-6,6], ticks ±5, línea
          punteada a 3σ, grid).
  Fila 2  (compton-y orientado [1e-7]): perfil 'Hacia el void' (C) y 'Hacia la
          estructura' (D) + isótropo (full), con errores. Debajo, S/N por región
          = señal / sqrt(err_JK² + <std_null²>), dos curvas (eje y en [-4,4]).
  Fila 3  (compton-y triaxialidad [1e-7]): perfil 'Perpendicular al void' (perp,
          cuadrados) y 'Tangencial al void' (along, círculos), con errores.
          Debajo, S/N por región igual que la fila 2.

Los nulls guardados son sólo de las diferencias (diff_void=C-D, diff_fil=along-perp);
el S/N por región usa <std_null²> = promedio sobre modos (shuffled/random/rotated)
del std del null de la diff, como piso de ruido común a las dos curvas de la fila.
Los colores de las curvas tSZ respetan oriented_stacking.py y se reusan en el S/N.

Requiere el mismo numpy que escribió los .pkl/.npz (numpy 2.x -> env cmb-fit):
    /home/ftoscano/miniconda3/envs/cmb-fit/bin/python combined_data_plots.py
"""

import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# regex de las carpetas
# ----------------------------------------------------------------------
LENS_RE = re.compile(
    r'^(?P<cat>WH|BOSS)_(?P<release>[^_]+)_(?P<binning>redshift|radius)_'
    r'(?P<nbins>\d+)bins_(?P<exec>no_errors|errors)_'
    r'(?P<zmin>[\d.]+)_(?P<zmax>[\d.]+)_(?P<rmin>[\d.]+)_(?P<rmax>[\d.]+)_'
    r'maxRv(?P<maxrv>[\d.]+)_(?P<reso>[\d.eE+-]+)Rvperpix_'
    r'(?P<delta>d23_all|d23_gt[\d.]+|d23_lt[\d.]+)_(?P<filt>.+)$')

TSZ_RE = re.compile(
    r'^(?P<cat>WH|BOSS)_(?P<release>[^_]+)_z(?P<zmin>[\d.]+)-(?P<zmax>[\d.]+)_'
    r'r(?P<rmin>[\d.]+)-(?P<rmax>[\d.]+)_lam(?P<lam>[\d.]+)_(?P<ztype>[^_]+)_'
    r'los(?P<los>[\w.+-]+)_d23(?P<d23>[\w.+-]+)_sz(?P<sz>[\d.]+)$')

CATS = ('WH', 'BOSS')
COLTITLE = {'WH': 'Wen&Han', 'BOSS': 'BOSS'}

# colores de región (consistentes con oriented_stacking.py) y reusados en el S/N
C_COLOR = 'xkcd:teal'        # cara al void  -> 'Hacia el void'
D_COLOR = 'xkcd:orange'      # opuesta       -> 'Hacia la estructura'
ALONG_COLOR = 'xkcd:purple'      # tangencial (N+S)  -> círculos
PERP_COLOR = 'xkcd:goldenrod'    # perpendicular (W+E) -> cuadrados

# orden y etiqueta de los slugs de archivo por caso
CASE_ORDER = ['all', 'high', 'low']
CASE_SLUG = {'all': 'all', 'high': 'd23gt', 'low': 'd23lt'}


def _close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# estilo por "caso" (consistente entre lensing y tSZ)
# ----------------------------------------------------------------------
def _case_style(label):
    """label -> (kind, color, linestyle, display). 'low' y 'high' comparten kind
    entre lensing (d23_lt/gt) y tSZ (open/closed) para que se agrupen igual."""
    l = label.lower()
    if l in ('all', 'd23_all'):
        return ('all', 'k', '-', 'all')
    if l.startswith('lam'):                       # riqueza: número tras 'lam'
        num = re.search(r'[\d.]+', l[3:])
        num = num.group() if num else ''
        return ('rich_' + num, None, '-', f'λ ≥ {num}')  # color asignado luego
    m = re.search(r'(?:gt|lt|ge)(-?\d*\.?\d+)', l)  # umbral tras gt/lt/ge (no el '23')
    thr = m.group(1) if m else ''
    if 'lt' in l or 'open' in l:
        return ('low', 'xkcd:azure', ':', f'δ23 < {thr}')
    if 'gt' in l or 'ge' in l or 'closed' in l:
        return ('high', 'xkcd:red', ':', f'δ23 > {thr}')
    return ('other', 'xkcd:grey', '-', label)


_RICH_COLORS = ['xkcd:purple', 'xkcd:teal', 'xkcd:orange', 'xkcd:brown', 'xkcd:pink']


def _assign_richness_colors(cases):
    """Da colores estables a los casos de riqueza (λ≥x) ordenados por umbral."""
    rich = sorted({c['label'] for c in cases if c['kind'].startswith('rich_')},
                  key=lambda s: float(re.search(r'[\d.]+', s).group()))
    cmap = {lab: _RICH_COLORS[i % len(_RICH_COLORS)] for i, lab in enumerate(rich)}
    for c in cases:
        if c['color'] is None:
            c['color'] = cmap.get(c['label'], 'xkcd:grey')


# ----------------------------------------------------------------------
# descubrimiento de corridas
# ----------------------------------------------------------------------
def discover_lensing(base, sel):
    """Devuelve {cat: [caso, ...]} con net signal y significancia por caso."""
    out = {c: [] for c in CATS}
    for d in sorted(glob.glob(os.path.join(base, 'lensing', '*'))):
        m = LENS_RE.match(os.path.basename(d))
        if not m:
            continue
        if m['release'] != sel['release'] or int(m['nbins']) != int(sel['n_bins']):
            continue
        if not all(_close(m[k], sel[k]) for k in
                   [('zmin'), ('zmax'), ('rmin'), ('rmax')]):
            continue
        if not (_close(m['maxrv'], sel['max_Rvoid']) and _close(m['reso'], sel['reso'])):
            continue
        pkls = glob.glob(os.path.join(d, 'Data_FullRun_*.pkl'))
        if not pkls:
            continue
        try:
            data = pickle.load(open(pkls[0], 'rb'))
        except Exception as e:
            print(f"[warn] no pude leer {pkls[0]}: {e}")
            continue
        bins = data.get('bins_data', [])
        if not bins:
            continue
        b = bins[0]
        r = np.asarray(b['r_frac']); prof = np.asarray(b['profile'])
        err = np.asarray(b['error'])
        nrm = b.get('null_rand_mean'); nrs = b.get('null_rand_std')
        nrand = b.get('n_randoms_done', 0) or 0
        if nrm is not None and np.any(np.isfinite(nrm)):
            net = prof - np.asarray(nrm)
            nerr = (np.sqrt(err ** 2 + (np.asarray(nrs) / np.sqrt(nrand)) ** 2)
                    if nrand > 0 else err)
        else:
            net, nerr = prof, err
        kind, color, ls, disp = _case_style(m['delta'])
        with np.errstate(divide='ignore', invalid='ignore'):
            sig = net / nerr
        out[m['cat']].append(dict(label=m['delta'], kind=kind, color=color, ls=ls,
                                  disp=disp, r=r, y=net, yerr=nerr, sig=sig))
    for c in CATS:
        _assign_richness_colors(out[c])
    return out


def discover_tsz(base, sel):
    """Devuelve {cat: [caso, ...]} con perfiles de región y nulls de cada .npz."""
    out = {c: [] for c in CATS}
    for d in sorted(glob.glob(os.path.join(base, 'tSZ', '*'))):
        m = TSZ_RE.match(os.path.basename(d))
        if not m:
            continue
        if m['release'] != sel['release']:
            continue
        if not all(_close(m[k], sel[k]) for k in ['zmin', 'zmax', 'rmin', 'rmax']):
            continue
        if sel.get('lam') is not None and not _close(m['lam'], sel['lam']):
            continue
        for npz in sorted(glob.glob(os.path.join(d, 'tSZ_oriented_*.npz'))):
            sample = os.path.basename(npz)[len('tSZ_oriented_'):-len('.npz')]
            try:
                z = np.load(npz)
            except Exception as e:
                print(f"[warn] no pude leer {npz}: {e}")
                continue
            kind, color, ls, disp = _case_style(sample)
            modes = [md for md in ('shuffled', 'random', 'rotated')
                     if f'null_{md}_void_diff_mean' in z.files]
            rec = dict(label=sample, kind=kind, color=color, ls=ls, disp=disp,
                       r=z['signal_r'],
                       # perfiles de región (fila 2: C/D/full ; fila 3: along/perp)
                       C=z['signal_C'], err_C=z['signal_err_C'],
                       D=z['signal_D'], err_D=z['signal_err_D'],
                       full=z['signal_full'], err_full=z['signal_err_full'],
                       along=z['signal_along'], err_along=z['signal_err_along'],
                       perp=z['signal_perp'], err_perp=z['signal_err_perp'],
                       nulls={})
            for md in modes:
                rec['nulls'][md] = dict(
                    void_std=z[f'null_{md}_void_diff_std'],
                    fil_std=z[f'null_{md}_fil_diff_std'])
            out[m['cat']].append(rec)
    for c in CATS:
        _assign_richness_colors(out[c])
    return out


# ----------------------------------------------------------------------
# helpers de ploteo
# ----------------------------------------------------------------------
def _share_y(axes):
    ys = [ax.get_ylim() for ax in axes if ax.has_data()]
    if ys:
        lo, hi = min(y[0] for y in ys), max(y[1] for y in ys)
        for ax in axes:
            ax.set_ylim(lo, hi)


def _style_sn(ax, ylim, tick):
    """Cajita de S/N: eje y en `ylim`, ticks ±`tick`, línea punteada a 3σ, grid."""
    ax.set_ylim(*ylim)
    ax.set_yticks([-tick, tick])
    ax.axhline(3, color='gray', ls=':', lw=0.9)
    ax.axhline(-3, color='gray', ls=':', lw=0.9)
    ax.grid(alpha=0.3)


def _combined_null_var(case, which):
    """<std_null²> = promedio sobre modos del std del null de la diff (por bin).
    which='void' (para C/D) o 'fil' (para along/perp). 0 si no hay nulls."""
    key = 'void_std' if which == 'void' else 'fil_std'
    stds = [nd[key] for nd in case['nulls'].values() if key in nd]
    if not stds:
        return 0.0
    return np.mean(np.array(stds) ** 2, axis=0)


def _plot_row_lensing(ax_p, ax_s, case, sn_ylim, sn_tick, scale=1e3):
    c = case
    ax_p.errorbar(c['r'], c['y'] * scale, yerr=c['yerr'] * scale, fmt='o',
                  color=c['color'], ls=c['ls'], capsize=2, ms=3, lw=1.4,
                  label=c['disp'])
    ax_p.axhline(0, color='k', ls=':', alpha=0.5)
    ax_p.axvline(1.0, color='gray', ls='--', alpha=0.5)
    ax_p.grid(alpha=0.2)
    ax_p.legend(fontsize=8, loc='best')

    ax_s.plot(c['r'], c['sig'], color=c['color'], lw=1.6)
    ax_s.axvline(1.0, color='gray', ls='--', alpha=0.5)
    _style_sn(ax_s, sn_ylim, sn_tick)


def _plot_row_void(ax_p, ax_s, case, sn_ylim, sn_tick, scale=1e7):
    c = case
    ax_p.errorbar(c['r'], c['C'] * scale, yerr=c['err_C'] * scale, fmt='o-',
                  color=C_COLOR, capsize=2, ms=3, lw=1.4, label='Hacia el void')
    ax_p.errorbar(c['r'], c['D'] * scale, yerr=c['err_D'] * scale, fmt='s-',
                  color=D_COLOR, capsize=2, ms=3, lw=1.4, label='Hacia la estructura')
    ax_p.plot(c['r'], c['full'] * scale, color='gray', ls=':', lw=1.3, label='isótropo')
    ax_p.axhline(0, color='k', ls=':', alpha=0.4)
    ax_p.grid(alpha=0.2)
    ax_p.legend(fontsize=8, loc='best')

    var = _combined_null_var(c, 'void')
    with np.errstate(divide='ignore', invalid='ignore'):
        sn_C = c['C'] / np.sqrt(c['err_C'] ** 2 + var)
        sn_D = c['D'] / np.sqrt(c['err_D'] ** 2 + var)
    ax_s.plot(c['r'], sn_C, color=C_COLOR, lw=1.6)
    ax_s.plot(c['r'], sn_D, color=D_COLOR, lw=1.6)
    _style_sn(ax_s, sn_ylim, sn_tick)


def _plot_row_fil(ax_p, ax_s, case, sn_ylim, sn_tick, scale=1e7):
    c = case
    ax_p.errorbar(c['r'], c['perp'] * scale, yerr=c['err_perp'] * scale, fmt='s-',
                  color=PERP_COLOR, capsize=2, ms=3, lw=1.4, label='Perpendicular al void')
    ax_p.errorbar(c['r'], c['along'] * scale, yerr=c['err_along'] * scale, fmt='o-',
                  color=ALONG_COLOR, capsize=2, ms=3, lw=1.4, label='Tangencial al void')
    ax_p.plot(c['r'], c['full'] * scale, color='gray', ls=':', lw=1.3, label='isótropo')
    ax_p.axhline(0, color='k', ls=':', alpha=0.4)
    ax_p.grid(alpha=0.2)
    ax_p.legend(fontsize=8, loc='best')

    var = _combined_null_var(c, 'fil')
    with np.errstate(divide='ignore', invalid='ignore'):
        sn_perp = c['perp'] / np.sqrt(c['err_perp'] ** 2 + var)
        sn_along = c['along'] / np.sqrt(c['err_along'] ** 2 + var)
    ax_s.plot(c['r'], sn_perp, color=PERP_COLOR, lw=1.6)
    ax_s.plot(c['r'], sn_along, color=ALONG_COLOR, lw=1.6)
    _style_sn(ax_s, sn_ylim, sn_tick)


# filas: (kind, plotter, ylabel, xlabel, sn_ylim, sn_tick)
ROWS = [
    ('lensing', _plot_row_lensing, r'$\kappa$  [$10^{-3}$]', r'$r/R_v$', (-6, 6), 5),
    ('void', _plot_row_void, r'compton-$y$ orientado  [$10^{-7}$]', r'$r$ [Mpc/h]', (-4, 4), 3),
    ('fil', _plot_row_fil, r'compton-$y$ triaxialidad  [$10^{-7}$]', r'$r$ [Mpc/h]', (-4, 4), 3),
]


def _find_case(cases, kind):
    return next((c for c in cases if c['kind'] == kind), None)


def _make_case_figure(case_kind, lens, tsz, base, sel):
    """Una figura (2 columnas WH|BOSS x 3 filas) para un único caso."""
    # display del caso (misma disp en lensing y tSZ)
    disp = case_kind
    for cat in CATS:
        c = _find_case(lens[cat], case_kind) or _find_case(tsz[cat], case_kind)
        if c:
            disp = c['disp']
            break

    fig = plt.figure(figsize=(9.5, 12.5))
    outer = fig.add_gridspec(3, 1, hspace=0.30)

    for ri, (rkind, plotter, ylab, xlab, sn_ylim, sn_tick) in enumerate(ROWS):
        inner = outer[ri].subgridspec(2, 2, height_ratios=[3, 1],
                                      hspace=0.0, wspace=0.06)
        ax_p = [fig.add_subplot(inner[0, j]) for j in range(2)]
        ax_s = [fig.add_subplot(inner[1, j], sharex=ax_p[j]) for j in range(2)]

        for j, cat in enumerate(CATS):
            src = lens[cat] if rkind == 'lensing' else tsz[cat]
            case = _find_case(src, case_kind)
            if case is None:
                ax_p[j].text(0.5, 0.5, f'sin datos {COLTITLE[cat]}',
                             ha='center', va='center', transform=ax_p[j].transAxes,
                             color='gray')
                _style_sn(ax_s[j], sn_ylim, sn_tick)
            else:
                plotter(ax_p[j], ax_s[j], case, sn_ylim, sn_tick)

            if ri == 0:
                ax_p[j].set_title(COLTITLE[cat], fontsize=13)
            ax_s[j].set_xlabel(xlab)
            plt.setp(ax_p[j].get_xticklabels(), visible=False)
            if j == 1:
                plt.setp(ax_p[j].get_yticklabels(), visible=False)
                plt.setp(ax_s[j].get_yticklabels(), visible=False)

        ax_p[0].set_ylabel(ylab)
        ax_s[0].set_ylabel('S/N')
        _share_y(ax_p)

    tag = (f"{sel['release']}_z{sel['zmin']}-{sel['zmax']}_r{sel['rmin']}-{sel['rmax']}"
           f"_maxRv{sel['max_Rvoid']:g}_{sel['n_bins']}bins")
    if sel.get('lam') is not None:
        tag += f"_lam{sel['lam']:g}"
    fig.suptitle(f"{disp}  |  {tag}", y=0.995, fontsize=12)
    slug = CASE_SLUG.get(case_kind, case_kind)
    out_dir = os.path.join(base, 'tSZ')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Combined_{tag}_{slug}.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[combined] figura '{disp}' guardada en {out_path}")


# ----------------------------------------------------------------------
# entrada
# ----------------------------------------------------------------------
def make_combined_plot(config):
    base = config['results_folder']
    sel = dict(release=config['release'], n_bins=config['n_bins'],
               zmin=config['zmin'], zmax=config['zmax'],
               rmin=config['rmin'], rmax=config['rmax'],
               max_Rvoid=config['max_Rvoid'],
               reso=(2 * config['max_Rvoid']) / config['npix_stamp'],
               lam=config.get('lambda_min'))
    print(f"[combined] selección: {sel}")

    lens = discover_lensing(base, sel)
    tsz = discover_tsz(base, sel)
    print(f"[combined] lensing: WH={len(lens['WH'])} BOSS={len(lens['BOSS'])} | "
          f"tSZ: WH={len(tsz['WH'])} BOSS={len(tsz['BOSS'])}")

    present = set()
    for cat in CATS:
        for c in lens[cat] + tsz[cat]:
            present.add(c['kind'])
    if not present:
        print("[combined] no encontré ninguna corrida con esa selección. Nada que plotear.")
        return

    kinds = [k for k in CASE_ORDER if k in present]
    kinds += [k for k in sorted(present) if k not in CASE_ORDER]
    print(f"[combined] casos a plotear: {kinds}")

    for case_kind in kinds:
        _make_case_figure(case_kind, lens, tsz, base, sel)


if __name__ == "__main__":
    config = {
        'results_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/',
        'release': 'PR4',
        'zmin': 0.1, 'zmax': 0.5,
        'rmin': 35.0, 'rmax': 70.0,
        'max_Rvoid': 2.5,
        'npix_stamp': 400,      # -> reso = 2*max_Rvoid/npix_stamp
        'n_bins': 1,
        'lambda_min': 35,       # corte de riqueza que nombra la carpeta tSZ (_lam..); None -> no filtra

        'out_path': None,       # None -> Results/Combined_{...}_{caso}.pdf
    }
    make_combined_plot(config)
