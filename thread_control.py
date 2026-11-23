"""
thread_control.py
=================
Control del número de hilos/procesos de la pipeline (máquina compartida, sin SLURM).

Problema que resuelve
    ProcessPoolExecutor(max_workers=N) crea N procesos, pero CADA proceso hereda las
    librerías nativas (OpenBLAS/MKL de numpy, OpenMP de healpy y de sklearn) con su
    pool de hilos por defecto = todos los núcleos de la máquina. En el worker,
    `pv @ vecs[i]` (radial_sums_healpix) es una llamada BLAS que levanta ese pool:
    N procesos x ~n_cores hilos -> cientos/miles de hilos y sobresuscripción.

Política
    - workers : n_workers procesos x threads_per_worker hilos (por defecto 1).
    - proceso principal : como mucho n_workers hilos (map2alm, rotate_alm, KMeans, ...).
      Mientras el pool trabaja, el principal está bloqueado esperando: el total activo
      no supera n_workers * threads_per_worker.

Este módulo NO importa numpy al cargarse, para poder fijar las variables de entorno
antes de que se carguen las librerías nativas (set_env_threads).
"""
import os
import sys

# BLAS: su pool se crea AL IMPORTAR numpy con este tamaño y los workers lo heredan por fork
# (hilos ociosos que threadpoolctl ya no puede destruir). La pipeline no hace BLAS pesado en
# el proceso principal -> BLAS = 1 siempre. OpenMP (SHT de healpy, KMeans) = n en el principal.
_BLAS_VARS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
_OMP_VARS = ('OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'DUCC0_NUM_THREADS')
_ENV_VARS = _BLAS_VARS + _OMP_VARS


def available_cores():
    """Núcleos que este proceso puede usar (respeta taskset/cgroups, no os.cpu_count())."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def resolve_workers(n_workers):
    """n_workers del config -> entero en [1, núcleos disponibles]."""
    n_av = available_cores()
    if n_workers is None:
        return n_av
    n = int(n_workers)
    if n < 1:
        raise ValueError(f'n_workers debe ser >= 1 (o None), no {n_workers}')
    if n > n_av:
        print(f'[threads] (!) n_workers={n} > núcleos disponibles ({n_av}): se usa {n_av}')
        n = n_av
    return n


def set_env_threads(n):
    """
    Variables de entorno de hilos: BLAS = 1, OpenMP = n. Sólo tiene efecto sobre librerías
    que todavía NO se cargaron: llamar antes de importar numpy/healpy/sklearn.
    Respeta valores que el usuario ya haya exportado en la shell.
    """
    n = str(max(1, int(n)))
    for v in _BLAS_VARS:
        os.environ.setdefault(v, '1')
    for v in _OMP_VARS:
        os.environ.setdefault(v, n)
    if 'numpy' in sys.modules:
        print('[threads] (!) numpy ya estaba importado (¿consola Spyder/IPython?): el pool BLAS '
              'ya existe y los workers lo heredan ocioso. Corré desde la terminal o exportá '
              'OPENBLAS_NUM_THREADS=1 antes de abrir la consola.')


def limit_threads(n, verbose=False, blas=None):
    """
    Limita en caliente los pools de hilos de las librerías YA cargadas en este proceso
    (OpenMP, incluido el libgomp de healpy/sklearn: n; BLAS: `blas`, por defecto n) vía
    threadpoolctl. No destruye hilos ya creados: sólo evita que se usen.
    """
    n = max(1, int(n))
    blas = n if blas is None else max(1, int(blas))
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
    except ImportError:
        if verbose:
            print('[threads] (!) threadpoolctl no está instalado: sólo se usan variables de '
                  'entorno (pip install threadpoolctl)')
        return None
    # sin 'with': queda aplicado en este proceso
    lim = threadpool_limits(limits={'openmp': n, 'blas': blas})
    if verbose:
        libs = ', '.join(f"{d['internal_api']}={d['num_threads']}" for d in threadpool_info())
        print(f'[threads] proceso principal: <= {n} hilos ({libs or "sin pools nativos cargados"})')
    return lim


def worker_init(n_threads):
    """initializer de ProcessPoolExecutor: cada worker con n_threads hilos nativos."""
    for v in _ENV_VARS:
        os.environ[v] = str(n_threads)
    limit_threads(n_threads)


def setup_main(config, verbose=True):
    """
    Llamar al inicio de cada corrida. Devuelve (n_workers, threads_per_worker) efectivos.
    """
    n_w = resolve_workers(config.get('n_workers'))
    tpw = max(1, int(config.get('threads_per_worker', 1) or 1))
    limit_threads(n_w, verbose=verbose, blas=1)
    if verbose:
        print(f'[threads] {n_w} workers x {tpw} hilo(s) = {n_w * tpw} hilos de cómputo '
              f'(núcleos disponibles: {available_cores()})')
    return n_w, tpw
