import os
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pandas.api import types as ptypes
from utiles.logger import log
from utiles.semilla import get_seed
from utiles.eda_memoria import obtener_muestra_pandas

# F3_normalidad.py
# Pruebas de normalidad por variable numérica: Shapiro-Wilk y Kolmogorov-Smirnov


def _crear_directorio(ruta):
    os.makedirs(ruta, exist_ok=True)


def _get_max_workers():
    return max(1, min(2, os.cpu_count() or 1))


def _sanitizar_nombre(nombre):
    return re.sub(r'[^A-Za-z0-9_]+', '_', nombre)


def _es_columna_temporal(df, col):
    if col not in df.columns:
        return False
    dtype = df[col].dtype
    return (
        ptypes.is_datetime64_any_dtype(dtype)
        or ptypes.is_timedelta64_dtype(dtype)
        or ptypes.is_period_dtype(dtype)
    )


def _columnas_numericas(df):
    return [
        col for col in df.columns
        if col not in {'SKU', 'LOCAL', 'BOLETA', 'GENERO'}
        and not _es_columna_temporal(df, col)
        and getattr(df[col].dtype, 'kind', '') in 'iufc'
    ]


def _obtener_muestra_pandas(serie, max_n=5000, seed=None):
    """Convierte una serie Dask/Pandas a pandas Series usando una muestra acotada."""
    seed = get_seed() if seed is None else seed
    try:
        s = obtener_muestra_pandas(serie.dropna(), max_rows=max_n, seed=seed)
    except Exception:
        return pd.Series([], dtype=float)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series([], dtype=float)
    return pd.Series(s, dtype=float)


def _obtener_bloques_serie(serie, chunk_size=5000):
    try:
        s = obtener_muestra_pandas(serie.dropna(), max_rows=chunk_size * 5)
    except Exception:
        return []
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            return []
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if len(s) == 0:
        return []
    return [s.iloc[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


def _evaluar_normalidad_columna(col, df, ruta_graficos, etiqueta):
    log(f"Evaluando normalidad para {col}")
    serie = df[col]
    bloques = _obtener_bloques_serie(serie, chunk_size=5000)
    if not bloques:
        log(f"Omitido {col}: no hay datos válidos")
        return None

    shapiro_stats = []
    shapiro_pvalues = []
    ks_stats = []
    ks_pvalues = []

    for bloque in bloques:
        if len(bloque) < 3:
            continue
        try:
            sh_stat, sh_p = stats.shapiro(bloque.values)
            shapiro_stats.append(sh_stat)
            shapiro_pvalues.append(sh_p)
        except Exception as e:
            log(f"Shapiro falló para {col} en bloque: {e}")

        try:
            m = float(bloque.mean())
            sd = float(bloque.std(ddof=0)) if float(bloque.std(ddof=0)) > 0 else 0.0
            if sd == 0.0:
                ks_stats.append(np.nan)
                ks_pvalues.append(np.nan)
            else:
                ks_stat, ks_p = stats.kstest(bloque.values, 'norm', args=(m, sd))
                ks_stats.append(ks_stat)
                ks_pvalues.append(ks_p)
        except Exception as e:
            log(f"KS falló para {col} en bloque: {e}")

    if not shapiro_pvalues and not ks_pvalues:
        return None

    s = pd.concat(bloques, axis=0)
    if len(s) > 5000:
        s = s.sample(n=5000, random_state=get_seed())

    try:
        plt.figure(figsize=(6, 6))
        stats.probplot(s.values, dist='norm', plot=plt)
        plt.title(f'QQ Plot de {col}')
        plt.tight_layout()
        sanitized = _sanitizar_nombre(col)
        nombre_qq = f"{etiqueta}_{sanitized}_qq.png"
        ruta_qq = os.path.join(ruta_graficos, nombre_qq)
        plt.savefig(ruta_qq, dpi=150)
        plt.close()
        log(f"QQ plot guardado: {ruta_qq}")
    except Exception as e:
        log(f"No se pudo generar QQ plot para {col}: {e}")

    return {
        'variable': col,
        'n': int(sum(len(b) for b in bloques)),
        'n_bloques': len(bloques),
        'shapiro_stat': float(np.nanmean(shapiro_stats)) if shapiro_stats else np.nan,
        'shapiro_pvalue': float(np.nanmean(shapiro_pvalues)) if shapiro_pvalues else np.nan,
        'ks_stat': float(np.nanmean(ks_stats)) if ks_stats else np.nan,
        'ks_pvalue': float(np.nanmean(ks_pvalues)) if ks_pvalues else np.nan
    }


def normalidad(df, etiqueta, output_base="output/analysis/normalidad"):
    """Ejecuta pruebas de normalidad y genera QQ plots.

    Salidas:
    - CSV: output/analysis/normalidad/<etiqueta>/normalidad.csv
    - TXT: output/analysis/normalidad/<etiqueta>/normalidad.txt
    - QQ plots: output/graficos/normalidad/<etiqueta>/*.png
    """
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)

    ruta_graficos = os.path.join('output', 'graficos', 'normalidad', etiqueta)
    _crear_directorio(ruta_graficos)

    columnas = _columnas_numericas(df)
    resultados = []

    if not columnas:
        log(f"No se encontraron columnas numéricas para normalidad: {etiqueta}")
        return {
            'normalidad_csv': None,
            'normalidad_txt': None,
            'graficos_folder': ruta_graficos,
            'columnas': columnas
        }

    max_workers = _get_max_workers()
    resultados_paralelos = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for col in columnas:
            future = executor.submit(_evaluar_normalidad_columna, col, df, ruta_graficos, etiqueta)
            futures.append(future)
        for future in futures:
            resultados_paralelos.append(future.result())

    resultados = [res for res in resultados_paralelos if res is not None]

    df_res = pd.DataFrame(resultados).set_index('variable') if resultados else pd.DataFrame()
    ruta_csv = os.path.join(ruta_salida, 'normalidad.csv')
    try:
        df_res.to_csv(ruta_csv)
        log(f"CSV de normalidad guardado: {ruta_csv}")
    except Exception as e:
        log(f"No se pudo guardar CSV de normalidad: {e}")

    ruta_txt = os.path.join(ruta_salida, 'normalidad.txt')
    try:
        with open(ruta_txt, 'w', encoding='utf-8') as fh:
            fh.write('Normalidad por variable\n')
            fh.write('======================\n')
            for idx, row in (df_res.reset_index().iterrows() if not df_res.empty else []):
                var = row['variable'] if 'variable' in row else row.name
                sh_p = row.get('shapiro_pvalue', np.nan)
                ks_p = row.get('ks_pvalue', np.nan)
                conclusion = 'Indeterminado'
                try:
                    if (not np.isnan(sh_p) and sh_p > 0.05) and (not np.isnan(ks_p) and ks_p > 0.05):
                        conclusion = 'No se rechaza normalidad (p>0.05)'
                    else:
                        conclusion = 'Rechazo de normalidad (p<=0.05)'
                except Exception:
                    conclusion = 'Indeterminado'
                fh.write(f"{var}: shapiro_p={sh_p}, ks_p={ks_p} -> {conclusion}\n")
        log(f"TXT de normalidad guardado: {ruta_txt}")
    except Exception as e:
        log(f"No se pudo guardar TXT de normalidad: {e}")

    return {
        'normalidad_csv': ruta_csv if not df_res.empty else None,
        'normalidad_txt': ruta_txt,
        'graficos_folder': ruta_graficos,
        'detalles': df_res
    }
