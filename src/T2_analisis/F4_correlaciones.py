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

# F4_correlaciones.py
# Análisis de correlaciones: elige Pearson si ambas variables son normales,
# en caso contrario usa Spearman. Guarda CSV con coeficientes y p-values y
# genera heatmap de correlaciones.


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


def _obtener_bloques_pares(sa, sb, chunk_size=5000):
    if len(sa) == 0 or len(sb) == 0:
        return []
    n = min(len(sa), len(sb))
    df_par = pd.DataFrame({'var1': sa.iloc[:n].values, 'var2': sb.iloc[:n].values})
    return [df_par.iloc[i:i + chunk_size] for i in range(0, len(df_par), chunk_size)]


def _evaluar_correlacion_par(par, df, normales):
    a, b = par
    sa = _obtener_muestra_pandas(df[a])
    sb = _obtener_muestra_pandas(df[b])
    try:
        bloques = _obtener_bloques_pares(sa, sb, chunk_size=5000)
        if not bloques:
            return {
                'var1': a,
                'var2': b,
                'method': 'none',
                'corr': np.nan,
                'pvalue': np.nan,
                'n': 0,
                'n_bloques': 0
            }

        corr_values = []
        p_values = []
        methods = []
        n_total = 0
        for bloque in bloques:
            vals_a = bloque['var1'].values
            vals_b = bloque['var2'].values
            n = min(len(vals_a), len(vals_b))
            n_total += int(n)
            if n < 3:
                continue
            if np.unique(vals_a).size < 2 or np.unique(vals_b).size < 2:
                continue
            if normales.get(a, False) and normales.get(b, False):
                method = 'pearson'
                try:
                    corr, pval = stats.pearsonr(vals_a, vals_b)
                except Exception:
                    corr, pval = np.nan, np.nan
            else:
                method = 'spearman'
                try:
                    corr, pval = stats.spearmanr(vals_a, vals_b)
                except Exception:
                    corr, pval = np.nan, np.nan
            corr_values.append(corr)
            p_values.append(pval)
            methods.append(method)

        if corr_values:
            corr = float(np.nanmean(corr_values))
            pval = float(np.nanmean(p_values)) if p_values else np.nan
            method = methods[0] if methods else 'none'
        else:
            corr = np.nan
            pval = np.nan
            method = 'none'

        return {
            'var1': a,
            'var2': b,
            'method': method,
            'corr': corr,
            'pvalue': pval,
            'n': int(n_total),
            'n_bloques': len(bloques)
        }
    except Exception:
        return {
            'var1': a,
            'var2': b,
            'method': 'error',
            'corr': np.nan,
            'pvalue': np.nan,
            'n': 0,
            'n_bloques': 0
        }


def correlaciones(df, etiqueta, output_base="output/analysis/correlaciones"):
    """Calcula correlaciones numéricas con una muestra acotada y guarda un heatmap."""
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)

    ruta_graficos = os.path.join('output', 'graficos', 'correlaciones', etiqueta)
    _crear_directorio(ruta_graficos)

    columnas = _columnas_numericas(df)
    if not columnas:
        log(f"No hay columnas numéricas para correlaciones: {etiqueta}")
        return None

    normales = {}
    for col in columnas:
        s = _obtener_muestra_pandas(df[col])
        if len(s) < 3:
            normales[col] = False
            continue
        try:
            stat, p = stats.shapiro(s.values)
            normales[col] = bool(p > 0.05)
        except Exception:
            normales[col] = False

    registros = []
    matriz = pd.DataFrame(index=columnas, columns=columnas, dtype=float)
    pares = [(a, b) for i, a in enumerate(columnas) for j, b in enumerate(columnas) if j >= i]

    max_workers = _get_max_workers()
    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for par in pares:
            future = executor.submit(_evaluar_correlacion_par, par, df, normales)
            futures.append(future)
        for future in futures:
            resultados.append(future.result())

    for resultado in resultados:
        registros.append(resultado)
        matriz.at[resultado['var1'], resultado['var2']] = resultado['corr']
        matriz.at[resultado['var2'], resultado['var1']] = resultado['corr']

    df_cor = pd.DataFrame.from_records(registros)
    ruta_csv = os.path.join(ruta_salida, 'correlaciones.csv')
    try:
        df_cor.to_csv(ruta_csv, index=False)
        log(f"CSV de correlaciones guardado: {ruta_csv}")
    except Exception as e:
        log(f"No se pudo guardar CSV de correlaciones: {e}")

    ruta_heat = None
    try:
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('coolwarm')
        im = plt.imshow(matriz.astype(float), cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, label='Coeficiente de correlación')
        plt.xticks(range(len(columnas)), columnas, rotation=45, ha='right')
        plt.yticks(range(len(columnas)), columnas)
        for i in range(len(columnas)):
            for j in range(len(columnas)):
                val = matriz.iat[i, j]
                if pd.notna(val):
                    plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color='black')
        plt.title('Heatmap de correlación')
        plt.tight_layout()
        ruta_heat = os.path.join(ruta_graficos, 'heatmap_correlacion.png')
        plt.savefig(ruta_heat, dpi=150)
        plt.close()
        log(f"Heatmap guardado: {ruta_heat}")
    except Exception as e:
        log(f"No se pudo generar heatmap de correlación: {e}")

    return {
        'correlaciones_csv': ruta_csv,
        'heatmap': ruta_heat if ruta_heat and os.path.exists(ruta_heat) else None,
        'detalles': df_cor
    }
