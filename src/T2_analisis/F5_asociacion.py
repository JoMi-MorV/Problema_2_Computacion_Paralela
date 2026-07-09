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
from utiles.logger import log
from utiles.semilla import get_seed
from utiles.eda_memoria import obtener_muestra_pandas

# F5_asociacion.py
# Pruebas de asociación entre variables categóricas: Chi-cuadrado y Cramer's V


def _crear_directorio(ruta):
    os.makedirs(ruta, exist_ok=True)


def _get_max_workers():
    return max(1, min(2, os.cpu_count() or 1))


def _sanitizar_nombre(nombre):
    return re.sub(r'[^A-Za-z0-9_]+', '_', nombre)


def _obtener_muestra_pandas(serie, max_n=10000, seed=None):
    """Devuelve una pandas.Series no nula con muestreo acotado."""
    seed = get_seed() if seed is None else seed
    try:
        s = obtener_muestra_pandas(serie.dropna(), max_rows=max_n, seed=seed)
    except Exception:
        return pd.Series([], dtype=object)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series([], dtype=object)
    return pd.Series(s)


def _cramers_v_from_table(table):
    try:
        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = table.values.sum()
        if n == 0:
            return np.nan, np.nan
        phi2 = chi2 / n
        r, k = table.shape
        denom = min(k - 1, r - 1)
        if denom <= 0:
            return chi2, np.nan
        cramers_v = np.sqrt(phi2 / denom)
        return chi2, cramers_v
    except Exception:
        return np.nan, np.nan


def _obtener_bloques_pares(sa, sb, chunk_size=5000):
    if len(sa) == 0 or len(sb) == 0:
        return []
    n = min(len(sa), len(sb))
    df_par = pd.DataFrame({'var1': sa.iloc[:n].values, 'var2': sb.iloc[:n].values})
    return [df_par.iloc[i:i + chunk_size] for i in range(0, len(df_par), chunk_size)]


def _guardar_grafico_contingencia(df, a, b, ruta_graficos, etiqueta):
    if a not in df.columns or b not in df.columns:
        return None
    try:
        muestra = obtener_muestra_pandas(df[[a, b]], max_rows=20000)
        muestra = muestra.dropna()
    except Exception:
        return None

    if muestra.empty or len(muestra) < 2:
        return None

    try:
        tabla = pd.crosstab(muestra[a], muestra[b])
        if tabla.empty or tabla.shape[0] < 2 or tabla.shape[1] < 2:
            return None
    except Exception:
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tabla.values, cmap='Blues')
        ax.set_xticks(range(tabla.shape[1]))
        ax.set_xticklabels(tabla.columns, rotation=45, ha='right')
        ax.set_yticks(range(tabla.shape[0]))
        ax.set_yticklabels(tabla.index)
        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                ax.text(j, i, str(tabla.iloc[i, j]), ha='center', va='center', color='black', fontsize=8)
        ax.set_title(f'Frecuencias observadas: {a} vs {b}')
        ax.set_xlabel(b)
        ax.set_ylabel(a)
        fig.colorbar(im, ax=ax, label='Conteo')
        plt.tight_layout()
        nombre_grafico = f"{etiqueta}_{_sanitizar_nombre(a)}_vs_{_sanitizar_nombre(b)}_contingencia.png"
        ruta = os.path.join(ruta_graficos, nombre_grafico)
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        log(f"Gráfico de contingencia guardado: {ruta}")
        return ruta
    except Exception as e:
        log(f"No se pudo generar gráfico de contingencia para {a} vs {b}: {e}")
        return None


def _evaluar_asociacion_par(par, df):
    a, b = par
    if a not in df.columns or b not in df.columns:
        log(f"Omitido par {a} vs {b}: columna/s no existe/n")
        return None, None

    sa = _obtener_muestra_pandas(df[a])
    sb = _obtener_muestra_pandas(df[b])
    if sa.empty or sb.empty:
        log(f"Omitido par {a} vs {b}: datos vacíos")
        return None, None

    bloques = _obtener_bloques_pares(sa, sb, chunk_size=5000)
    if not bloques:
        return None, None

    chi2_values = []
    p_values = []
    dofs = []
    cramers_values = []

    for bloque in bloques:
        try:
            table = pd.crosstab(bloque['var1'], bloque['var2'])
            chi2, p, dof, expected = stats.chi2_contingency(table)
            chi2_values.append(chi2)
            p_values.append(p)
            dofs.append(dof)
            _, cramers_v = _cramers_v_from_table(table)
            cramers_values.append(cramers_v)
        except Exception as e:
            log(f"Chi2 falló para {a} vs {b} en bloque: {e}")

    if not chi2_values:
        chi2_record = {
            'var1': a,
            'var2': b,
            'chi2': np.nan,
            'pvalue': np.nan,
            'dof': np.nan,
            'n': int(min(len(sa), len(sb))),
            'n_bloques': len(bloques)
        }
        cramers_record = {
            'var1': a,
            'var2': b,
            'cramers_v': np.nan,
            'n_bloques': len(bloques)
        }
        return chi2_record, cramers_record

    chi2_record = {
        'var1': a,
        'var2': b,
        'chi2': float(np.nanmean(chi2_values)),
        'pvalue': float(np.nanmean(p_values)),
        'dof': float(np.nanmean(dofs)) if dofs else np.nan,
        'n': int(min(len(sa), len(sb))),
        'n_bloques': len(bloques)
    }

    cramers_record = {
        'var1': a,
        'var2': b,
        'cramers_v': float(np.nanmean(cramers_values)) if cramers_values else np.nan,
        'n_bloques': len(bloques)
    }

    return chi2_record, cramers_record


def asociacion(df, etiqueta, output_base="output/analysis/asociacion"):
    """Calcula asociación entre variables categóricas y guarda CSVs y gráficos de contingencia.

    Guarda:
    - output/analysis/asociacion/<etiqueta>/chi2.csv
    - output/analysis/asociacion/<etiqueta>/cramers_v.csv
    """
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)
    ruta_graficos = os.path.join('output', 'graficos', 'asociacion', etiqueta)
    _crear_directorio(ruta_graficos)

    pares = [
        ('CANAL', 'LOCAL'),
        ('CANAL', 'GENERO'),
        ('SEGMENTO_MONTO', 'GENERO'),
        ('ES_FIN_DE_SEMANA', 'CANAL')
    ]

    chi2_records = []
    cramers_records = []
    graficos = []

    max_workers = _get_max_workers()
    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for par in pares:
            future = executor.submit(_evaluar_asociacion_par, par, df)
            futures.append(future)
        for future in futures:
            resultados.append(future.result())

    for par, (chi2_record, cramers_record) in zip(pares, resultados):
        if chi2_record is not None:
            chi2_records.append(chi2_record)
        if cramers_record is not None:
            cramers_records.append(cramers_record)
        ruta_grafico = _guardar_grafico_contingencia(df, par[0], par[1], ruta_graficos, etiqueta)
        if ruta_grafico:
            graficos.append(ruta_grafico)

    df_chi2 = pd.DataFrame.from_records(chi2_records)
    df_cram = pd.DataFrame.from_records(cramers_records)

    ruta_chi = os.path.join(ruta_salida, 'chi2.csv')
    ruta_cram = os.path.join(ruta_salida, 'cramers_v.csv')
    try:
        df_chi2.to_csv(ruta_chi, index=False)
        log(f"CSV chi2 guardado: {ruta_chi}")
    except Exception as e:
        log(f"No se pudo guardar chi2.csv: {e}")

    try:
        df_cram.to_csv(ruta_cram, index=False)
        log(f"CSV cramers_v guardado: {ruta_cram}")
    except Exception as e:
        log(f"No se pudo guardar cramers_v.csv: {e}")

    return {
        'chi2_csv': ruta_chi if not df_chi2.empty else None,
        'cramers_csv': ruta_cram if not df_cram.empty else None,
        'graficos': graficos,
        'chi2_details': df_chi2,
        'cramers_details': df_cram
    }
