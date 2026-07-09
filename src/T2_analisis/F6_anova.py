import os
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

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_TUKEY = True
except Exception:
    HAS_TUKEY = False


def _crear_directorio(ruta):
    os.makedirs(ruta, exist_ok=True)


def _get_max_workers():
    return max(1, min(2, os.cpu_count() or 1))


def _obtener_muestra_tabla(df, cols, max_n=10000, seed=None):
    df2 = df[cols]
    seed = get_seed() if seed is None else seed
    try:
        pdf = obtener_muestra_pandas(df2.dropna(), max_rows=max_n, seed=seed)
    except Exception:
        return pd.DataFrame([], columns=cols)
    return pdf


def _obtener_bloques_tabla(tabla, chunk_size=5000):
    if tabla.empty:
        return []
    return [tabla.iloc[i:i + chunk_size] for i in range(0, len(tabla), chunk_size)]


def _evaluar_anova_factor(factor, df):
    if 'MONTO APLICADO' not in df.columns or factor not in df.columns:
        log(f"Omitido ANOVA {factor}: columna faltante")
        return None, None

    tabla = _obtener_muestra_tabla(df, ['MONTO APLICADO', factor])
    if tabla.empty:
        log(f"Omitido ANOVA {factor}: no hay datos válidos")
        return None, None

    bloques = _obtener_bloques_tabla(tabla, chunk_size=5000)
    if not bloques:
        return None, None

    f_stats = []
    p_values = []
    n_groups_list = []

    for bloque in bloques:
        groups = [group['MONTO APLICADO'].values for name, group in bloque.groupby(factor)]
        if len(groups) < 2:
            continue
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            f_stats.append(f_stat)
            p_values.append(p_val)
            n_groups_list.append(len(groups))
        except Exception as e:
            log(f"ANOVA falló para {factor} en bloque: {e}")

    if not f_stats:
        registro = {
            'factor': factor,
            'f_stat': np.nan,
            'pvalue': np.nan,
            'n_groups': np.nan,
            'n_bloques': len(bloques)
        }
        return registro, None

    registro = {
        'factor': factor,
        'f_stat': float(np.nanmean(f_stats)),
        'pvalue': float(np.nanmean(p_values)),
        'n_groups': int(np.nanmean(n_groups_list)) if n_groups_list else np.nan,
        'n_bloques': len(bloques)
    }

    df_tukey = None
    if not np.isnan(registro['pvalue']) and registro['pvalue'] <= 0.05 and HAS_TUKEY:
        try:
            tuk = pairwise_tukeyhsd(endog=tabla['MONTO APLICADO'], groups=tabla[factor], alpha=0.05)
            data = tuk._results_table.data
            header = data[0]
            rows = data[1:]
            df_tukey = pd.DataFrame(rows, columns=header)
            df_tukey['factor'] = factor
        except Exception as e:
            log(f"Tukey falló para {factor}: {e}")
    elif not HAS_TUKEY and not np.isnan(p_val) and p_val <= 0.05:
        log(f"Tukey no disponible (statsmodels). Se omitirá para {factor}.")

    return registro, df_tukey


def anova(df, etiqueta, output_base="output/analysis/anova"):
    """Realiza ANOVA sobre 'MONTO APLICADO' por factores categóricos y opcionalmente Tukey.

    Salidas:
    - output/analysis/anova/<etiqueta>/anova.csv
    - output/analysis/anova/<etiqueta>/tukey.csv
    - output/graficos/anova/<etiqueta>/boxplot_anova.png (combinado)
    """
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)
    ruta_graficos = os.path.join('output', 'graficos', 'anova', etiqueta)
    _crear_directorio(ruta_graficos)

    factores = ['CANAL', 'LOCAL']
    registros = []
    tukey_all = []

    max_workers = _get_max_workers()
    resultados_factor = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for factor in factores:
            future = executor.submit(_evaluar_anova_factor, factor, df)
            futures.append(future)
        for future in futures:
            resultados_factor.append(future.result())

    for registro, df_tukey in resultados_factor:
        if registro is not None:
            registros.append(registro)
        if df_tukey is not None and not df_tukey.empty:
            tukey_all.append(df_tukey)

    df_anova = pd.DataFrame.from_records(registros)
    ruta_anova = os.path.join(ruta_salida, 'anova.csv')
    try:
        df_anova.to_csv(ruta_anova, index=False)
        log(f"CSV ANOVA guardado: {ruta_anova}")
    except Exception as e:
        log(f"No se pudo guardar anova.csv: {e}")

    if tukey_all:
        df_tukey = pd.concat(tukey_all, ignore_index=True)
    else:
        df_tukey = pd.DataFrame()

    ruta_tukey = os.path.join(ruta_salida, 'tukey.csv')
    try:
        df_tukey.to_csv(ruta_tukey, index=False)
        log(f"CSV Tukey guardado: {ruta_tukey}")
    except Exception as e:
        log(f"No se pudo guardar tukey.csv: {e}")

    ruta_box = None
    try:
        fig, axes = plt.subplots(1, len(factores), figsize=(6 * len(factores), 6))
        if len(factores) == 1:
            axes = [axes]
        for ax, factor in zip(axes, factores):
            if 'MONTO APLICADO' not in df.columns or factor not in df.columns:
                ax.set_visible(False)
                continue
            tabla = _obtener_muestra_tabla(df, ['MONTO APLICADO', factor])
            if tabla.empty:
                ax.set_visible(False)
                continue
            tabla.boxplot(column='MONTO APLICADO', by=factor, ax=ax)
            ax.set_title(f'MONTO APLICADO por {factor}')
            ax.set_xlabel(factor)
            ax.set_ylabel('MONTO APLICADO')
        plt.suptitle('')
        plt.tight_layout()
        ruta_box = os.path.join(ruta_graficos, 'boxplot_anova.png')
        plt.savefig(ruta_box, dpi=150)
        plt.close()
        log(f"Boxplot ANOVA guardado: {ruta_box}")
    except Exception as e:
        log(f"No se pudo generar boxplot ANOVA: {e}")

    return {
        'anova_csv': ruta_anova if not df_anova.empty else None,
        'tukey_csv': ruta_tukey if not df_tukey.empty else None,
        'boxplot': ruta_box if ruta_box and os.path.exists(ruta_box) else None,
        'detalles': df_anova,
        'tukey_details': df_tukey
    }
