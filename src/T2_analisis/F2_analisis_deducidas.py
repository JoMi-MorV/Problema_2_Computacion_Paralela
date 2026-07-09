import os
import re
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
from pandas.api import types as ptypes

# F2_analisis_deducidas.py
# Funciones reutilizables para el análisis inicial y el reanálisis de las variables deducidas.


VARIABLES_DEDUCIDAS = [
    'MONTO_POR_UNIDAD', 'EDAD', 'HORA_TRANSACCION',
    'FRECUENCIA_COMPRA', 'RECENCIA', 'MONTO_BRUTO',
    'ES_FIN_DE_SEMANA', 'SEGMENTO_MONTO'
]


def _crear_directorio(ruta):
    os.makedirs(ruta, exist_ok=True)


def _guardar_csv(df, ruta):
    df.to_csv(ruta, index=True)


def _obtener_ruta_grafico(ruta_salida, subcarpeta, nombre_archivo):
    etiqueta = os.path.basename(os.path.dirname(ruta_salida))
    ruta_graficos = os.path.join('output', 'graficos', subcarpeta, etiqueta)
    os.makedirs(ruta_graficos, exist_ok=True)
    return os.path.join(ruta_graficos, nombre_archivo)


def _sanitizar_nombre(nombre):
    return re.sub(r'[^A-Za-z0-9_]+', '_', nombre)


def _obtener_muestra(df, cols, sample_frac=0.01, max_rows=20000, seed=None):
    seed = get_seed() if seed is None else seed
    datos = df[cols] if cols else df
    return obtener_muestra_pandas(datos, max_rows=max_rows, seed=seed)


def _columnas_numericas(df, columnas):
    return [col for col in columnas if col in df.columns and getattr(df[col].dtype, 'kind', '') in 'iufc']


def _calcular_estadisticas_completas(df, columnas):
    filas = []
    for col in columnas:
        serie = df[col]
        log(f"Calculando métricas completas para {col}")
        q = serie.quantile([0.25, 0.5, 0.75])
        mean = serie.mean()
        std = serie.std()
        var = serie.var()
        min_val = serie.min()
        max_val = serie.max()
        skew = serie.skew()
        kurt = serie.kurtosis()
        mean, std, var, min_val, max_val, skew, kurt = dask.compute(
            mean, std, var, min_val, max_val, skew, kurt
        )
        q = q.compute()
        q1 = q.loc[0.25] if 0.25 in q.index else np.nan
        median = q.loc[0.5] if 0.5 in q.index else np.nan
        q3 = q.loc[0.75] if 0.75 in q.index else np.nan
        iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
        mode = np.nan
        try:
            muestra_col = obtener_muestra_pandas(serie, max_rows=10000)
            vc = muestra_col.value_counts(dropna=False)
            mode = vc.idxmax() if len(vc) > 0 else np.nan
        except Exception:
            mode = np.nan
        rango = max_val - min_val if pd.notna(min_val) and pd.notna(max_val) else np.nan
        coef_var = std / mean if mean not in (0, None, np.nan) else np.nan
        filas.append({
            'variable': col,
            'media': mean,
            'mediana': median,
            'moda': mode,
            'mínimo': min_val,
            'máximo': max_val,
            'rango': rango,
            'cuartil_1': q1,
            'cuartil_3': q3,
            'IQR': iqr,
            'desviación_estándar': std,
            'varianza': var,
            'coeficiente_variación': coef_var,
            'asimetría': skew,
            'curtosis': kurt,
        })
    return pd.DataFrame(filas).set_index('variable')


def _guardar_graficos_descriptivos(df, columnas, etiqueta):
    ruta_graficos = os.path.join('output', 'graficos', 'descriptivos', etiqueta)
    os.makedirs(ruta_graficos, exist_ok=True)
    for col in columnas:
        sanitized = _sanitizar_nombre(col)
        muestra = _obtener_muestra(df[[col]], [col], sample_frac=0.02, max_rows=12000)
        if muestra.empty:
            log(f"Omitido gráfico para {col}: muestra vacía")
            continue
        valores = muestra[col].dropna()
        if valores.empty:
            log(f"Omitido gráfico para {col}: no hay datos válidos")
            continue

        # Histograma
        log(f"Generando histograma para {col}")
        plt.figure(figsize=(8, 5))
        plt.hist(valores, bins=30, color='tab:blue', alpha=0.7)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        ruta_hist = os.path.join(ruta_graficos, f"{etiqueta}_{sanitized}_histograma.png")
        plt.savefig(ruta_hist, dpi=150)
        plt.close()
        log(f"Histograma guardado: {ruta_hist}")

        # KDE
        if len(valores) > 1:
            try:
                log(f"Generando KDE para {col}")
                kde = stats.gaussian_kde(valores)
                xs = np.linspace(valores.min(), valores.max(), 200)
                plt.figure(figsize=(8, 5))
                plt.plot(xs, kde(xs), color='tab:green')
                plt.title(f'KDE de {col}')
                plt.xlabel(col)
                plt.ylabel('Densidad')
                plt.tight_layout()
                ruta_kde = os.path.join(ruta_graficos, f"{etiqueta}_{sanitized}_kde.png")
                plt.savefig(ruta_kde, dpi=150)
                plt.close()
                log(f"KDE guardado: {ruta_kde}")
            except Exception:
                log(f"No se pudo generar KDE para {col}")

        # QQ plot
        try:
            log(f"Generando QQ plot para {col}")
            plt.figure(figsize=(8, 5))
            stats.probplot(valores, dist='norm', plot=plt)
            plt.title(f'QQ Plot de {col}')
            plt.tight_layout()
            ruta_qq = os.path.join(ruta_graficos, f"{etiqueta}_{sanitized}_qq.png")
            plt.savefig(ruta_qq, dpi=150)
            plt.close()
            log(f"QQ plot guardado: {ruta_qq}")
        except Exception:
            log(f"No se pudo generar QQ plot para {col}")


def analisis_deducidas(df, etiqueta, output_base="output/analysis/deducidas"):
    """Genera estadísticas y gráficos para las variables derivadas del pipeline."""
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)
    log(f"Iniciando análisis de variables deducidas: {etiqueta}")

    columnas_deducidas = [col for col in VARIABLES_DEDUCIDAS if col in df.columns]

    estadisticas = None
    columnas_numericas = _columnas_numericas(df, columnas_deducidas)

    if columnas_deducidas:
        log("Generando estadísticas descriptivas de variables deducidas")
        muestra_df = _obtener_muestra(df[columnas_deducidas], columnas_deducidas, max_rows=50000)
        estadisticas = muestra_df.describe().transpose()
        ruta_estadisticas = os.path.join(ruta_salida, "tabla_estadisticas_deducidas.csv")
        _guardar_csv(estadisticas, ruta_estadisticas)
        log(f"Tabla de estadísticas guardada: {ruta_estadisticas}")

    if columnas_numericas:
        log("Generando tabla de estadísticas completa para variables deducidas")
        completas = _calcular_estadisticas_completas(df, columnas_numericas)
        ruta_completa = os.path.join(ruta_salida, "tabla_estadisticas_completa.csv")
        _guardar_csv(completas, ruta_completa)
        log(f"Tabla de estadísticas completa guardada: {ruta_completa}")

        log("Guardando gráficos descriptivos individuales para variables deducidas")
        _guardar_graficos_descriptivos(df, columnas_numericas, etiqueta)

    if columnas_deducidas:
        try:
            log("Calculando matriz de correlación de variables deducidas")
            correlacion = _obtener_muestra(df[columnas_deducidas], columnas_deducidas, max_rows=50000).corr()
            ruta_corr = os.path.join(ruta_salida, "matriz_correlacion_deducidas.csv")
            _guardar_csv(correlacion, ruta_corr)
            log(f"Tabla de correlación guardada: {ruta_corr}")
        except Exception:
            log("No se pudo calcular la matriz de correlación de variables deducidas.")
            pass

    muestra = _obtener_muestra(df, columnas_deducidas)

    if not muestra.empty:
        log("Generando histograma de variables deducidas")
        plt.figure(figsize=(12, 16))
        for idx, col in enumerate(columnas_deducidas, start=1):
            plt.subplot(len(columnas_deducidas), 1, idx)
            if muestra[col].dtype == 'bool':
                muestra[col].value_counts().plot(kind='bar', color='tab:blue', alpha=0.7)
                plt.ylabel('Frecuencia')
            else:
                muestra[col].hist(bins=30, color='tab:blue', alpha=0.7)
                plt.ylabel('Frecuencia')
            plt.title(f"Histograma de {col}")
            plt.xlabel(col)
        plt.tight_layout()
        nombre_hist = f"{etiqueta}_histogramas_deducidas.png"
        ruta_hist = _obtener_ruta_grafico(ruta_salida, 'deducidas', nombre_hist)
        plt.savefig(ruta_hist, dpi=150)
        plt.close()
        log(f"Histograma guardado: {ruta_hist}")

    if 'MONTO_POR_UNIDAD' in df.columns and 'SEGMENTO_MONTO' in df.columns:
        log("Generando boxplot de MONTO_POR_UNIDAD por SEGMENTO_MONTO")
        muestra_seg = _obtener_muestra(df[['MONTO_POR_UNIDAD', 'SEGMENTO_MONTO']], ['MONTO_POR_UNIDAD', 'SEGMENTO_MONTO'])
        plt.figure(figsize=(8, 6))
        muestra_seg.boxplot(column='MONTO_POR_UNIDAD', by='SEGMENTO_MONTO', grid=False)
        plt.title('Boxplot de MONTO_POR_UNIDAD por SEGMENTO_MONTO')
        plt.suptitle('')
        plt.xlabel('SEGMENTO_MONTO')
        plt.ylabel('MONTO_POR_UNIDAD')
        plt.tight_layout()
        nombre_boxplot = f"{etiqueta}_boxplot_monto_por_segmento.png"
        ruta_boxplot = _obtener_ruta_grafico(ruta_salida, 'deducidas', nombre_boxplot)
        plt.savefig(ruta_boxplot, dpi=150)
        plt.close()
        log(f"Boxplot guardado: {ruta_boxplot}")

    # --- Análisis avanzado adicional para variables deducidas ---
    # 1) Frecuencias para variables categóricas deducidas
    categorias = []
    if not muestra.empty:
        for c in columnas_deducidas:
            try:
                if not ptypes.is_numeric_dtype(df[c].dtype):
                    categorias.append(c)
            except Exception:
                # fallback: treat as categorical if cannot determine
                categorias.append(c)
    for cat in categorias:
        try:
            muestra_cat = _obtener_muestra(df[[cat]], [cat], max_rows=50000)
            conteo = muestra_cat[cat].value_counts(dropna=False).rename_axis(cat).reset_index(name='count')
            ruta_frec = os.path.join(ruta_salida, f"tabla_frecuencias_{cat}.csv")
            _guardar_csv(conteo, ruta_frec)
            log(f"Tabla de frecuencias deducidas guardada: {ruta_frec}")
        except Exception:
            log(f"No se pudo generar tabla de frecuencias para {cat}")

    # 2) Pruebas de normalidad (Shapiro) para variables numéricas deducidas
    normalidad_records = []
    for col in columnas_numericas:
        try:
            muestra_col = _obtener_muestra(df[[col]], [col], sample_frac=0.02, max_rows=5000)
            serie = muestra_col[col].dropna()
            if len(serie) < 3:
                sh_p = np.nan
            else:
                sh_stat, sh_p = stats.shapiro(serie.values)
        except Exception:
            sh_p = np.nan
        normalidad_records.append({'variable': col, 'shapiro_pvalue': sh_p})
    try:
        df_norm = pd.DataFrame(normalidad_records).set_index('variable')
        ruta_norm = os.path.join(ruta_salida, 'normalidad_deducidas.csv')
        _guardar_csv(df_norm, ruta_norm)
        log(f"Tabla de normalidad deducidas guardada: {ruta_norm}")
    except Exception:
        log("No se pudo guardar tabla de normalidad deducidas")

    # 3) Correlaciones y gráficos para pares relevantes
    pares = [
        ('EDAD', ['MONTO_POR_UNIDAD', 'MONTO_BRUTO', 'MONTO APLICADO']),
        ('RECENCIA', ['FRECUENCIA_COMPRA']),
        ('MONTO_BRUTO', ['MONTO_POR_UNIDAD', 'MONTO APLICADO']),
        ('ES_FIN_DE_SEMANA', ['MONTO_POR_UNIDAD', 'MONTO APLICADO']),
        ('SEGMENTO_MONTO', ['MONTO_POR_UNIDAD', 'MONTO_BRUTO', 'MONTO APLICADO'])
    ]
    corr_records = []
    for left, rights in pares:
        if left not in df.columns:
            continue
        for right in rights:
            if right not in df.columns:
                continue
            log(f"Analizando asociación: {left} vs {right}")
            muestra_par = _obtener_muestra(df[[left, right]], [left, right], sample_frac=0.02, max_rows=20000)
            if muestra_par.empty:
                log(f"Omitido par {left} vs {right}: muestra vacía")
                continue
            a = muestra_par[left].dropna()
            b = muestra_par[right].dropna()
            if a.empty or b.empty:
                continue
            # align lengths
            n = min(len(a), len(b))
            a = a.iloc[:n]
            b = b.iloc[:n]

            # normalidad rápida
            try:
                sh_a = stats.shapiro(a.values)[1] if len(a) >= 3 else np.nan
            except Exception:
                sh_a = np.nan
            try:
                sh_b = stats.shapiro(b.values)[1] if len(b) >= 3 else np.nan
            except Exception:
                sh_b = np.nan

            use_pearson = False
            try:
                if (not np.isnan(sh_a) and sh_a > 0.05) and (not np.isnan(sh_b) and sh_b > 0.05):
                    use_pearson = True
            except Exception:
                use_pearson = False
            # determine if numeric or categorical using pandas types
            try:
                is_bool_a = ptypes.is_bool_dtype(a.dtype)
                is_bool_b = ptypes.is_bool_dtype(b.dtype)
                is_num_a = ptypes.is_numeric_dtype(a.dtype)
                is_num_b = ptypes.is_numeric_dtype(b.dtype)
            except Exception:
                is_bool_a = False
                is_bool_b = False
                is_num_a = False
                is_num_b = False

            if is_bool_a or is_bool_b or (not is_num_a) or (not is_num_b):
                # use group comparison (boxplot) if one is categorical/boolean
                try:
                    nombre_box = f"{etiqueta}_{_sanitizar_nombre(left)}_vs_{_sanitizar_nombre(right)}_box.png"
                    ruta_box = _obtener_ruta_grafico(ruta_salida, 'deducidas', nombre_box)
                    plt.figure(figsize=(8, 5))
                    if a.dtype == bool or not np.issubdtype(a.dtype, np.number):
                        # group by left
                        try:
                            muestra_par.boxplot(column=right, by=left)
                        except Exception:
                            muestra_par.groupby(left)[right].plot(kind='box')
                    else:
                        try:
                            muestra_par.boxplot(column=left, by=right)
                        except Exception:
                            muestra_par.groupby(right)[left].plot(kind='box')
                    plt.title(f'{left} vs {right}')
                    plt.suptitle('')
                    plt.tight_layout()
                    plt.savefig(ruta_box, dpi=150)
                    plt.close()
                    log(f"Boxplot guardado: {ruta_box}")
                except Exception:
                    log(f"No se pudo generar boxplot para {left} vs {right}")
                corr_records.append({'var1': left, 'var2': right, 'method': 'boxplot', 'corr': np.nan, 'pvalue': np.nan, 'n': n})
            else:
                # numeric-numeric: correlation
                if np.unique(a.values).size < 2 or np.unique(b.values).size < 2:
                    corr, p = np.nan, np.nan
                    method = 'constant'
                else:
                    try:
                        if use_pearson:
                            corr, p = stats.pearsonr(a.values, b.values)
                            method = 'pearson'
                        else:
                            corr, p = stats.spearmanr(a.values, b.values)
                            method = 'spearman'
                    except Exception:
                        corr, p = np.nan, np.nan
                        method = 'error'

                # scatter plot with line
                try:
                    nombre_scatter = f"{etiqueta}_{_sanitizar_nombre(left)}_vs_{_sanitizar_nombre(right)}_scatter.png"
                    ruta_scatter = _obtener_ruta_grafico(ruta_salida, 'deducidas', nombre_scatter)
                    plt.figure(figsize=(6, 6))
                    plt.scatter(a, b, alpha=0.6)
                    plt.xlabel(left)
                    plt.ylabel(right)
                    plt.title(f'{left} vs {right} ({method})')
                    # regression line if pearson
                    if method == 'pearson' and np.isfinite(corr):
                        try:
                            m, c = np.polyfit(a, b, 1)
                            xs = np.linspace(a.min(), a.max(), 100)
                            plt.plot(xs, m * xs + c, color='red')
                        except Exception:
                            pass
                    plt.tight_layout()
                    plt.savefig(ruta_scatter, dpi=150)
                    plt.close()
                    log(f"Scatter guardado: {ruta_scatter}")
                except Exception:
                    log(f"No se pudo generar scatter para {left} vs {right}")

                corr_records.append({'var1': left, 'var2': right, 'method': method, 'corr': corr, 'pvalue': p, 'n': n})

    # guardar correlaciones específicas
    try:
        df_corr_esp = pd.DataFrame.from_records(corr_records)
        ruta_corr_esp = os.path.join(ruta_salida, 'correlaciones_deducidas.csv')
        df_corr_esp.to_csv(ruta_corr_esp, index=False)
        log(f"Correlaciones específicas guardadas: {ruta_corr_esp}")
    except Exception:
        log("No se pudo guardar correlaciones específicas")

    return {
        'estadisticas': estadisticas,
        'columnas': columnas_deducidas,
        'output_folder': ruta_salida,
        'correlaciones_especificas': ruta_corr_esp if 'ruta_corr_esp' in locals() else None,
        'normalidad': ruta_norm if 'ruta_norm' in locals() else None
    }
