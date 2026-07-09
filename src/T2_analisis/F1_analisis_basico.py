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

# F1_analisis_basico.py
# Funciones reutilizables para el análisis inicial y el reanálisis de las variables básicas.


NUMERICAS_BASICAS = [
    'SKU', 'UNIDADES', 'PORCENTAJE DESCUENTO',
    'MONTO APLICADO', 'BOLETA', 'LOCAL', 'GENERO'
]
CATEGORICAS_BASICAS = ['CANAL', 'PRODUCTO', 'CODIGO CLIENTE', 'RUN CLIENTE']


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
        muestra = _obtener_muestra(df[[col]], [col], sample_frac=0.02, max_rows=15000)
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


def analisis_basico(df, etiqueta, output_base="output/analysis/basic"):
    """Genera estadísticas, tablas de frecuencia y gráficos para las variables básicas."""
    ruta_salida = os.path.join(output_base, etiqueta)
    _crear_directorio(ruta_salida)
    log(f"Iniciando análisis de variables básicas: {etiqueta}")

    columnas_numericas = [col for col in NUMERICAS_BASICAS if col in df.columns]
    columnas_categoricas = [col for col in CATEGORICAS_BASICAS if col in df.columns]
    columnas_numericas_validas = _columnas_numericas(df, columnas_numericas)

    estadisticas = None
    if columnas_numericas_validas:
        log("Generando estadísticas descriptivas básicas")
        muestra_df = _obtener_muestra(df[columnas_numericas_validas], columnas_numericas_validas, max_rows=50000)
        estadisticas = muestra_df.describe().transpose()
        ruta_estadisticas = os.path.join(ruta_salida, "tabla_estadisticas_basicas.csv")
        _guardar_csv(estadisticas, ruta_estadisticas)
        log(f"Tabla de estadísticas guardada: {ruta_estadisticas}")

        log("Generando tabla de estadísticas completa para variables numéricas básicas")
        completas = _calcular_estadisticas_completas(df, columnas_numericas_validas)
        ruta_completa = os.path.join(ruta_salida, "tabla_estadisticas_completa.csv")
        _guardar_csv(completas, ruta_completa)
        log(f"Tabla de estadísticas completa guardada: {ruta_completa}")

        log("Guardando gráficos descriptivos individuales para variables numéricas básicas")
        _guardar_graficos_descriptivos(df, columnas_numericas_validas, etiqueta)

    for col in columnas_categoricas:
        log(f"Generando tabla de frecuencias para {col}")
        muestra_col = _obtener_muestra(df[[col]], [col], max_rows=50000)
        conteo = muestra_col[col].value_counts(dropna=False).rename_axis(col).reset_index(name='count')
        ruta_frecuencia = os.path.join(ruta_salida, f"tabla_frecuencias_{col}.csv")
        _guardar_csv(conteo, ruta_frecuencia)
        log(f"Tabla de frecuencias guardada: {ruta_frecuencia}")

    if columnas_numericas_validas:
        try:
            log("Calculando matriz de correlación básica")
            correlacion = _obtener_muestra(df[columnas_numericas_validas], columnas_numericas_validas, max_rows=50000).corr()
            ruta_corr_csv = os.path.join(ruta_salida, "matriz_correlacion_basica.csv")
            _guardar_csv(correlacion, ruta_corr_csv)
            log(f"Tabla de correlación guardada: {ruta_corr_csv}")
        except Exception:
            log("No se pudo calcular la matriz de correlación básica.")
            pass

    muestra = _obtener_muestra(df, columnas_numericas_validas)

    if not muestra.empty:
        log("Generando histograma de variables básicas")
        plt.figure(figsize=(12, 16))
        for idx, col in enumerate(columnas_numericas_validas, start=1):
            plt.subplot(len(columnas_numericas_validas), 1, idx)
            muestra[col].hist(bins=30, color="tab:blue", alpha=0.7)
            plt.title(f"Histograma de {col}")
            plt.xlabel(col)
            plt.ylabel("Frecuencia")
        plt.tight_layout()
        nombre_hist = f"{etiqueta}_histogramas_basicos.png"
        ruta_hist = _obtener_ruta_grafico(ruta_salida, 'basic', nombre_hist)
        plt.savefig(ruta_hist, dpi=150)
        plt.close()
        log(f"Histograma guardado: {ruta_hist}")

    if 'MONTO APLICADO' in df.columns and 'CANAL' in df.columns:
        log("Generando boxplot de MONTO APLICADO por CANAL")
        muestra_cat = _obtener_muestra(df[['MONTO APLICADO', 'CANAL']], ['MONTO APLICADO', 'CANAL'])
        plt.figure(figsize=(8, 6))
        muestra_cat.boxplot(column='MONTO APLICADO', by='CANAL', grid=False)
        plt.title('Boxplot de MONTO APLICADO por CANAL')
        plt.suptitle('')
        plt.xlabel('CANAL')
        plt.ylabel('MONTO APLICADO')
        plt.tight_layout()
        nombre_boxplot = f"{etiqueta}_boxplot_monto_por_canal.png"
        ruta_boxplot = _obtener_ruta_grafico(ruta_salida, 'basic', nombre_boxplot)
        plt.savefig(ruta_boxplot, dpi=150)
        plt.close()
        log(f"Boxplot guardado: {ruta_boxplot}")

    if columnas_numericas_validas:
        try:
            log("Generando matriz de correlación básica")
            corr = _obtener_muestra(df[columnas_numericas_validas], columnas_numericas_validas, max_rows=50000).corr()
            plt.figure(figsize=(8, 6))
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlación')
            plt.xticks(range(len(columnas_numericas_validas)), columnas_numericas_validas, rotation=45, ha='right')
            plt.yticks(range(len(columnas_numericas_validas)), columnas_numericas_validas)
            plt.title('Matriz de correlación básica')
            plt.tight_layout()
            nombre_corr = f"{etiqueta}_matriz_correlacion_basica.png"
            ruta_corr = _obtener_ruta_grafico(ruta_salida, 'basic', nombre_corr)
            plt.savefig(ruta_corr, dpi=150)
            plt.close()
            log(f"Matriz de correlación guardada: {ruta_corr}")
        except Exception:
            log("No se pudo generar la matriz de correlación básica.")
            pass

    return {
        'estadisticas': estadisticas,
        'categoricas': columnas_categoricas,
        'numericas': columnas_numericas_validas,
        'output_folder': ruta_salida
    }
