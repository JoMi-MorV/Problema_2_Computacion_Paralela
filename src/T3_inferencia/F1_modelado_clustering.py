import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utiles.logger import log
from utiles.semilla import get_seed

# Modelado descriptivo mediante clustering K-Means.
# Esta etapa segmenta clientes según su comportamiento de compra.

DEFAULT_K = 3
K_VALUES = range(2, 9)
MAX_CLIENTES_CLUSTER = 30000
OUTPUT_CLUSTER_DIR = Path('output/modelado/clustering')
OUTPUT_GRAFICOS_DIR = Path('output/graficos/modelado')

VARIABLES_RESUMEN = {
    'MONTO APLICADO': {
        'promedio': 'MONTO_APLICADO_PROMEDIO',
        'total': 'MONTO_APLICADO_TOTAL'
    },
    'PORCENTAJE DESCUENTO': {
        'promedio': 'DESCUENTO_PROMEDIO'
    },
    'EDAD': {
        'promedio': 'EDAD_PROMEDIO'
    },
    'RECENCIA': {
        'min': 'RECENCIA_DIAS'
    },
    'MONTO_POR_UNIDAD': {
        'promedio': 'MONTO_POR_UNIDAD_PROMEDIO'
    }
}


def _crear_directorios():
    OUTPUT_CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_column_names(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join([str(x) for x in col if x is not None and str(x) != ''])
            for col in df.columns.values
        ]
    return df


def _registrar_variables_omitidas(presentes, esperadas):
    omitidas = [var for var in esperadas if var not in presentes]
    for var in omitidas:
        log(f"Variable opcional omitida en clustering: {var}")


def _asegurar_variables_deducidas(df):
    if 'FECHA NACIMIENTO' in df.columns and 'EDAD' not in df.columns:
        try:
            df['FECHA NACIMIENTO'] = dd.to_datetime(df['FECHA NACIMIENTO'], errors='coerce')
            df['EDAD'] = 2026 - df['FECHA NACIMIENTO'].map_partitions(lambda s: s.dt.year)
            log('Se calculó EDAD a partir de FECHA NACIMIENTO para clustering.')
        except Exception:
            log('No se pudo calcular EDAD a partir de FECHA NACIMIENTO.')

    if 'FECHA' in df.columns and 'RECENCIA' not in df.columns:
        try:
            df['FECHA'] = dd.to_datetime(df['FECHA'], errors='coerce')
            fecha_maxima = df['FECHA'].max().compute()
            df['RECENCIA'] = (fecha_maxima - df['FECHA']).map_partitions(lambda s: s.dt.days)
            log('Se calculó RECENCIA a partir de FECHA para clustering.')
        except Exception:
            log('No se pudo calcular RECENCIA a partir de FECHA.')

    if 'MONTO_POR_UNIDAD' not in df.columns and 'MONTO APLICADO' in df.columns and 'UNIDADES' in df.columns:
        df['MONTO_POR_UNIDAD'] = df['MONTO APLICADO'] / df['UNIDADES']
        log('Se calculó MONTO_POR_UNIDAD a partir de MONTO APLICADO y UNIDADES para clustering.')

    return df


def _crear_dataset_clientes(df):
    if 'CODIGO CLIENTE' not in df.columns:
        raise ValueError('Imposible generar clustering: falta columna CODIGO CLIENTE.')

    df = _asegurar_variables_deducidas(df)
    columnas_disponibles = set(df.columns)
    _registrar_variables_omitidas(columnas_disponibles, VARIABLES_RESUMEN.keys())

    df_cliente = df.dropna(subset=['CODIGO CLIENTE'])[['CODIGO CLIENTE'] + [col for col in VARIABLES_RESUMEN if col in columnas_disponibles]]

    operador_agg = {
        'promedio': 'mean',
        'total': 'sum',
        'min': 'min'
    }

    if hasattr(df_cliente, 'compute'):
        convertir_cols = {col: float for col in df_cliente.columns if col != 'CODIGO CLIENTE'}
        df_cliente = df_cliente.map_partitions(lambda pdf: pdf.astype(convertir_cols, errors='ignore'))
        grupos = df_cliente.groupby('CODIGO CLIENTE')
        agg_dict = {}
        for columna, operaciones in VARIABLES_RESUMEN.items():
            if columna not in df_cliente.columns:
                continue
            for operacion, nombre in operaciones.items():
                funcion = operador_agg.get(operacion)
                if funcion is None:
                    continue
                agg_dict[columna] = agg_dict.get(columna, []) + [funcion]

        resumen = grupos.agg(agg_dict)
        resumen = dask.compute(resumen)[0]
    else:
        groups = df_cliente.groupby('CODIGO CLIENTE')
        agg_dict = {}
        for columna, operaciones in VARIABLES_RESUMEN.items():
            if columna not in df_cliente.columns:
                continue
            for operacion in operaciones:
                funcion = operador_agg.get(operacion)
                if funcion is None:
                    continue
                agg_dict[columna] = agg_dict.get(columna, []) + [funcion]
        resumen = groups.agg(agg_dict)

    resumen = _flatten_column_names(resumen)

    renombrar = {}
    for columna, operaciones in VARIABLES_RESUMEN.items():
        if columna not in df_cliente.columns:
            continue
        for operacion, nombre in operaciones.items():
            funcion = operador_agg.get(operacion)
            if funcion is None:
                continue
            clave = f"{columna}_{funcion}"
            if clave in resumen.columns:
                renombrar[clave] = nombre
    if renombrar:
        resumen = resumen.rename(columns=renombrar)

    resumen = resumen.reset_index()

    if 'COMPRA_CUENTA' not in resumen.columns:
        if hasattr(df_cliente, 'compute'):
            conteo = df_cliente.groupby('CODIGO CLIENTE').size().compute()
        else:
            conteo = df_cliente.groupby('CODIGO CLIENTE').size()
        conteo = conteo.rename('COMPRA_CUENTA')
        if isinstance(resumen.index, pd.Index) and resumen.index.name == 'CODIGO CLIENTE':
            resumen = resumen.join(conteo, on='CODIGO CLIENTE')
        else:
            resumen = resumen.merge(conteo.reset_index(), on='CODIGO CLIENTE', how='left')

    if resumen.empty:
        raise ValueError('El dataset de clientes queda vacío después de agrupar por cliente.')

    return resumen


def _seleccionar_columnas_clustering(resumen):
    columnas_objetivo = [
        'COMPRA_CUENTA', 'MONTO_APLICADO_PROMEDIO', 'MONTO_APLICADO_TOTAL',
        'DESCUENTO_PROMEDIO', 'EDAD_PROMEDIO', 'RECENCIA_DIAS',
        'MONTO_POR_UNIDAD_PROMEDIO'
    ]
    usadas = [col for col in columnas_objetivo if col in resumen.columns]
    if not usadas:
        raise ValueError('No hay variables numéricas disponibles para clustering por cliente.')
    return usadas


def _limpiar_y_estandarizar(resumen, columnas, seed):
    datos = resumen[['CODIGO CLIENTE'] + columnas].copy()
    datos[columnas] = datos[columnas].apply(pd.to_numeric, errors='coerce')
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna(subset=columnas)

    if datos.empty:
        raise ValueError('No hay clientes válidos para clustering después de limpiar nulos e infinitos.')

    if len(datos) > MAX_CLIENTES_CLUSTER:
        log(f'Muestreando clientes para clustering: {len(datos)} -> {MAX_CLIENTES_CLUSTER}')
        datos = datos.sample(n=MAX_CLIENTES_CLUSTER, random_state=seed).reset_index(drop=True)

    scaler = StandardScaler()
    matriz = scaler.fit_transform(datos[columnas])
    return datos.reset_index(drop=True), matriz


def _evaluar_k_elbow(X):
    inercia = []
    for k in K_VALUES:
        modelo = KMeans(n_clusters=k, random_state=get_seed(), n_init=10)
        modelo.fit(X)
        inercia.append({'k': k, 'inercia': float(modelo.inertia_)})
    return pd.DataFrame(inercia)


def _guardar_metodo_codo(inercia_df):
    ruta_csv = OUTPUT_CLUSTER_DIR / 'metodo_codo.csv'
    inercia_df.to_csv(ruta_csv, index=False)
    log(f'Método del codo guardado: {ruta_csv}')

    plt.figure(figsize=(8, 5))
    plt.plot(inercia_df['k'], inercia_df['inercia'], marker='o', linestyle='-', color='tab:blue')
    plt.title('Método del codo para K-Means')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia (suma de distancias al cuadrado)')
    plt.xticks(inercia_df['k'])
    plt.grid(alpha=0.3)
    ruta_png = OUTPUT_GRAFICOS_DIR / 'metodo_codo.png'
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    log(f'Gráfico del método del codo guardado: {ruta_png}')


def _ajustar_kmeans(X, k, seed):
    modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
    modelo.fit(X)
    return modelo


def _guardar_clusters(datos, labels):
    datos = datos.copy()
    datos['CLUSTER'] = labels
    ruta_csv = OUTPUT_CLUSTER_DIR / 'clientes_con_cluster.csv'
    datos.to_csv(ruta_csv, index=False)
    log(f'Resultados de clientes con cluster guardados: {ruta_csv}')
    return datos


def _calcular_metrica_silhouette(X, labels):
    if len(set(labels)) < 2:
        raise ValueError('No se puede calcular silhouette_score con un solo cluster.')
    return float(silhouette_score(X, labels))


def _guardar_metricas(k, sil_score, num_clientes, columnas):
    ruta_csv = OUTPUT_CLUSTER_DIR / 'metricas_clustering.csv'
    modelo = 'KMeans'
    datos = pd.DataFrame([{
        'modelo': modelo,
        'k': k,
        'silhouette_score': sil_score,
        'clientes_utilizados': num_clientes,
        'variables_usadas': ', '.join(columnas)
    }])
    datos.to_csv(ruta_csv, index=False)
    log(f'Métricas de clustering guardadas: {ruta_csv}')


def _guardar_resumen_clusters(datos, columnas):
    agrupado = datos.groupby('CLUSTER')
    total = len(datos)
    filas = []
    for cluster, grupo in agrupado:
        registro = {
            'cluster': cluster,
            'clientes': len(grupo),
            'porcentaje': 100 * len(grupo) / total,
        }
        for col in columnas:
            registro[f'{col}_media'] = grupo[col].mean()
            registro[f'{col}_mediana'] = grupo[col].median()
            registro[f'{col}_min'] = grupo[col].min()
            registro[f'{col}_max'] = grupo[col].max()
        filas.append(registro)
    resumen = pd.DataFrame(filas).sort_values('cluster')
    ruta_csv = OUTPUT_CLUSTER_DIR / 'resumen_clusters.csv'
    resumen.to_csv(ruta_csv, index=False)
    log(f'Resumen por cluster guardado: {ruta_csv}')
    return resumen


def _interpretar_clusters(datos, columnas):
    promedios = datos.groupby('CLUSTER')[columnas].mean()
    global_prom = datos[columnas].mean()
    interpretaciones = []

    for cluster, fila in promedios.iterrows():
        partes = [f'Cluster {cluster}:']
        comparaciones = []
        for col in columnas:
            valor = fila[col]
            nivel = 'alto' if valor > global_prom[col] else 'bajo' if valor < global_prom[col] else 'similar'
            comparaciones.append(f'{col} {nivel}')
        resumen = ', '.join(comparaciones)
        partes.append(f'Promedios: {resumen}.')
        if 'COMPRA_CUENTA' in columnas and 'MONTO_APLICADO_TOTAL' in columnas:
            if fila.get('COMPRA_CUENTA', 0) > global_prom.get('COMPRA_CUENTA', 0) and fila.get('MONTO_APLICADO_TOTAL', 0) > global_prom.get('MONTO_APLICADO_TOTAL', 0):
                partes.append('Clientes frecuentes y de mayor valor.')
            elif fila.get('COMPRA_CUENTA', 0) < global_prom.get('COMPRA_CUENTA', 0) and fila.get('RECENCIA_DIAS', np.inf) > global_prom.get('RECENCIA_DIAS', np.inf):
                partes.append('Clientes menos activos con mayor recencia.')
        interpretaciones.append(' '.join(partes))

    ruta_txt = OUTPUT_CLUSTER_DIR / 'interpretacion_clusters.txt'
    with open(ruta_txt, 'w', encoding='utf-8') as archivo:
        archivo.write('Interpretación automática por cluster:\n\n')
        archivo.write('\n'.join(interpretaciones))
    log(f'Interpretación de clusters guardada: {ruta_txt}')


def _graficar_datos_cluster(datos):
    if 'CLUSTER' in datos.columns:
        conteo = datos['CLUSTER'].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(conteo.index.astype(str), conteo.values, color='tab:blue', alpha=0.8)
        plt.title('Distribución de clientes por cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Clientes')
        plt.tight_layout()
        ruta = OUTPUT_GRAFICOS_DIR / 'distribucion_clusters.png'
        plt.savefig(ruta, dpi=150)
        plt.close()
        log(f'Gráfico de distribución por cluster guardado: {ruta}')

    if 'MONTO_APLICADO_TOTAL' in datos.columns and 'COMPRA_CUENTA' in datos.columns:
        plt.figure(figsize=(8, 6))
        for cluster in sorted(datos['CLUSTER'].unique()):
            sub = datos[datos['CLUSTER'] == cluster]
            plt.scatter(sub['MONTO_APLICADO_TOTAL'], sub['COMPRA_CUENTA'], alpha=0.6, label=f'Cluster {cluster}')
        plt.title('Monto total vs frecuencia de compra por cluster')
        plt.xlabel('Monto total aplicado')
        plt.ylabel('Cantidad de compras')
        plt.legend()
        plt.tight_layout()
        ruta = OUTPUT_GRAFICOS_DIR / 'clusters_monto_frecuencia.png'
        plt.savefig(ruta, dpi=150)
        plt.close()
        log(f'Gráfico monto vs frecuencia guardado: {ruta}')

    if 'RECENCIA_DIAS' in datos.columns and 'MONTO_APLICADO_TOTAL' in datos.columns:
        plt.figure(figsize=(8, 6))
        for cluster in sorted(datos['CLUSTER'].unique()):
            sub = datos[datos['CLUSTER'] == cluster]
            plt.scatter(sub['RECENCIA_DIAS'], sub['MONTO_APLICADO_TOTAL'], alpha=0.6, label=f'Cluster {cluster}')
        plt.title('Recencia vs monto total por cluster')
        plt.xlabel('Recencia (días)')
        plt.ylabel('Monto total aplicado')
        plt.legend()
        plt.tight_layout()
        ruta = OUTPUT_GRAFICOS_DIR / 'clusters_recencia_monto.png'
        plt.savefig(ruta, dpi=150)
        plt.close()
        log(f'Gráfico recencia vs monto guardado: {ruta}')


def modelado_descriptivo_clustering(df, seed=None):
    seed = get_seed() if seed is None else seed
    _crear_directorios()
    log('ETAPA 10: MODELADO DESCRIPTIVO - CLUSTERING')
    log('Generando dataset por cliente')
    clientes = _crear_dataset_clientes(df)
    columnas = _seleccionar_columnas_clustering(clientes)
    log(f'Variables usadas para clustering: {columnas}')
    datos_limpios, X = _limpiar_y_estandarizar(clientes, columnas, seed)

    log('Aplicando método del codo para escoger k')
    inercia_df = _evaluar_k_elbow(X)
    _guardar_metodo_codo(inercia_df)

    k = DEFAULT_K
    log(f'Seleccionando k = {k} para el modelo de clustering')
    modelo = _ajustar_kmeans(X, k, seed)
    clientes_cluster = _guardar_clusters(datos_limpios, modelo.labels_)

    log('Calculando métricas de calidad del clustering')
    sil_score = _calcular_metrica_silhouette(X, modelo.labels_)
    _guardar_metricas(k, sil_score, len(clientes_cluster), columnas)

    log('Generando resumen por cluster')
    _guardar_resumen_clusters(clientes_cluster, columnas)

    log('Generando interpretación automática de clusters')
    _interpretar_clusters(clientes_cluster, columnas)

    log('Generando gráficos de apoyo para el modelado')
    _graficar_datos_cluster(clientes_cluster)
    log('Modelado descriptivo finalizado correctamente')
