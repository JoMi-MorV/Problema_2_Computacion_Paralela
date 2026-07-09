from utiles.logger import log  ##comentario importa el logger para emitir mensajes
import dask.dataframe as dd  ##comentario Dask para manipular el dataframe grande de forma perezosa
import numpy as np  ##comentario Numpy para operaciones numéricas y NaN
import os
import json

# F4_tratamiento_variables_deducidas.py
# Validación y limpieza de las variables derivadas que se generan en F3.


NORMALIZAR_COLUMNAS = [
    'MONTO_POR_UNIDAD', 'EDAD', 'RECENCIA',
    'MONTO_BRUTO', 'FRECUENCIA_COMPRA'
]  ##comentario columnas que se escalarán para análisis posteriores


def validar_deducidas(df):
    """Valida y corrige reglas de dominio sobre las variables derivadas."""
    log("===== VALIDACIÓN DE VARIABLES DEDUCIDAS =====")  ##comentario inicio de la validación de variables derivadas

    cols_num = ['MONTO_POR_UNIDAD', 'EDAD', 'HORA_TRANSACCION',
                'FRECUENCIA_COMPRA', 'RECENCIA', 'MONTO_BRUTO']  ##comentario columnas numéricas derivadas

    for col in cols_num:
        if col in df.columns:
            df[col] = dd.to_numeric(df[col], errors='coerce')  ##comentario convierte a numérico y marca errores como NaN

    if 'EDAD' in df.columns:
        fuera_rango = (df['EDAD'] < 12) | (df['EDAD'] > 100)  ##comentario rango lógico de edad válido
        df['EDAD'] = df['EDAD'].where(~fuera_rango, np.nan)  ##comentario invalida edades ilógicas
        log("EDAD: Rango [12-100] aplicado.")  ##comentario registra el ajuste de edad

    if 'MONTO_BRUTO' in df.columns:
        df['MONTO_BRUTO'] = df['MONTO_BRUTO'].where(df['MONTO_BRUTO'] >= 0, np.nan)  ##comentario valores negativos no son válidos
        log("MONTO_BRUTO: Valores negativos corregidos a NaN.")  ##comentario informa corrección

    if 'FRECUENCIA_COMPRA' in df.columns:
        df['FRECUENCIA_COMPRA'] = df['FRECUENCIA_COMPRA'].where(df['FRECUENCIA_COMPRA'] >= 1, 1)  ##comentario ajusta la frecuencia mínima a 1
        log("FRECUENCIA_COMPRA: Normalizada a mínimo 1.")  ##comentario registro de normalización mínima

    if 'ES_FIN_DE_SEMANA' in df.columns:
        df['ES_FIN_DE_SEMANA'] = df['ES_FIN_DE_SEMANA'].astype(bool)  ##comentario asegura tipo booleano
        log("ES_FIN_DE_SEMANA: Formato booleano verificado.")  ##comentario registro de verificación de booleano

    if 'SEGMENTO_MONTO' in df.columns:
        niveles_validos = ['Bajo', 'Medio', 'Alto']  ##comentario categorías permitidas para el segmento
        fuera_rango = ~df['SEGMENTO_MONTO'].isin(niveles_validos)  ##comentario identifica valores no esperados
        df['SEGMENTO_MONTO'] = df['SEGMENTO_MONTO'].where(~fuera_rango, 'Desconocido')  ##comentario marca categorías inválidas
        log("SEGMENTO_MONTO: Valores inválidos marcados como 'Desconocido'.")  ##comentario registro de valores inválidos

    log("VALIDACIÓN DE DEDUCIDAS COMPLETADA")  ##comentario finaliza la validación de variables derivadas
    return df  ##comentario devuelve el DataFrame corregido


def busca_nulos_deducidas(df):
    """Registra los nulos de las variables deducidas para decidir la estrategia de limpieza."""
    log("===== ANÁLISIS DE NULOS EN VARIABLES DEDUCIDAS =====")  ##comentario inicio del análisis de nulos en variables derivadas

    columnas_deducidas = [
        'MONTO_POR_UNIDAD', 'EDAD', 'HORA_TRANSACCION',
        'FRECUENCIA_COMPRA', 'RECENCIA', 'MONTO_BRUTO',
        'ES_FIN_DE_SEMANA', 'SEGMENTO_MONTO'
    ]  ##comentario lista de variables deducidas a evaluar

    presentes = []  ##comentario construye la lista de columnas reales en el DataFrame
    for col in columnas_deducidas:
        if col in df.columns:
            presentes.append(col)
    if not presentes:
        log("No hay variables deducidas presentes para analizar.")  ##comentario no hay variables derivadas disponibles
        return

    total_registros = df.shape[0].compute()  ##comentario total de filas para calcular porcentajes
    nulos_por_col = df[presentes].isna().sum().compute()  ##comentario suma de valores faltantes por columna

    for col, nulos in nulos_por_col.items():
        porcentaje = (nulos / total_registros) * 100 if total_registros else 0  ##comentario calcula porcentaje de nulos
        log(f"\nColumna: {col}")  ##comentario visualiza nombre de la columna
        log(f"Nulos: {nulos}")  ##comentario muestra cantidad de nulos
        log(f"% Nulos: {porcentaje:.2f}%")  ##comentario muestra porcentaje de nulos


def _mediana_aproximada(df, col):
    # Dask no implementa median() exacta en todos los casos; usamos un método aproximado.
    try:
        return df[col].median_approximate().compute()
    except AttributeError:
        return df[col].quantile(0.5).compute()


def limpiar_nulos_deducidos(df):
    """Limpia los nulos de las variables deducidas según porcentaje y tipo de columna."""
    log("===== PROCESANDO ESTRATEGIA ESTRATIFICADA DE VARIABLES DEDUCIDAS =====")  ##comentario inicio de la limpieza de variables derivadas

    columnas_deducidas = [
        'MONTO_POR_UNIDAD', 'EDAD', 'HORA_TRANSACCION',
        'FRECUENCIA_COMPRA', 'RECENCIA', 'MONTO_BRUTO',
        'ES_FIN_DE_SEMANA', 'SEGMENTO_MONTO'
    ]  ##comentario columnas derivadas sujetas a limpieza

    presentes = []  ##comentario lista de columnas presentes
    for col in columnas_deducidas:
        if col in df.columns:
            presentes.append(col)
    if not presentes:
        log("No hay variables deducidas presentes para limpiar.")  ##comentario no hay columnas para procesar
        return df

    total_registros = df.shape[0].compute()  ##comentario obtiene el número de registros totales
    nulos_por_col = df[presentes].isna().sum().compute()  ##comentario suma de nulos de cada columna

    for col, nulos in nulos_por_col.items():
        porcentaje = (nulos / total_registros) * 100 if total_registros else 0  ##comentario calcula porcentaje de ausencias
        if nulos == 0:
            continue  ##comentario omite columnas sin nulos

        if porcentaje <= 5:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Zona Segura: ELIMINANDO filas.")  ##comentario estrategia de borrado de filas
            df = df.dropna(subset=[col])  ##comentario elimina filas con valores faltantes en la columna
        elif porcentaje <= 20:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Zona Imputación: RELLENANDO.")  ##comentario estrategia de imputación mediana o moda
            if np.issubdtype(df[col].dtype, np.number):
                valor_imputacion = _mediana_aproximada(df, col)  ##comentario calcula mediana aproximada compatible con Dask
                df[col] = df[col].fillna(valor_imputacion)  ##comentario rellena nulos numéricos con la mediana
            else:
                moda = df[col].mode().compute()  ##comentario calcula la moda para datos categóricos
                if len(moda) > 0:
                    df[col] = df[col].fillna(moda.iloc[0])  ##comentario rellena nulos categóricos con la moda
        else:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Zona Peligro: DESCARTANDO variable.")  ##comentario estrategia de descarte de columna
            df = df.drop(columns=[col])  ##comentario elimina la columna del DataFrame

    log("Limpieza estratificada de deducidas finalizada.")  ##comentario fin de la limpieza de variables derivadas
    return df  ##comentario devuelve DataFrame limpio


def normalizar_variables(df, output_dir='output/preprocesamiento/modelos'):
    """Estandariza las variables derivadas y guarda los parámetros usados en un JSON."""
    log("===== NORMALIZACIÓN DE VARIABLES DEDUCIDAS =====")  ##comentario inicio de la normalización para análisis posteriores
    parametros = {}  ##comentario almacena medias y desviaciones usadas

    for col in NORMALIZAR_COLUMNAS:
        if col not in df.columns:
            continue  ##comentario omite columnas no presentes

        media = df[col].mean().compute()  ##comentario calcula la media de la columna
        desviacion = df[col].std().compute()  ##comentario calcula la desviación estándar de la columna
        if desviacion == 0 or np.isnan(desviacion):
            log(f"{col}: desviación estándar inválida, no se escala.")  ##comentario evita división por cero
            continue

        nombre_escalado = f"{col}_SCALED"  ##comentario nombre de la nueva columna escalada
        df[nombre_escalado] = (df[col] - media) / desviacion  ##comentario aplica estandarización tipo StandardScaler
        parametros[col] = {'media': float(media), 'desviacion': float(desviacion)}  ##comentario guarda parámetros utilizados
        log(f"{col}: normalizado con media={media:.4f} y std={desviacion:.4f}.")  ##comentario registra los parámetros de escalado

    os.makedirs(output_dir, exist_ok=True)
    ruta_parametros = os.path.join(output_dir, 'parametros_normalizacion.json')
    try:
        with open(ruta_parametros, 'w', encoding='utf-8') as f:
            json.dump(parametros, f, indent=2, ensure_ascii=False)
        log(f"Parámetros de normalización guardados en {ruta_parametros}")
    except Exception as e:
        log(f"No se pudieron guardar los parámetros de normalización: {e}")

    log("Normalización completada con parámetros fijos.")  ##comentario fin de la normalización
    return df  ##comentario devuelve el DataFrame con columnas escaladas
