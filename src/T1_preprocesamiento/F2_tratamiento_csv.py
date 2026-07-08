from utiles.logger import log  ##comentario logger para registrar mensajes y métricas
import dask.dataframe as dd  ##comentario Dask DataFrame se usa para manipular datos grandes en paralelo
import numpy as np  ##comentario Numpy se usa para funciones numéricas y NaN

SEMILLA_ALEATORIEDAD = 42  ##comentario semilla fija para cualquier proceso aleatorio futuro

# F2_tratamiento_csv.py
# Validación, limpieza y tratamiento inicial de las columnas básicas del dataset.



def validar_datos(df):
    log("===== VALIDACIÓN, COERCIÓN Y REGLAS DE DOMINIO =====")  ##comentario inicio de los controles de calidad de datos

    columnas_numericas = [
        'SKU', 'UNIDADES', 'PORCENTAJE DESCUENTO',
        'MONTO APLICADO', 'BOLETA', 'LOCAL', 'GENERO'
    ]  ##comentario columnas que deben ser numéricas

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = dd.to_numeric(df[col], errors='coerce')  ##comentario coerciona las columnas a numérico conservando NaN
    log("Conversión numérica perezosa aplicada")  ##comentario registro de que se aplicó la conversión

    if 'PORCENTAJE DESCUENTO' in df.columns:
        fuera_rango = (df['PORCENTAJE DESCUENTO'] < 0) | (df['PORCENTAJE DESCUENTO'] > 1)  ##comentario definimos valores no válidos
        n_fuera = fuera_rango.sum().compute()  ##comentario contamos valores fuera de rango
        log(f"PORCENTAJE DESCUENTO: fuera de rango detectados = {n_fuera}")  ##comentario registro del conteo de valores inválidos
        df['PORCENTAJE DESCUENTO'] = df['PORCENTAJE DESCUENTO'].where(~fuera_rango, np.nan)  ##comentario marca los valores inválidos como NaN

    if 'GENERO' in df.columns:
        fuera_rango = ~df['GENERO'].isin([1, 2])  ##comentario solo se aceptan códigos 1 o 2
        n_fuera = fuera_rango.sum().compute()  ##comentario contar errores en género
        log(f"GENERO: valores inválidos detectados = {n_fuera}")  ##comentario registrar valores inválidos
        df['GENERO'] = df['GENERO'].where(~fuera_rango, np.nan)  ##comentario convertir valores inválidos a NaN

    if 'CODIGO CLIENTE' in df.columns:
        df['CODIGO CLIENTE'] = df['CODIGO CLIENTE'].astype(str).str.lower()  ##comentario normalizar cadenas UUID a minúsculas
        es_uuid_valido = df['CODIGO CLIENTE'].str.match(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        )  ##comentario validar el patrón UUID
        n_invalidos_uuid = (~es_uuid_valido).sum().compute()  ##comentario contar UUID inválidos
        log(f"CODIGO CLIENTE: UUID inválidos detectados = {n_invalidos_uuid}")  ##comentario registrar errores de UUID
        df['CODIGO CLIENTE'] = df['CODIGO CLIENTE'].where(es_uuid_valido, np.nan)  ##comentario invalidar UUIDs incorrectos

    columnas_fecha = []  ##comentario lista de columnas de fecha presentes en el dataset
    for col in ['FECHA', 'FECHA NACIMIENTO']:
        if col in df.columns:
            columnas_fecha.append(col)  ##comentario añadir columna de fecha válida a la lista

    for col in columnas_fecha:
        df[col] = dd.to_datetime(df[col], errors='coerce')  ##comentario convertir columnas a tipo fecha
    if columnas_fecha:
        log("Conversión de fechas aplicada")  ##comentario indicar que la conversión de fechas se ejecutó

    log("VALIDACIÓN COMPLETADA (TIPO + FORMATO + RANGO)")  ##comentario marca el final del proceso de validación
    return df  ##comentario devuelve el DataFrame validado


def _log_nulos_por_grupo(df, columnas, titulo):
    presentes = []  ##comentario columnas que existen en el DataFrame
    for col in columnas:
        if col in df.columns:
            presentes.append(col)  ##comentario guardar solo las columnas presentes
    if not presentes:
        log(f"{titulo}: no están presentes las columnas esperadas.")  ##comentario informar si no hay columnas para evaluar
        return

    total_registros = df.shape[0].compute()  ##comentario obtener el tamaño total del dataset
    nulos_por_col = df[presentes].isna().sum().compute()  ##comentario calcular nulos por columna

    log(f"\n--- {titulo} ---")  ##comentario encabezado del reporte
    for col, nulos in nulos_por_col.items():
        porcentaje = (nulos / total_registros) * 100 if total_registros else 0  ##comentario calcular porcentaje de nulos
        log(f"\nColumna: {col}")  ##comentario imprimir el nombre de la columna
        log(f"Nulos: {nulos}")  ##comentario imprimir cantidad de nulos
        log(f"% Nulos: {porcentaje:.2f}%")  ##comentario imprimir porcentaje de nulos


def busca_nulos(df):
    log("===== ANÁLISIS DE VALORES FALTANTES =====")  ##comentario inicio del análisis de ausencias

    _log_nulos_por_grupo(df, [
        'SKU', 'UNIDADES', 'PORCENTAJE DESCUENTO',
        'MONTO APLICADO', 'BOLETA', 'LOCAL', 'GENERO'
    ], "MÉTRICAS NUMÉRICAS")  ##comentario evalúa nulos en columnas numéricas

    _log_nulos_por_grupo(df, ['FECHA', 'FECHA NACIMIENTO'], "MÉTRICAS FECHAS")  ##comentario evalúa nulos en fechas

    _log_nulos_por_grupo(df, [
        'CANAL', 'PRODUCTO', 'NOMBRES',
        'APELLIDOS', 'RUN CLIENTE', 'CODIGO CLIENTE'
    ], "MÉTRICAS TEXTO")  ##comentario evalúa nulos en columnas de texto

    log("\nANÁLISIS DE NULOS COMPLETADO")  ##comentario fin del análisis de ausencias


def analizar_mecanismo_missingness(df):
    log("===== ANÁLISIS DEL MECANISMO DE AUSENCIA (MCAR) =====")  ##comentario inicio de la validación MCAR aproximada
    np.random.seed(SEMILLA_ALEATORIEDAD)  ##comentario fija la semilla para reproducibilidad

    columnas_objetivo = [
        'PORCENTAJE DESCUENTO', 'FECHA NACIMIENTO', 'MONTO APLICADO',
        'UNIDADES', 'CODIGO CLIENTE'
    ]  ##comentario columnas clave para analizar missingness
    columnas_numericas = [
        'MONTO APLICADO', 'UNIDADES', 'SKU', 'PORCENTAJE DESCUENTO'
    ]  ##comentario columnas numéricas relevantes para el análisis

    total_registros = df.shape[0].compute()  ##comentario número total de filas
    for col in columnas_objetivo:
        if col not in df.columns:
            continue  ##comentario omite columnas no presentes

        nulos = df[col].isna().sum().compute()  ##comentario cuenta los valores faltantes
        if nulos == 0:
            continue  ##comentario no hay missingness para analizar

        porcentaje = (nulos / total_registros) * 100 if total_registros else 0  ##comentario calcula el porcentaje de ausencias
        log(f"Columna {col}: {porcentaje:.2f}% nulos.")  ##comentario informar porcentaje de datos faltantes

        for var in columnas_numericas:
            if var not in df.columns or var == col:
                continue  ##comentario omite la misma columna o columnas no presentes

            promedio_con = df[var].where(df[col].isna()).mean().compute()  ##comentario promedio cuando falta el valor objetivo
            promedio_sin = df[var].where(~df[col].isna()).mean().compute()  ##comentario promedio cuando no falta el valor objetivo
            if promedio_sin != 0 and not np.isnan(promedio_con) and not np.isnan(promedio_sin):
                diferencia = abs(promedio_con - promedio_sin) / abs(promedio_sin)  ##comentario calcula la diferencia relativa de medias
                if diferencia > 0.1:
                    log(f"  * Señal de no MCAR en {col} respecto a {var}: diferencia relativa {diferencia:.2f}")  ##comentario sugiere mecanismo no MCAR
                else:
                    log(f"  * Sin evidencia fuerte de no MCAR entre {col} y {var}.")  ##comentario sugiere posible MCAR

    log("Análisis del mecanismo de missingness completado.")  ##comentario fin de la evaluación MCAR
    return df  ##comentario devuelve el DataFrame sin modificar


def detectar_outliers(df):
    log("===== DETECCIÓN DE OUTLIERS CON IQR =====")  ##comentario inicio de la detección de valores extremos
    columnas_numericas = [
        'SKU', 'UNIDADES', 'PORCENTAJE DESCUENTO',
        'MONTO APLICADO', 'BOLETA', 'LOCAL', 'GENERO'
    ]  ##comentario columnas numéricas a evaluar

    for col in columnas_numericas:
        if col not in df.columns:
            continue  ##comentario omite columnas no presentes

        cuartiles = df[col].quantile([0.25, 0.75]).compute()  ##comentario calcula los cuartiles de la columna
        if cuartiles.isna().any():
            log(f"  * {col}: no se pudo calcular IQR por falta de datos.")  ##comentario si hay datos insuficientes
            continue

        q1 = cuartiles.loc[0.25]  ##comentario primer cuartil
        q3 = cuartiles.loc[0.75]  ##comentario tercer cuartil
        iqr = q3 - q1  ##comentario rango intercuartil (IQR)
        if iqr == 0 or np.isnan(iqr):
            log(f"  * {col}: IQR nulo, no se detectan outliers robustos.")  ##comentario evita división por cero
            continue

        limite_inferior = q1 - 1.5 * iqr  ##comentario límite inferior para outliers
        limite_superior = q3 + 1.5 * iqr  ##comentario límite superior para outliers
        outliers = (df[col] < limite_inferior) | (df[col] > limite_superior)  ##comentario identifica filas extremas
        n_outliers = outliers.sum().compute()  ##comentario cuenta los outliers detectados
        porcentaje_outliers = (n_outliers / df.shape[0].compute()) * 100 if df.shape[0].compute() else 0  ##comentario porcentaje de outliers

        log(f"  * {col}: {n_outliers} outliers detectados ({porcentaje_outliers:.2f}%).")  ##comentario reporta la detección de valores extremos
        df[col] = df[col].where(~outliers, np.nan)  ##comentario marca los outliers como ausentes para tratarlos luego

    log("Detección de outliers completada.")  ##comentario fin de la detección
    return df  ##comentario devuelve el DataFrame con outliers marcados


def _mediana_aproximada(df, col):
    # Dask no implementa median() exacta en todos los casos; usamos un método aproximado.
    try:
        return df[col].median_approximate().compute()
    except AttributeError:
        return df[col].quantile(0.5).compute()


def limpiar_nulos(df):
    log("===== PROCESANDO ESTRATEGIA DE LIMPIEZA ESTRATIFICADA DE VARIABLES BÁSICAS =====")  ##comentario inicio de la limpieza de nulos básicos
    np.random.seed(SEMILLA_ALEATORIEDAD)  ##comentario fija la semilla para garantizar reproducibilidad

    df = detectar_outliers(df)  ##comentario identifica y marca valores extremos antes de limpiar nulos
    df = analizar_mecanismo_missingness(df)  ##comentario examina si los nulos parecen MCAR

    columnas_a_limpiar = [
        'SKU', 'UNIDADES', 'PORCENTAJE DESCUENTO',
        'MONTO APLICADO', 'BOLETA', 'LOCAL', 'GENERO'
    ]  ##comentario columnas que se limpiarán según umbrales

    presentes = []  ##comentario columnas disponibles para limpieza
    for col in columnas_a_limpiar:
        if col in df.columns:
            presentes.append(col)  ##comentario agrega la columna si existe
    if not presentes:
        log("No hay columnas básicas disponibles para limpieza.")  ##comentario informa si no hay columnas para procesar
        return df

    total_registros = df.shape[0].compute()  ##comentario tamaño del dataset actual
    nulos_por_col = df[presentes].isna().sum().compute()  ##comentario estima los nulos por columna

    for col, nulos in nulos_por_col.items():
        porcentaje = (nulos / total_registros) * 100 if total_registros else 0  ##comentario porcentaje de nulos en la columna
        if nulos == 0:
            continue  ##comentario omite columnas sin nulos

        if porcentaje <= 5:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Estrategia: BORRADO (Zona Segura).")  ##comentario estrategia de eliminación para pocos nulos
            df = df.dropna(subset=[col])  ##comentario elimina las filas que faltan en la columna
        elif porcentaje <= 20:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Estrategia: IMPUTACIÓN (Zona Media).")  ##comentario estrategia de imputación para porcentaje moderado
            valor_imputacion = _mediana_aproximada(df, col)  ##comentario calcula mediana aproximada compatible con Dask
            df[col] = df[col].fillna(valor_imputacion)  ##comentario rellena los nulos con la mediana
        else:
            log(f"Columna {col}: {porcentaje:.2f}% nulos. Estrategia: DESCARTAR VARIABLE (Zona Peligro).")  ##comentario estrategia de descarte para muchos nulos
            df = df.drop(columns=[col])  ##comentario elimina la columna completa

    log("Limpieza de nulos finalizada.")  ##comentario fin de la limpieza de nulos
    return df  ##comentario devuelve el DataFrame resultante
