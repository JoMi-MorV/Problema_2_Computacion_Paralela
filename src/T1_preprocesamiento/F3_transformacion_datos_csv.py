import dask.dataframe as dd  ##comentario importa Dask DataFrame para el pipeline de transformación
import numpy as np  ##comentario importa Numpy para operaciones numéricas

# F3_transformacion_datos_csv.py
# Generación de variables deducidas a partir de los datos básicos.


def crear_variables_deducidas(df):
    """Genera variables derivadas como monto por unidad, edad, hora, frecuencia, recencia y segmento."""
    df = df.reset_index(drop=True)  ##comentario restablece el índice para que sea único y continuo

    if 'MONTO APLICADO' in df.columns and 'UNIDADES' in df.columns:
        df['MONTO_POR_UNIDAD'] = df['MONTO APLICADO'] / df['UNIDADES']  ##comentario calcula el monto por unidad vendida

    if 'FECHA NACIMIENTO' in df.columns:
        df['EDAD'] = 2026 - df['FECHA NACIMIENTO'].dt.year  ##comentario calcula la edad a partir de la fecha de nacimiento

    if 'FECHA' in df.columns:
        df['HORA_TRANSACCION'] = df['FECHA'].dt.hour  ##comentario extrae la hora de la transacción

    if 'CODIGO CLIENTE' in df.columns and 'BOLETA' in df.columns:
        frecuencia = (
            df.groupby('CODIGO CLIENTE')['BOLETA']
            .count()
            .reset_index()
        )  ##comentario agrupa por cliente y cuenta transacciones
        frecuencia.columns = ['CODIGO CLIENTE', 'FRECUENCIA_COMPRA']  ##comentario renombra la columna de frecuencia
        df = df.merge(frecuencia, on='CODIGO CLIENTE', how='left')  ##comentario une la frecuencia de compra al dataset

    if 'FECHA' in df.columns:
        fecha_maxima = df['FECHA'].max().compute()  ##comentario obtiene la última fecha de transacción
        df['RECENCIA'] = (fecha_maxima - df['FECHA']).dt.days  ##comentario calcula recencia en días
        df['ES_FIN_DE_SEMANA'] = df['FECHA'].dt.dayofweek >= 5  ##comentario determina si la transacción fue fin de semana

    if 'MONTO APLICADO' in df.columns:
        quantiles = df['MONTO APLICADO'].quantile([0.33, 0.66]).compute()  ##comentario calcula cuantiles para segmentación
        p33 = quantiles.loc[0.33]  ##comentario percentil 33
        p66 = quantiles.loc[0.66]  ##comentario percentil 66

        df['SEGMENTO_MONTO'] = 'Bajo'  ##comentario valor por defecto del segmento
        df['SEGMENTO_MONTO'] = df['SEGMENTO_MONTO'].mask(df['MONTO APLICADO'] > p33, 'Medio')  ##comentario segmenta montos intermedios
        df['SEGMENTO_MONTO'] = df['SEGMENTO_MONTO'].mask(df['MONTO APLICADO'] > p66, 'Alto')  ##comentario segmenta los montos altos

    if 'MONTO APLICADO' in df.columns and 'PORCENTAJE DESCUENTO' in df.columns:
        df['MONTO_BRUTO'] = df['MONTO APLICADO'] / (1 - df['PORCENTAJE DESCUENTO'])  ##comentario calcula monto bruto antes del descuento

    return df  ##comentario devuelve el DataFrame con las variables nuevas
