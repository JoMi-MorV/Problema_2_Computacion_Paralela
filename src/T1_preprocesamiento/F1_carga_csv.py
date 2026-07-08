import dask.dataframe as dd  ##comentario importa Dask DataFrame para la carga paralela
import sys  ##comentario importa sys para terminar el programa si ocurre un error crítico
from utiles.logger import log  ##comentario importa la función de logging del proyecto

# F1_carga_csv.py
# Carga de datos con Dask DataFrame para gestionar archivos CSV grandes sin
# cargar todo el dataset en memoria RAM.


def cargar_datos_paralelo(ruta_archivo):
    log("===== CARGA DEL CSV CON DASK =====")  ##comentario indica el inicio del proceso de carga
    try:
        print("Inicializando lectura perezosa con Dask...")  ##comentario muestra mensaje en consola
        df_dask = dd.read_csv(
            ruta_archivo,  ##comentario ruta del archivo CSV de entrada
            sep=';',  ##comentario separador de campos del CSV
            blocksize="64MB",  ##comentario define el tamaño de partición para la lectura
            assume_missing=True  ##comentario asume valores faltantes en columnas numéricas
        )
        print(f"¡Estructura virtual mapeada exitosamente en {df_dask.npartitions} particiones!")  ##comentario informa proveedor del número de particiones
        return df_dask  ##comentario devuelve el DataFrame diferido de Dask
    except Exception as e:
        log(f"Ocurrió un error al leer con Dask: {e}")  ##comentario registra el error en el log
        sys.exit(1)  ##comentario detiene la ejecución en caso de falla de carga
 