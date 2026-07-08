import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import sys
import os
from utiles.logger import log
from src.T1_preprocesamiento.F1_carga_csv import cargar_datos_paralelo
from src.T1_preprocesamiento.F2_tratamiento_csv import (
    validar_datos,
    busca_nulos,
    limpiar_nulos
)
from src.T1_preprocesamiento.F3_transformacion_datos_csv import crear_variables_deducidas
from src.T1_preprocesamiento.F4_tratamiento_variables_deducidas import (
    validar_deducidas,
    busca_nulos_deducidas,
    limpiar_nulos_deducidos,
    normalizar_variables
)

# main.py
# Ejecución secuencial del pipeline de preprocesamiento.

if len(sys.argv) < 2:
    log("Uso: python3 main.py <data/ventas_completas.csv>")
    sys.exit(1)

archivo_csv = sys.argv[1]
if not os.path.exists(archivo_csv):
    log(f"Error: El archivo '{archivo_csv}' no existe.")
    sys.exit(1)

log(f"El archivo '{archivo_csv}' existe.")

# 1. Cargar el CSV con Dask para preservar memoria.
df = cargar_datos_paralelo(archivo_csv)
log(f"En '{archivo_csv}' hay {df.shape[0].compute()} registros.")

# 2. Validación y coerción de tipos básicas.
df = validar_datos(df)

# 4. Auditoría de nulos antes de limpieza.
busca_nulos(df)

# 5. Limpieza de variables básicas.
df = limpiar_nulos(df)
log(f"Después de limpieza básica hay {df.shape[0].compute()} registros.")

# 6. Ingeniería de características: variables deducidas.
df = crear_variables_deducidas(df)
log("Variables deducidas creadas exitosamente.")

# 7. Validación de las variables derivadas.
df = validar_deducidas(df)

# 8. Auditoría de nulos en variables derivadas.
busca_nulos_deducidas(df)

# 9. Limpieza de variables derivadas.
df = limpiar_nulos_deducidos(df)
log(f"Después de limpieza de variables deducidas hay {df.shape[0].compute()} registros.")

log("Pipeline completado. Estado final del dataset listo para el análisis.")