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

# 3. Auditoría de nulos antes de limpieza.
busca_nulos(df)

# 4. Limpieza de variables básicas.
df = limpiar_nulos(df)
log(f"Después de limpieza básica hay {df.shape[0].compute()} registros.")

log("Pipeline completado. Estado final del dataset listo para el análisis.")