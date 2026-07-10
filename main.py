import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
from utiles.logger import log
from src.T1_preprocesamiento.F1_carga_csv import cargar_datos_paralelo
from src.T1_preprocesamiento.F2_tratamiento_csv import (
    validar_datos,
    busca_nulos,
    limpiar_nulos,
    detectar_outliers,
    analizar_mecanismo_missingness
)
from src.T1_preprocesamiento.F3_transformacion_datos_csv import crear_variables_deducidas
from src.T1_preprocesamiento.F4_tratamiento_variables_deducidas import (
    validar_deducidas,
    busca_nulos_deducidas,
    limpiar_nulos_deducidos,
    normalizar_variables
)

# Módulos de análisis (solo usados en ETAPA 9 — EDA)
from src.T2_analisis.F1_analisis_basico import analisis_basico
from src.T2_analisis.F2_analisis_deducidas import analisis_deducidas
from src.T2_analisis.F3_normalidad import normalidad
from src.T2_analisis.F4_correlaciones import correlaciones
from src.T2_analisis.F5_asociacion import asociacion
from src.T2_analisis.F6_anova import anova
import pandas as pd
import time
from utiles.semilla import set_global_seed

# main.py
# Ejecución secuencial del pipeline de preprocesamiento.


if len(sys.argv) < 2:
    log("Uso: python3 main.py data/ventas_completas.csv")
    log("Ejecuta este comando desde la carpeta que contiene main.py.")
    sys.exit(1)

archivo_csv = sys.argv[1]
if not os.path.exists(archivo_csv):
    log(f"Error: El archivo '{archivo_csv}' no existe.")
    log("Ejemplo de ejecución correcta: python3 main.py data/ventas_completas.csv")
    log("Asegúrate de ejecutar el comando desde la carpeta que contiene main.py.")
    sys.exit(1)

seed = set_global_seed()
log(f"Inicio de pipeline para archivo: {archivo_csv}")
log(f"CPYD_SEED={seed}")

# --------------------------------------------------
# ETAPA 1 — CARGA DEL CSV
# --------------------------------------------------
start = time.time()
df = cargar_datos_paralelo(archivo_csv)
n_partitions = getattr(df, 'npartitions', 'desconocido')
try:
    n_registros = df.shape[0].compute()
except Exception:
    n_registros = 'desconocido'
n_columnas = len(df.columns)
log(f"ETAPA 1: Registros={n_registros}, Columnas={n_columnas}, Particiones={n_partitions}")
log(f"Tiempo ETAPA 1: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 2 — VALIDACIÓN DEL DATASET
# --------------------------------------------------
start = time.time()
df = validar_datos(df)
log(f"ETAPA 2 completada. Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 3 — DIAGNÓSTICO DEL DATASET ORIGINAL
# --------------------------------------------------
start = time.time()
os.makedirs('output/diagnostico', exist_ok=True)
try:
    total = df.shape[0].compute()
    cols = len(df.columns)
    dtypes = pd.DataFrame({'variable': df.columns, 'dtype': [str(df[c].dtype) for c in df.columns]})
    dtypes.to_csv('output/diagnostico/tipos_columnas.csv', index=False)
    # nulos y porcentajes
    nulos = df.isna().sum().compute()
    nulos_df = pd.DataFrame({'variable': nulos.index, 'nulos': nulos.values})
    nulos_df['porcentaje'] = (nulos_df['nulos'] / total) * 100 if total else 0
    nulos_df.to_csv('output/diagnostico/nulos.csv', index=False)
    # duplicados
    try:
        dup = int(df.duplicated().sum().compute())
    except Exception:
        dup = -1
    pd.DataFrame({'duplicados': [dup]}).to_csv('output/diagnostico/duplicados.csv', index=False)
    # estadísticas simples
    try:
        desc = df.describe().compute().transpose()
        desc.to_csv('output/diagnostico/estadisticas_simples.csv')
    except Exception:
        log('No se pudieron calcular todas las estadísticas descriptivas.')
except Exception as e:
    log(f"Error en diagnóstico: {e}")
log(f"ETAPA 3 completada. Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 4 — PREPROCESAMIENTO (variables básicas)
# --------------------------------------------------
start = time.time()
log("ETAPA 4: Preprocesamiento de variables básicas")
busca_nulos(df)
df = detectar_outliers(df)
df = analizar_mecanismo_missingness(df)
df = limpiar_nulos(df)
log(f"ETAPA 4 completada. Registros actuales: {df.shape[0].compute()} Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 5 — VARIABLES DEDUCIDAS
# --------------------------------------------------
start = time.time()
df = crear_variables_deducidas(df)
df = validar_deducidas(df)
os.makedirs('output/preprocesamiento/deducidas', exist_ok=True)
try:
    deducidas_cols = [
        'MONTO_POR_UNIDAD', 'EDAD', 'HORA_TRANSACCION',
        'FRECUENCIA_COMPRA', 'RECENCIA', 'MONTO_BRUTO',
        'ES_FIN_DE_SEMANA', 'SEGMENTO_MONTO'
    ]
    presentes = [col for col in deducidas_cols if col in df.columns]
    pd.DataFrame({
        'variable': presentes,
        'dtype': [str(df[col].dtype) for col in presentes]
    }).to_csv('output/preprocesamiento/deducidas/variables_deducidas.csv', index=False)
    log('Resumen de variables deducidas guardado en output/preprocesamiento/deducidas/variables_deducidas.csv')
except Exception as e:
    log(f"No se pudo guardar el resumen de variables deducidas: {e}")
log(f"ETAPA 5 completada. Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 6 — LIMPIEZA DE VARIABLES DEDUCIDAS
# --------------------------------------------------
start = time.time()
busca_nulos_deducidas(df)
df = limpiar_nulos_deducidos(df)
log(f"ETAPA 6 completada. Registros actuales: {df.shape[0].compute()} Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 7 — NORMALIZACIÓN
# --------------------------------------------------
start = time.time()
df = normalizar_variables(df)
os.makedirs('output/preprocesamiento/modelos', exist_ok=True)
log(f"ETAPA 7 completada. Tiempo: {time.time() - start:.2f}s")

# --------------------------------------------------
# ETAPA 8 — DATASET FINAL
# --------------------------------------------------
total_final = df.shape[0].compute()
variables_final = len(df.columns)
log("========================================")
log("PREPROCESAMIENTO COMPLETADO")
log("Dataset limpio y preparado.")
log(f"Registros finales: {total_final}")
log(f"Variables finales: {variables_final}")
log("========================================")

# --------------------------------------------------
# ETAPA 9 — ANÁLISIS EXPLORATORIO ESTADÍSTICO (EDA)
# --------------------------------------------------
start = time.time()

log("ETAPA 9: Ejecutando EDA (estadísticas completas, distribuciones, normalidad, correlaciones y asociaciones)")

# Estadística descriptiva completa y gráficos descriptivos básicos
analisis_basico(df, 'EDA_basicas')

# Análisis avanzado para variables deducidas
analisis_deducidas(df, 'EDA_deducidas')

# Normalidad, correlaciones y asociaciones
try:
    normalidad(df, 'EDA_general')
except Exception as e:
    log(f"Normalidad (EDA) falló: {e}")
try:
    correlaciones(df, 'EDA_general')
except Exception as e:
    log(f"Correlaciones (EDA) falló: {e}")
try:
    asociacion(df, 'EDA_general')
except Exception as e:
    log(f"Asociación (EDA) falló: {e}")
try:
    anova(df, 'EDA_general')
except Exception as e:
    log(f"ANOVA (EDA) falló: {e}")
log(f"ETAPA 9 completada. Tiempo total EDA: {time.time() - start:.2f}s")

log("Pipeline completo. Todos los outputs están en la carpeta output/.")