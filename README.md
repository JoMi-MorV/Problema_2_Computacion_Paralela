# Pipeline de Preprocesamiento con Dask

Proyecto de preprocesamiento orientado a cargar, validar y limpiar datos de ventas utilizando Dask para reducir la presión sobre la memoria RAM.

---

## Instalar

Para preparar el entorno e instalar las dependencias necesarias en Ubuntu 24.04, ejecuta:

```bash
sudo apt update
sudo apt install python3-pip
sudo apt install python3-matplotlib
pip install --upgrade dask[dataframe] pandas --break-system-packages
```

## Estructura
proyecto/
├─ data/
│  └── ventas_completas.csv
│
├── src/
│  ├── T1_preprocesamiento/
│  │    ├── F1_carga_csv.py
│  │    ├── F2_tratamiento_csv.py
│  │    ├── F3_transformacion_datos_csv.py
│  │    └── F4_tratamiento_variables_deducidas.py
│ 
├── utiles/
│  └── logger.py
│
├── main.py
└── README.md

## Arquitectura del Sistema

### 1. `F1_carga_csv.py` (Módulo de Ingesta)

Este módulo constituye la puerta de entrada del conjunto de datos. Se encarga de cargar el CSV utilizando **Dask DataFrame**, que construye un grafo de tareas en lugar de materializar todo el dataset en memoria. La lectura es perezosa y se divide en particiones lógicas mediante `blocksize`, lo que mejora la eficiencia del uso de RAM.

#### Responsabilidades

- Leer archivos CSV grandes con Dask.
- Mantener la carga mínima en memoria hasta que se requieran resultados.
- Construir un DataFrame distribuido con particiones independientes.
- Detectar y reportar errores de lectura.

### 2. `F2_tratamiento_csv.py` (Validación y limpieza inicial)

Este módulo aplica transformación y validación a las columnas básicas del dataset. Mantiene el DataFrame en Dask para conservar operaciones perezosas y reducir el uso de memoria.

#### Responsabilidades

- Coercer columnas numéricas a tipos adecuados.
- Validar dominio de `PORCENTAJE DESCUENTO` y `GENERO`.
- Validar el formato UUID de `CODIGO CLIENTE`.
- Convertir fechas y validar formatos.
- Analizar valores nulos en grupos de columnas.
- Limpiar nulos según umbrales: eliminación, imputación o descarte de columna.

### 3. `F3_transformacion_datos_csv.py` (Variables deducidas)

Este módulo genera nuevas variables derivadas del dataset original, preservando el paralelismo de Dask.

#### Responsabilidades

- Calcular `MONTO_POR_UNIDAD`.
- Extraer `EDAD` y `HORA_TRANSACCION` de fechas.
- Calcular `FRECUENCIA_COMPRA` con `groupby()` y `merge()`.
- Calcular `RECENCIA`, `MONTO_BRUTO` y `ES_FIN_DE_SEMANA`.
- Segmentar `MONTO APLICADO` en `SEGMENTO_MONTO`.

### 4. `F4_tratamiento_variables_deducidas.py` (Validación, limpieza y normalización de variables derivadas)

Este módulo valida, limpia y normaliza las columnas generadas en `F3`, manteniendo el uso eficiente de memoria con Dask.

#### Responsabilidades

- Coercer variables numéricas derivadas.
- Aplicar reglas de dominio para `EDAD`, `MONTO_BRUTO` y `FRECUENCIA_COMPRA`.
- Validar valores booleanos y categorías esperadas.
- Medir y limpiar nulos en las columnas derivadas.
- Normalizar variables derivadas relevantes para análisis posteriores.

#### Normalización

- Se utiliza una estandarización tipo `StandardScaler`.
- Cada variable derivada se transforma con `(valor - media) / desviación_estándar`.
- Se documentan los parámetros de normalización: media y desviación estándar.

### 5. `main.py` (Orquestador del pipeline)

Este script ahora ejecuta el flujo completo con análisis antes y después de cada limpieza:

1. Cargar el CSV con Dask.
2. Validar los datos básicos.
3. Auditoría de nulos básicos.
4. Limpieza de variables básicas.
5. Generar variables deducidas.
6. Validar variables deducidas.
7. Auditoría de nulos en variables deducidas.
8. Limpieza de variables deducidas.

#### Responsabilidades

- Validar la existencia del archivo de entrada.
- Ejecutar cada fase del pipeline secuencialmente.
- Guardar resultados de análisis y gráficos en carpetas de salida.
- Facilitar un flujo reproducible de inspección, limpieza y reanálisis.