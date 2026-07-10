# Pipeline de procesamiento y EDA para ventas

Este proyecto orquesta la carga, validación, limpieza y análisis exploratorio de un archivo CSV de ventas usando Dask, pandas y matplotlib. La idea central es mantener el uso de memoria controlado, especialmente durante la etapa de EDA, evitando materializar datasets completos en memoria.

## El archivo ventas_completas.csv debe ser cargado manualmente a la carpeta data (cuando se descarga el código no viene con el archivo.csv)

## Qué hace el pipeline

El flujo actual incluye estas etapas:

1. Carga del archivo CSV con Dask.
2. Validación y coerción de tipos de las columnas base.
3. Diagnóstico inicial de tipos, nulos y duplicados.
4. Limpieza y tratamiento de variables básicas.
5. Generación de variables derivadas.
6. Validación y limpieza de las variables derivadas.
7. Normalización de variables relevantes para análisis posteriores.
8. EDA con estadísticas, gráficos y pruebas de asociación.

## Estructura del proyecto

```text
proyecto/
├── data/
│   ├── ventas_completas.csv
│   └── ventas_prueba.csv
├── output/
│   ├── analysis/
│   ├── diagnostico/
│   ├── graficos/
│   └── preprocesamiento/
├── src/
│   ├── T1_preprocesamiento/
│   │   ├── F1_carga_csv.py
│   │   ├── F2_tratamiento_csv.py
│   │   ├── F3_transformacion_datos_csv.py
│   │   └── F4_tratamiento_variables_deducidas.py
│   ├── T2_analisis/
│   │   ├── F1_analisis_basico.py
│   │   ├── F2_analisis_deducidas.py
│   │   ├── F3_normalidad.py
│   │   ├── F4_correlaciones.py
│   │   ├── F5_asociacion.py
│   │   └── F6_anova.py
│   └── T3_inferencia/
├── utiles/
│   ├── eda_memoria.py
│   ├── logger.py
│   └── semilla.py
├── main.py
└── README.md
```

## Requisitos

Se recomienda usar Python 3.10+ y las siguientes dependencias:

```bash
pip install dask[dataframe] pandas numpy matplotlib scipy statsmodels
```

En entornos Ubuntu, también puede ser necesario instalar:

```bash
sudo apt update
sudo apt install python3-pip python3-matplotlib
```

## Ejecución

Ejecuta el pipeline indicando la ruta al CSV de entrada:

```bash
python3 main.py data/ventas_completas.csv
```

El pipeline crea los resultados en la carpeta output/.

## Módulos principales

### main.py
Orquesta todo el flujo. Inicia la carga, aplica cada etapa de preprocesamiento y luego ejecuta el EDA.

### src/T1_preprocesamiento/F1_carga_csv.py
Carga el CSV de forma perezosa con Dask para evitar cargar el dataset completo en memoria.

### src/T1_preprocesamiento/F2_tratamiento_csv.py
Valida tipos, domina rangos, detecta UUIDs inválidos, convierte fechas y analiza nulos/outliers en columnas básicas.

### src/T1_preprocesamiento/F3_transformacion_datos_csv.py
Genera variables derivadas como monto por unidad, edad, hora, frecuencia, recencia, fin de semana y segmento de monto.

### src/T1_preprocesamiento/F4_tratamiento_variables_deducidas.py
Valida, limpia y normaliza las variables derivadas, además de guardar los parámetros de normalización en JSON.

### src/T2_analisis/F1_analisis_basico.py
Produce estadísticas descriptivas, tablas de frecuencia y gráficos para variables básicas.

### src/T2_analisis/F2_analisis_deducidas.py
Produce estadísticas y gráficos para variables deducidas.

### src/T2_analisis/F3_normalidad.py
Ejecuta pruebas de normalidad sobre muestras acotadas y guarda QQ plots.

### src/T2_analisis/F4_correlaciones.py
Calcula correlaciones numéricas y guarda un heatmap de correlación.

### src/T2_analisis/F5_asociacion.py
Calcula chi-cuadrado y Cramér's V para asociaciones categóricas y genera gráficos de contingencia simples.

### src/T2_analisis/F6_anova.py
Ejecuta ANOVA sobre montos por factores categóricos y, cuando está disponible, Tukey.

## Salidas generadas

El pipeline deja resultados en estas rutas:

- output/diagnostico/: tipos de columna, nulos, duplicados y estadísticas básicas.
- output/preprocesamiento/deducidas/: resumen de variables derivadas.
- output/preprocesamiento/modelos/parametros_normalizacion.json: parámetros usados durante la normalización.
- output/analysis/: CSV con estadísticas, correlaciones, normalidad, asociación y ANOVA.
- output/graficos/: gráficos de EDA, normalidad, correlaciones, asociación y ANOVA.

## Notas de memoria

Para evitar que el EDA consuma demasiada RAM, el proyecto trabaja sobre muestras acotadas en memoria mediante [utiles/eda_memoria.py](utiles/eda_memoria.py). Esto permite mantener el análisis útil incluso con archivos grandes.

