# Pipeline de procesamiento y EDA para ventas

Este proyecto orquesta la carga, validación, limpieza y análisis exploratorio de un archivo CSV de ventas usando Dask, pandas y matplotlib. La idea central es mantener el uso de memoria controlado, especialmente durante la etapa de EDA, evitando materializar datasets completos en memoria.

## Requisitos previos

El archivo `data/ventas_completas.csv` debe estar presente en la carpeta `data/` antes de ejecutar el pipeline. En este repositorio ya se incluye ese archivo.

## Qué hace el pipeline

El flujo actual incluye estas etapas:

1. Carga del archivo CSV con Dask.
2. Validación y coerción de tipos de las columnas base.
3. Diagnóstico inicial de tipos, nulos y duplicados.
4. Limpieza y tratamiento de variables básicas.
5. Generación de variables derivadas.
6. Validación y limpieza de las variables derivadas.
7. Normalización de variables derivadas.
8. EDA con estadísticas, gráficos y pruebas de asociación.
9. Modelado descriptivo mediante clustering K-Means para segmentar clientes por perfil de consumo.

## Estructura del proyecto

```text
Problema_2_Computacion_Paralela-main/
├── data/
│   └── ventas_completas.csv
├── output/                 # creado automáticamente por el pipeline
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
│   ├── T3_inferencia/
│   │   └── F1_modelado_clustering.py
├── utiles/
│   ├── eda_memoria.py
│   ├── logger.py
│   └── semilla.py
├── main.py
└── README.md
```

## Dependencias

Se recomienda usar Python 3.10+ y las siguientes dependencias:
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install python3-matplotlib
pip install --upgrade dask[dataframe] pandas --break-system-packages
pip install scikit-learn --break-system-packages
```

## Ejecución

Ejecuta el pipeline desde la carpeta que contiene `main.py` e indicando la ruta al CSV de entrada:

```bash
cd /ruta/a/Problema_2_Computacion_Paralela-main
python3 main.py data/ventas_completas.csv
```

Si ves un error como:

```bash
python3: can't open file '/ruta/incorrecta/main.py': [Errno 2] No such file or directory
```

verifica tu carpeta actual con `pwd` y vuelve a ejecutar desde el directorio correcto.

El pipeline crea los resultados en la carpeta `output/`.

## Módulos principales

### main.py
Orquesta todo el flujo. Inicia la carga, aplica cada etapa de preprocesamiento y luego ejecuta el EDA.

### src/T1_preprocesamiento/F1_carga_csv.py
Carga el CSV de forma perezosa con Dask para evitar cargar el dataset completo en memoria.

### src/T1_preprocesamiento/F2_tratamiento_csv.py
Valida tipos, detecta nulos y outliers, analiza mecanismos de missingness y limpia datos básicos.

### src/T1_preprocesamiento/F3_transformacion_datos_csv.py
Genera variables derivadas como monto por unidad, edad, hora, frecuencia, recencia, fin de semana y segmento de monto.

### src/T1_preprocesamiento/F4_tratamiento_variables_deducidas.py
Valida, limpia y normaliza las variables derivadas.

### src/T2_analisis/F1_analisis_basico.py
Produce estadísticas descriptivas, tablas de frecuencia y gráficos para variables básicas.

### src/T2_analisis/F2_analisis_deducidas.py
Produce estadísticas y gráficos para variables deducidas.

### src/T2_analisis/F3_normalidad.py
Ejecuta pruebas de normalidad sobre muestras acotadas y guarda QQ plots.

### src/T2_analisis/F4_correlaciones.py
Calcula correlaciones numéricas y guarda resultados de correlación.

### src/T2_analisis/F5_asociacion.py
Calcula chi-cuadrado y Cramér's V para asociaciones categóricas y genera gráficos de contingencia.

### src/T2_analisis/F6_anova.py
Ejecuta ANOVA sobre montos por factores categóricos y, cuando está disponible, prueba de Tukey.

### src/T3_inferencia/F1_modelado_clustering.py
Implementa el modelado descriptivo con clustering K-Means sobre clientes. Agrupa las transacciones por `CODIGO CLIENTE`, calcula indicadores agregados y genera resultados, métricas, interpretaciones y gráficos en `output/modelado/clustering/` y `output/graficos/modelado/`.

## Salidas generadas

El pipeline deja resultados en estas rutas:

- `output/diagnostico/`: tipos de columna, nulos, duplicados y estadísticas básicas.
- `output/preprocesamiento/deducidas/`: resumen de variables deducidas.
- `output/preprocesamiento/modelos/`: parámetros usados durante la normalización.
- `output/analysis/`: CSVs con estadísticas, correlaciones, normalidad, asociación y ANOVA.
- `output/modelado/clustering/`: resultados de clustering, métricas, resumen y análisis de perfiles de cliente.
- `output/graficos/modelado/`: gráficos del método del codo y visualizaciones del clustering.
- `output/graficos/`: gráficos de EDA, normalidad, correlaciones, asociación y ANOVA.
- `output/log.txt`: registro de ejecución.

## Notas de memoria

Para evitar que el EDA consuma demasiada RAM, el proyecto trabaja sobre muestras acotadas en memoria mediante [`utiles/eda_memoria.py`](utiles/eda_memoria.py). Esto permite mantener el análisis útil incluso con archivos grandes.

