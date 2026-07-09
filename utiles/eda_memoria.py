import os
import pandas as pd

MAX_ROWS_EDA = 50000
MAX_ROWS_PLOTS = 20000
MAX_ROWS_NORMALIDAD = 10000
MAX_WORKERS_EDA = max(1, min(3, os.cpu_count() or 1))


def obtener_muestra_pandas(obj, cols=None, max_rows=MAX_ROWS_EDA, seed=None):
    """Devuelve una muestra pequeña en memoria para análisis exploratorio.

    Usa head() sobre Dask/DataFrame para evitar materializar columnas completas.
    """
    if cols is not None:
        obj = obj[cols]

    try:
        muestra = obj.head(max_rows).compute()
    except Exception:
        try:
            muestra = obj.head(max_rows)
        except Exception:
            return pd.DataFrame() if hasattr(obj, 'columns') else pd.Series(dtype=float)

    if isinstance(muestra, pd.Series):
        if len(muestra) > max_rows:
            muestra = muestra.sample(n=max_rows, random_state=seed)
        return muestra.reset_index(drop=True)

    if not isinstance(muestra, pd.DataFrame):
        muestra = pd.DataFrame(muestra)

    if len(muestra) > max_rows:
        muestra = muestra.sample(n=max_rows, random_state=seed)

    return muestra.reset_index(drop=True)
