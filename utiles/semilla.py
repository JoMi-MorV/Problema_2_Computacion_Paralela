import os
import random
import numpy as np

# utiles/semilla.py
# Lectura y aplicación de la semilla determinista CPYD_SEED.

CPYD_SEED_ENV = "CPYD_SEED"
DEFAULT_SEED = 42


def get_seed(default=DEFAULT_SEED):
    """Devuelve la semilla entera definida por la variable de entorno CPYD_SEED.

    Si CPYD_SEED no existe o no es un entero válido, retorna el valor predeterminado.
    """
    raw = os.environ.get(CPYD_SEED_ENV, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def set_global_seed(default=DEFAULT_SEED):
    """Establece la semilla global para numpy y random."""
    seed = get_seed(default)
    np.random.seed(seed)
    random.seed(seed)
    return seed
