import os

# utiles/logger.py
# Logger simple que imprime en consola y guarda mensajes en output/log.txt.


os.makedirs("output", exist_ok=True)
LOG_FILE = "output/log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("===== INICIO DEL PROCESO =====\n")


def log(mensaje):
    print(mensaje)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(mensaje + "\n")


def log_oculto(mensaje):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(mensaje + "\n")