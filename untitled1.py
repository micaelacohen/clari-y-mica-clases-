import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
 
def preparar_tiempo(df, columna_timestamp="timestamp"):
    """
    Crea una columna 'tiempo' que comienza en 0 segundos.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con los datos del experimento.
    columna_timestamp : str
        Nombre de la columna que contiene el tiempo registrado por Bonsai.

    Returns
    -------
    pandas.DataFrame
        Copia del DataFrame con una nueva columna 'tiempo'.

    Raises
    ------
    KeyError
        Si la columna de timestamp no existe.
    """
    if columna_timestamp not in df.columns:
        raise KeyError(f"No existe la columna '{columna_timestamp}'.")

    df = df.copy()
    df["tiempo"] = df[columna_timestamp] - df[columna_timestamp].min()
    return df


def calcular_hits_totales(hit):
    """ 
    Calcula la cantidad total de hits.

    Parameters
    ----------
    hit : pandas.Series
        Serie booleana donde True indica que ocurrió un hit.

    Returns
    -------
    int
        Cantidad total de hits.

    Raises
    ------
    ValueError
        Si la serie está vacía.
    """
    if len(hit) == 0:
        raise ValueError("La serie de hits está vacía.")

    return int(hit.sum())


def calcular_hit_rate(hit, tiempo):
    """
    Calcula el hit rate promedio en hits por segundo.

    Parameters
    ----------
    hit : pandas.Series
        Serie booleana donde True indica que ocurrió un hit.
    tiempo : pandas.Series
        Serie con el tiempo relativo en segundos.

    Returns
    -------
    float
        Hits por segundo.

    Raises
    ------
    ValueError
        Si la duración del registro es cero o negativa.
    """
    duracion = tiempo.max() - tiempo.min()

    if duracion <= 0:
        raise ValueError("La duración debe ser mayor a cero.")

    return hit.sum() / duracion


def calcular_tiempo_primer_hit(hit, tiempo):
    """
    Calcula el tiempo en el que ocurrió el primer hit.

    Parameters
    ----------
    hit : pandas.Series
        Serie booleana donde True indica que ocurrió un hit.
    tiempo : pandas.Series
        Serie con el tiempo relativo en segundos.

    Returns
    -------
    float or None
        Tiempo del primer hit. Devuelve None si no hubo hits.

    Raises
    ------
    ValueError
        Si las series tienen distinta longitud.
    """
    if len(hit) != len(tiempo):
        raise ValueError("Las series 'hit' y 'tiempo' deben tener la misma longitud.")

    tiempos_hit = tiempo[hit]

    if len(tiempos_hit) == 0:
        return None

    return float(tiempos_hit.min())


def calcular_tiempo_en_roi(activity_roi, tiempo):
    """
    Calcula cuánto tiempo total estuvo el sujeto dentro del ROI.

    Parameters
    ----------
    activity_roi : pandas.Series
        Serie booleana donde True indica que el sujeto está dentro del ROI.
    tiempo : pandas.Series
        Serie con el tiempo relativo en segundos.

    Returns
    -------
    float
        Tiempo total aproximado dentro del ROI, en segundos.

    Raises
    ------
    ValueError
        Si las series tienen distinta longitud.
    """
    if len(activity_roi) != len(tiempo):
        raise ValueError("Las series 'activity_roi' y 'tiempo' deben tener la misma longitud.")

    dt_promedio = tiempo.diff().median()

    if pd.isna(dt_promedio):
        return 0.0

    return float(activity_roi.sum() * dt_promedio)


def calcular_distancia_recorrida(x, y):
    """
    Calcula la distancia total recorrida en el plano XY.

    Parameters
    ----------
    x : pandas.Series
        Coordenada X.
    y : pandas.Series
        Coordenada Y.

    Returns
    -------
    float
        Distancia total recorrida en unidades de píxeles.

    Raises
    ------
    ValueError
        Si las series tienen distinta longitud.
    """
    if len(x) != len(y):
        raise ValueError("Las series 'x' e 'y' deben tener la misma longitud.")

    dx = x.diff()
    dy = y.diff()

    distancia = np.sqrt(dx**2 + dy**2)

    return float(distancia.sum())


def calcular_distancia_por_hit(x, y, hit):
    """
    Calcula cuánta distancia recorrió el sujeto por cada hit logrado.

    Parameters
    ----------
    x : pandas.Series
        Coordenada X.
    y : pandas.Series
        Coordenada Y.
    hit : pandas.Series
        Serie booleana donde True indica que ocurrió un hit.

    Returns
    -------
    float
        Distancia recorrida por hit.

    Raises
    ------
    ValueError
        Si no hubo hits.
    """
    hits_totales = hit.sum()

    if hits_totales == 0:
        raise ValueError("No se puede calcular distancia por hit porque no hubo hits.")

    distancia_total = calcular_distancia_recorrida(x, y)

    return distancia_total / hits_totales


def calcular_variabilidad_lateral(x):
    """
    Calcula la variabilidad lateral del movimiento.

    Parameters
    ----------
    x : pandas.Series
        Coordenada X del sujeto.

    Returns
    -------
    float
        Desvío estándar de la coordenada X.

    Raises
    ------
    ValueError
        Si la serie está vacía.
    """
    if len(x) == 0:
        raise ValueError("La serie de coordenada X está vacía.")

    return float(x.std())

# =========================
# CARGA DE DATOS
# =========================

df = pd.read_csv("/Users/clarabaietti/Documents/Clases programacion/output_salto_competicion_bruno_mica.csv")

# explorar columnas
print(df.columns)

# crear tiempo relativo
df = preparar_tiempo(df)

# =========================
# MÉTRICAS SUJETO A
# =========================

hits_A = calcular_hits_totales(df["hit_A"])
hit_rate_A = calcular_hit_rate(df["hit_A"], df["tiempo"])
primer_hit_A = calcular_tiempo_primer_hit(df["hit_A"], df["tiempo"])

tiempo_roi_A = calcular_tiempo_en_roi(
    df["activity_roi_A"],
    df["tiempo"]
)

distancia_A = calcular_distancia_recorrida(
    df["x_A"],
    df["y_A"]
)

distancia_por_hit_A = calcular_distancia_por_hit(
    df["x_A"],
    df["y_A"],
    df["hit_A"]
)

variabilidad_A = calcular_variabilidad_lateral(df["x_A"])


# =========================
# MÉTRICAS SUJETO B
# =========================

hits_B = calcular_hits_totales(df["hit_B"])
hit_rate_B = calcular_hit_rate(df["hit_B"], df["tiempo"])
primer_hit_B = calcular_tiempo_primer_hit(df["hit_B"], df["tiempo"])

tiempo_roi_B = calcular_tiempo_en_roi(
    df["activity_roi_B"],
    df["tiempo"]
)

distancia_B = calcular_distancia_recorrida(
    df["x_B"],
    df["y_B"]
)

distancia_por_hit_B = calcular_distancia_por_hit(
    df["x_B"],
    df["y_B"],
    df["hit_B"]
)

variabilidad_B = calcular_variabilidad_lateral(df["x_B"])


# =========================
# RESULTADOS
# =========================

print("\n===== RESULTADOS =====")

print("\nSUJETO A")
print("Hits totales:", hits_A)
print("Hit rate:", hit_rate_A)
print("Tiempo primer hit:", primer_hit_A)
print("Tiempo en ROI:", tiempo_roi_A)
print("Distancia recorrida:", distancia_A)
print("Distancia por hit:", distancia_por_hit_A)
print("Variabilidad lateral:", variabilidad_A)

print("\nSUJETO B")
print("Hits totales:", hits_B)
print("Hit rate:", hit_rate_B)
print("Tiempo primer hit:", primer_hit_B)
print("Tiempo en ROI:", tiempo_roi_B)
print("Distancia recorrida:", distancia_B)
print("Distancia por hit:", distancia_por_hit_B)
print("Variabilidad lateral:", variabilidad_B)


# =========================
# DETERMINAR GANADOR
# =========================

print("\n===== GANADOR =====")

if hits_A > hits_B:
    print("Ganó el sujeto A porque tuvo más hits.")
elif hits_B > hits_A:
    print("Ganó el sujeto B porque tuvo más hits.")
else:
    print("Empate en hits.")
    
    if hit_rate_A > hit_rate_B:
        print("Desempata a favor del sujeto A por mayor hit rate.")
    elif hit_rate_B > hit_rate_A:
        print("Desempata a favor del sujeto B por mayor hit rate.")
    else:
        print("Empate total.")


# =========================
# GRÁFICO 1
# Hits totales
# =========================

plt.figure(figsize=(5,5))

plt.bar(
    ["Sujeto A", "Sujeto B"],
    [hits_A, hits_B]
)

plt.title("Hits Totales")
plt.ylabel("Cantidad de Hits")

plt.show()


# =========================
# GRÁFICO 2
# Hit rate
# =========================

plt.figure(figsize=(5,5))

plt.bar(
    ["Sujeto A", "Sujeto B"],
    [hit_rate_A, hit_rate_B]
)

plt.title("Hit Rate")
plt.ylabel("Hits por segundo")

plt.show()


# =========================
# GRÁFICO 3
# Distancia por hit
# =========================

plt.figure(figsize=(5,5))

plt.bar(
    ["Sujeto A", "Sujeto B"],
    [distancia_por_hit_A, distancia_por_hit_B]
)

plt.title("Distancia por Hit")
plt.ylabel("Distancia")

plt.show()


# =========================
# GRÁFICO 4
# Hits acumulados
# =========================

hits_acumulados_A = df["hit_A"].cumsum()
hits_acumulados_B = df["hit_B"].cumsum()

plt.figure(figsize=(8,5))

plt.plot(
    df["tiempo"],
    hits_acumulados_A,
    label="Sujeto A"
)

plt.plot(
    df["tiempo"],
    hits_acumulados_B,
    label="Sujeto B"
)

plt.title("Hits Acumulados")
plt.xlabel("Tiempo")
plt.ylabel("Hits")

plt.legend()

plt.show()


# =========================
# GRÁFICO 5
# Trayectoria XY
# =========================

plt.figure(figsize=(7,7))

plt.plot(
    df["x_A"],
    df["y_A"],
    label="Sujeto A"
)

plt.plot(
    df["x_B"],
    df["y_B"],
    label="Sujeto B"
)

plt.title("Trayectoria XY")
plt.xlabel("X")
plt.ylabel("Y")

plt.legend()

plt.show()

            
            

            
        
      