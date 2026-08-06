import pandas as pd
import numpy as np

ruta_processed = "data_lake/processed/battery_data_clean.csv"
ruta_ml_features = "data_lake/consumers/ml_features/features_vehiculo.csv"

df = pd.read_csv(ruta_processed, parse_dates=["timestamp"])

temp_por_vehiculo = df.groupby("vehicle_id")["temp"]
temp_max = temp_por_vehiculo.max()
temp_min = temp_por_vehiculo.min() 
temp_rango = temp_max - temp_min

df_ordenado = df.sort_values(["timestamp"])
dias_por_vehiculo = df_ordenado.groupby("vehicle_id")["timestamp"]
primer_dia = dias_por_vehiculo.first()
ultimo_dia = dias_por_vehiculo.last()
dias_activacion = (ultimo_dia - primer_dia).dt.days

features = pd.DataFrame({
    "vehicle_id": temp_max.index,
    "temp_max": temp_max,
    "temp_min": temp_min,
    "temp_rango": temp_rango,
    "dias_activacion": dias_activacion
})

features.to_csv(ruta_ml_features, index=False)
print("Features de ML generadas y guardads en:", ruta_ml_features)
print(features.head())