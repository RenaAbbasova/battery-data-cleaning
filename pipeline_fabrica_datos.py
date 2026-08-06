import pandas as pd
import numpy as np

# leer datos crudos tal como vienen

ruta_entrada = "data_lake/raw/battery_readings_raw.csv"
ruta_salida = "data_lake/processed/battery_data_clean.csv"

df = pd.read_csv(ruta_entrada)
print("Datos crudos cargados: ", df.shape)


es_fabricante_a = df["source"] == "fabricanteA"    
es_fabricante_b = df["source"] == "fabricanteB"

df.loc[es_fabricante_a, "timestamp"] =  pd.to_datetime(df.loc[es_fabricante_a, "timestamp"], format="%d/%m/%Y %H:%M")
df.loc[es_fabricante_b, "timestamp"] = pd.to_datetime(df.loc[es_fabricante_b, "timestamp"], format="%Y-%m-%dT%H:%M:%S")   


df["timestamp"] = pd.to_datetime(df["timestamp"])

es_farenheit = df["temp_unit"] == "F"
df.loc[es_farenheit, "temp"] = (df.loc[es_farenheit, "temp"] - 32)/1.8


df.loc[es_farenheit, "temp_unit"] = "C"
outliers = df["state_of_health_pct"] > 100

# print(df.dtypes)
# print(outliers)

df.loc[outliers, "state_of_health_pct"] = np.nan
print("Datos duplicados: ", df.duplicated().sum())
df = df.drop_duplicates()
df["vehicle_id"] = df["vehicle_id"].str.upper()
print(df.head(10))
df.to_csv(ruta_salida, index=False)
soh_vehiculo = df.groupby("vehicle_id")["state_of_health_pct"].mean().sort_values(ascending=True)
print("Promedio de estado de salud por vehículo:", soh_vehiculo)
print("Datos limpios guardados en battery_data_clean.csv")
print("Datos limpios guardados en:", ruta_salida)
