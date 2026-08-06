import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("battery_data_clean.csv", parse_dates=["timestamp"])
print(df.head(10))
# print(df.info())
# print(df.describe())
# print(df.shape)

temp_media = df.groupby("vehicle_id")["temp"].mean().sort_values(ascending=False)

# print(temp_media)
soh_vehiculo = df.groupby("vehicle_id")["state_of_health_pct"].mean().sort_values(ascending=False) 
temp_df = temp_media.reset_index()
soh_df = soh_vehiculo.reset_index()
combinada = pd.merge(temp_df, soh_df, on="vehicle_id")
# print(combinada.head(10))
corr = combinada["temp"].corr(combinada["state_of_health_pct"])
print("Correlación entre temperatura promedio y estado de salud promedio: ", corr)
