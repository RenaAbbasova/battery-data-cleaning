import pandas as pd
ruta_processed = "data_lake/processed/battery_data_clean.csv"
ruta_dashboard = "data_lake/consumers/dashboard/resumen_usuario.csv"

df = pd.read_csv(ruta_processed, parse_dates=["timestamp"])

df_ordenado = df.sort_values(by=["timestamp"])
antes = df_ordenado .groupby("vehicle_id").first()
despues = df_ordenado.groupby("vehicle_id").last()

degradacion = despues["state_of_health_pct"] - antes["state_of_health_pct"]
print(degradacion.sort_values())

resumen = pd.DataFrame({
    "vehicle_id": despues.index,
    "soh_actual": despues["state_of_health_pct"],
    "fecha_ultimo_registro": despues["timestamp"],
    "degradacion_total": degradacion
})

resumen.to_csv(ruta_dashboard, index=False)
print("Resumen de usuario guardado en:", ruta_dashboard)
