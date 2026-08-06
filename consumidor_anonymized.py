import pandas as pd
import numpy as np
ruta_ml_features = "data_lake/consumers/ml_features/features_vehiculo.csv"
ruta_anonymized = "data_lake/consumers/anonymized/datos_anonimizados.csv"

df_features = pd.read_csv(ruta_ml_features)
df_features["anon_id"]  = ["anon_" + str(i) for i in range(len(df_features))]
df_anonymized = df_features[["anon_id", "temp_max", "temp_min", "temp_rango", "dias_activacion"]]
df_anonymized.to_csv(ruta_anonymized, index=False)
print("Datos anonimizados guardados en:", ruta_anonymized) 
print(df_anonymized.head())