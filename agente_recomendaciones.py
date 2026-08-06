import pandas as pd

ruta_ml_features = "data_lake/consumers/ml_features/features_vehiculo.csv"

df = pd.read_csv(ruta_ml_features)

def generar_recomendacion_ia(fila):
    """
    En una versión real, aquí se llamaría a la API de Anthropic/OpenAI,
    enviando los datos del vehículo y pidiendo una recomendación personalizada.
    Por ahora, simulamos la respuesta con lógica simple.
    """
    prompt = f"""
    Vehículo con temperatura máxima de {fila['temp_max']:.1f}°C,
    rango de temperatura de {fila['temp_rango']:.1f}°C,
    y {fila['dias_activacion']} días de actividad registrada.
    Genera una recomendación breve para el usuario sobre el cuidado de su batería.
    """
    
    # --- Aquí iría la llamada real a la API ---
    # respuesta = client.messages.create(model="...", messages=[{"role": "user", "content": prompt}])
    # return respuesta.content[0].text
    
    # --- Simulación mientras no hay API key ---
    if fila["temp_rango"] > 20:
        return "Tu batería sufre variaciones de temperatura altas. Evita cargar bajo sol directo."
    else:
        return "Tu batería mantiene condiciones estables. Sigue así."

df["recomendacion"] = df.apply(generar_recomendacion_ia, axis=1)

print(df[["vehicle_id", "temp_rango", "recomendacion"]])