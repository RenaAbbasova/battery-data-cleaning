import subprocess
scripts = [
    "pipeline_fabrica_datos.py",
    "consumidor_dashboard.py",
    "consumidor_ml_features.py",
    "consumidor_anonymized.py"
]

for script in scripts:
    print(f"Ejecutando {script}...")
    resultado = subprocess.run(["python3", script])
    if resultado.returncode != 0:
        print(f"Error: {script} fallo, deteniendo el pipeline")
        break 
print("\Fabrica de datos ejecutada")