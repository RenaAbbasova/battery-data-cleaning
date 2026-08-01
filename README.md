#Battery data cleaning
#Que hice
- Normaliza timestamps de dos formatos distintos (dd/mm/yyyy vs ISO)
- Unifica temperatura de Fahrenheit a Celsius
- Detecta y trata outliers imposibles (state of health > 100%)
- Elimina duplicados
- Normaliza IDs de vehículo inconsistentes
- Calcula estado de salud medio por vehículo

## Cómo ejecutarlo

python3 limpieza_baterias.py

