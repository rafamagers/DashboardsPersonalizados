import pandas as pd
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import bartlett

def realizar_tests(ruta_archivo):
    # Leer el archivo CSV
    datos = pd.read_csv(ruta_archivo)
    
    # Seleccionar las columnas a partir de la cuarta columna
    datos_seleccionados = datos.iloc[:, 3:]
    
    # Test de Bartlett
    chi_square_value, p_value = calculate_bartlett_sphericity(datos_seleccionados)
    
    # Test KMO
    kmo_all, kmo_model = calculate_kmo(datos_seleccionados)
    
    # Resultados
    resultados = {
        'bartlett': {
            'chi_square_value': chi_square_value,
            'p_value': p_value
        },
        'kmo': kmo_model
    }
    
    return resultados

# Ejemplo de uso
ruta_csv = "Z:\Downloads\Ejercicio_Escala-de-Cordialidad_Base-de-Datos.csv"
resultados = realizar_tests(ruta_csv)

# Imprimir los resultados
print("Test de Bartlett:")
print("Chi-Square Value:", resultados['bartlett']['chi_square_value'])
print("P-Value:", resultados['bartlett']['p_value'])
print("\nTest KMO:")
print("KMO Model:", resultados['kmo'])