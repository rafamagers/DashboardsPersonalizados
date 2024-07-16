
import numpy as np
import pandas as pd
from RyStats.common import polychoric_correlation_serial
# Crear un DataFrame de ejemplo
data = {
    'col1': [1, 2, 3, 2, 1],
    'col2': [3, 1, 2, 3, 1],
}
df = pd.DataFrame(data)
# Convertir el DataFrame a una matriz NumPy
ordinal_data = df.to_numpy()
ordinal_data_transposed = ordinal_data.transpose()  # o ordinal_data.T

print(ordinal_data)    
# Ahora puedes pasar ordinal_data a polychoric_correlation
print(polychoric_correlation_serial(ordinal_data_transposed))
