import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ejemplo de datos y colores
y_data = [1, 2, 3, 4]
labelito = "Ejemplo"
color = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33']

# Crear subplots
fig = make_subplots(rows=1, cols=1)

# Añadir algunos datos de ejemplo
fig.add_trace(go.Bar(x=[1, 2, 3, 4], y=y_data), row=1, col=1)

annotations = []

# Ciclo para generar las anotaciones
for k in range(len(y_data)):
    # Definir el cuadrado de color usando el símbolo unicode
    color_square = f"<span style='color:{color[k]}'>■</span>"
    # Crear el texto con el cuadrado y el texto de la etiqueta
    texto_con_color = f"{color_square} {labelito}"
    # Añadir la anotación
    annotations.append(dict(
        xref='x1',
        yref='y1',
        x=10,  # Ajusta esto según el rango de tu subplot
        y=len(y_data) - k,  # Ajuste de y para cada anotación
        text=texto_con_color,
        font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'),
        showarrow=False,
        align='left'
    ))

# Actualizar layout de la figura con las anotaciones
fig.update_layout(
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    height=600,  # Ajusta la altura según sea necesario
    annotations=annotations,
    showlegend=False,
    margin=dict(l=120, r=10, t=140, b=80)
)

# Mostrar la figura en Streamlit
st.plotly_chart(fig)
