import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go

import numpy as np
np.random.seed(0)
data = {
    'A': np.random.choice(['foo', 'bar', 'baz', 'qux'], size=200),
    'B': np.random.choice(['red', 'green', 'blue'], size=200),
    'E': np.random.choice(['foo', 'bar', 'baz', 'qux'], size=200),
    'F': np.random.choice(['foo', 'bar', 'baz', 'qux'], size=200),
    'C': np.random.choice(['apple', 'banana', 'orange'], size=200),
    'D': np.random.choice(['cat', 'dog', 'bird'], size=200)
}
df = pd.DataFrame(data)

# Altura base del gráfico
base_height = 200

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='first-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        placeholder="Seleccione una columna"
    ),
    html.Div([
        dcc.Dropdown(
            id='second-dropdown',
            placeholder="Seleccione una segunda columna",
            disabled=True
        ),
        html.Button('Añadir', id='add-button', n_clicks=0, disabled=True)
    ], style={'display': 'flex', 'align-items': 'center'}),
    dcc.Graph(id='heatmap', style={'display': 'none'}),  # Ocultar el gráfico inicialmente
    dcc.Store(id='selected-columns', data=[])  # Componente oculto para almacenar las columnas seleccionadas
])

# Callback para actualizar las opciones del segundo dropdown
@app.callback(
    Output('second-dropdown', 'options'),
    Output('second-dropdown', 'disabled'),
    Output('add-button', 'disabled'),
    Input('first-dropdown', 'value')
)
def set_second_dropdown_options(selected_column):
    if selected_column is None:
        return [], True, True
    
    unique_values = set(df[selected_column].unique())
    matching_columns = [
        col for col in df.columns
        if set(df[col].unique()) == unique_values and col != selected_column
    ]
    return [{'label': col, 'value': col} for col in matching_columns], False, False

# Callback para actualizar la lista de columnas seleccionadas
@app.callback(
    Output('selected-columns', 'data'),
    Input('add-button', 'n_clicks'),
    Input('first-dropdown', 'value'),  # Agregar input para el primer dropdown
    State('second-dropdown', 'value'),
    State('selected-columns', 'data')
)
def update_selected_columns(n_clicks, first_col, second_col, selected_columns):
    if n_clicks == 0:
        return [first_col] if first_col else []
    
    if second_col and second_col not in selected_columns:
        selected_columns.append(second_col)
    
    if first_col and first_col not in selected_columns:
        selected_columns.insert(0, first_col)
    
    return selected_columns

# Callback para actualizar el heatmap
@app.callback(
    Output('heatmap', 'figure'),
    Output('heatmap', 'style'),  # Agregar output para el estilo del gráfico
    Input('selected-columns', 'data'),
    State('first-dropdown', 'value')
)
def update_heatmap(selected_columns, first_col):
    if not selected_columns or first_col not in selected_columns:
        return go.Figure(), {'display': 'none'}  # Si no se ha seleccionado ninguna columna o si la primera columna no está seleccionada
    
    unique_values = sorted(set(df[first_col].unique()))
    heatmap_data = pd.DataFrame(columns=unique_values)
    heatmap_text = pd.DataFrame(columns=unique_values)
    
    for col in selected_columns:
        heatmap_data.loc[col] = 0
        for val in unique_values:
            count = (df[col] == val).sum()
            percentage = count / len(df) * 100
            heatmap_data.loc[col, val] = percentage
            heatmap_text.loc[col, val] = f"{percentage:.0f}% ({count})"  # Agregar la frecuencia entre paréntesis
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.fillna(0).values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text=heatmap_text.fillna("").values,
        colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 0, 255)']],
        zmin=0,
        zmax=100,
        hoverinfo="text",
        showscale=True
    ))
    
    # Añadir etiquetas de texto
    for i, row in heatmap_data.iterrows():
        for j, val in row.items():  # Cambiado iteritems() a items()
            fig.add_annotation(
                text=f"{heatmap_text.loc[i, j]}",
                x=j,
                y=i,
                showarrow=False,
                font=dict(color="black", size=12),
                xanchor="center",
                yanchor="middle"
            )
    
    fig.update_layout(
        xaxis_title="Valores únicos",
        yaxis_title="Columnas del DataFrame",
        xaxis=dict(side="top"),  # Colocar etiquetas de columnas en la parte superior
        coloraxis_colorbar=dict(
            title="Porcentaje",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]
        )
    )
    
    # Calcular el nuevo tamaño del gráfico en función del número de filas
    new_height = base_height + len(selected_columns) * 100
    
    return fig, {'display': 'block', 'height': new_height}

if __name__ == '__main__':
    app.run_server(debug=True)
