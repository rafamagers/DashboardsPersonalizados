from tkinter import ALL
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd

# Ejemplo de DataFrame
data = {
    'Nombre': ['Juan', 'María', 'Pedro', 'Ana', 'Juan'],
    'Edad': [25, 30, 35, 40, 20],
    'Género': ['M', 'F', 'M', 'F', 'F']
}
df = pd.DataFrame(data)

app = dash.Dash(__name__)

# Función para generar el contenido dinámico basado en el DataFrame
def generate_table(column):
    column_values = df[column].unique()
    rows = []
    for value in column_values:
        row = html.Div([
            html.Div(value, style={'display': 'inline-block', 'width': '30%'}),
            dcc.Input(id={'type': 'input', 'index': value}, value=value, type='text', style={'display': 'inline-block', 'width': '60%'})
        ])
        rows.append(row)
    return rows

# Diseño de la aplicación
app.layout = html.Div([
    dcc.Dropdown(
        id='dropdown-column',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[0]  # Columna predeterminada
    ),
    html.Div(id='output-container'),
    html.Button('Confirmar cambios', id='confirm-button', n_clicks=0),
    html.Div(id='output-container2'),
])

# Callback para actualizar la tabla según la columna seleccionada
@app.callback(
    Output('output-container', 'children'),
    [Input('dropdown-column', 'value')]
)
def update_table(column):
    if column is None:
        return []
    else:
        return generate_table(column)

# Callback para actualizar el DataFrame con los cambios
@app.callback(
    Output('output-container2', 'children'),
    [Input('confirm-button', 'n_clicks'),
     Input('dropdown-column', 'value')],
   [State('output-container', 'children')]
  
)
def update_df(n_clicks,columna, children):
    
    if n_clicks > 0:
        print(df)
        updated_df = df.copy()  # Copia del DataFrame para no modificar el original
        for child in children:
            print(child)
            print("gee")
          
            input = child["props"]["children"][0]["props"]["children"]  # Obtener el ID del input
            updated_df.loc[updated_df[columna] == input, columna] = int(child["props"]["children"][1]["props"]["value"])
        # Actualizar el DataFrame original con los cambios
        df.update(updated_df)
        print(df)
        print(df["Nombre"][0]+df["Nombre"][1])
        return 'Cambios confirmados y DataFrame actualizado.'
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
