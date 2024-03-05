import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import io
import base64
from dash.exceptions import PreventUpdate

# Crear una aplicación Dash
app = dash.Dash(__name__)

# Inicializar df como un DataFrame vacío
df = pd.DataFrame()

# Diseño de la aplicación
app.layout = html.Div([
    html.H1("Gráficos Interactivos con Atributos Seleccionables"),
    
    # Upload de archivo CSV
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra y suelta o ', html.A('selecciona un archivo CSV')]),
        multiple=False
    ),
    
    html.Div(id='output-data-upload'),
    
    # Dropdown para seleccionar el tipo de gráfico
    html.Label("Seleccionar tipo de gráfico:"),
    dcc.Dropdown(
        id='tipo-grafico-dropdown',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Histograma', 'value': 'histogram'},
            {'label': 'Polígono de Promedios', 'value': 'line'},
            {'label': 'Gráfico de Barras', 'value': 'bar'},
            {'label': 'Gráfico de Torta', 'value': 'pie'}
        ],
        value='scatter'
    ),
    
    # Dropdown para seleccionar el atributo del eje X
    html.Label("Seleccionar atributo del eje X:"),
    dcc.Dropdown(
        id='x-axis-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar el atributo del eje Y
    html.Label("Seleccionar atributo del eje Y:"),
    dcc.Dropdown(
        id='y-axis-dropdown',
        value=''
    ),
    
    # Gráfico dinámico
    dcc.Graph(id='dynamic-plot'),
    
    # Sección para Bubble Chart
    html.Hr(),  # Línea horizontal para separar las secciones
    
    html.H2("Bubble Chart Configurable"),
    
    # Dropdown para seleccionar la dimensión X del Bubble Chart
    html.Label("Seleccionar dimensión X:"),
    dcc.Dropdown(
        id='bubble-x-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión Y del Bubble Chart
    html.Label("Seleccionar dimensión Y:"),
    dcc.Dropdown(
        id='bubble-y-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión del tamaño de las burbujas
    html.Label("Seleccionar dimensión del tamaño de las burbujas:"),
    dcc.Dropdown(
        id='bubble-size-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión del color de las burbujas
    html.Label("Seleccionar dimensión del color de las burbujas:"),
    dcc.Dropdown(
        id='bubble-color-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión de la etiqueta de las burbujas
    html.Label("Seleccionar dimensión de la etiqueta de las burbujas:"),
    dcc.Dropdown(
        id='bubble-label-dropdown',
        value=''
    ),
    
    # Gráfico de Burbujas
    dcc.Graph(id='bubble-plot'),
])

# Función para cargar los datos desde un archivo CSV
def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    global df  # Hacer referencia a la variable global df
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    options = [{'label': col, 'value': col} for col in df.columns]
    return options

# Callback para cargar datos cuando se carga un archivo CSV y actualizar los dropdowns
@app.callback([Output('output-data-upload', 'children'),
               Output('x-axis-dropdown', 'options'),
               Output('y-axis-dropdown', 'options'),
               Output('bubble-x-dropdown', 'options'),
               Output('bubble-y-dropdown', 'options'),
               Output('bubble-size-dropdown', 'options'),
               Output('bubble-color-dropdown', 'options'),
               Output('bubble-label-dropdown', 'options')],
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is None:
        raise PreventUpdate
    else:
        options = parse_contents(contents)
        return [html.P(f'Se ha cargado un archivo con {len(df)} filas y {len(df.columns)} columnas.')], options, options, options, options, options, options, options

# Callback para actualizar el gráfico dinámico
@app.callback(
    Output('dynamic-plot', 'figure'),
    [Input('tipo-grafico-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')],
)
def update_dynamic_plot(selected_grafico, selected_x, selected_y):
    if selected_grafico == 'scatter':
        fig = px.scatter(df, x=selected_x, y=selected_y)
    elif selected_grafico == 'histogram':
        fig = px.histogram(df, x=selected_x, y=selected_y, marginal="rug", nbins=20)
    elif selected_grafico == 'line':
        df_grouped = df.groupby(selected_x).agg({selected_y: 'mean'}).reset_index()
        fig = px.line(df_grouped, x=selected_x, y=selected_y, markers=True)
    elif selected_grafico == 'bar':
        fig = px.bar(df, x=selected_x, y=selected_y)
    elif selected_grafico == 'pie':
        fig = px.pie(df, names=selected_x, values=selected_y)
    else:
        fig = px.scatter(df, x=selected_x, y=selected_y)

    fig.update_layout(
        xaxis_title=f'{selected_x}',
        yaxis_title=f'{selected_y}',
        title=f'{selected_grafico.capitalize()}: {selected_y} vs. {selected_x}'
    )
    return fig

# Callback para actualizar el gráfico de Burbujas
@app.callback(
    Output('bubble-plot', 'figure'),
    [Input('bubble-x-dropdown', 'value'),
     Input('bubble-y-dropdown', 'value'),
     Input('bubble-size-dropdown', 'value'),
     Input('bubble-color-dropdown', 'value'),
     Input('bubble-label-dropdown', 'value')],
)
def update_bubble_plot(selected_x, selected_y, selected_size, selected_color, selected_label):
    if not all([selected_x, selected_y, selected_size, selected_color, selected_label]):
        raise PreventUpdate

    fig = px.scatter(df, x=selected_x, y=selected_y, size=selected_size, color=selected_color, labels={selected_x: selected_x, selected_y: selected_y},
                     hover_name=selected_label, title=f'Bubble Chart: {selected_size} vs. {selected_color} vs. {selected_x} vs. {selected_y}')
    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
