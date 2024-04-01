import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import io
import base64
from dash.exceptions import PreventUpdate
from ydata_profiling import ProfileReport
import webbrowser
import dash_bootstrap_components as dbc
external_stylesheets = ['assets/styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Inicializar df como un DataFrame vacío

df = pd.DataFrame()


# Diseño de la aplicación



app.layout = html.Div([

    html.H1("Análisis descriptivo y exploratorio con Atributos Seleccionables", style={'textAlign': 'center'}),
   
    
    # Sección de carga de datos
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Arrastra y suelta o ', html.A('selecciona un archivo CSV')]),
            multiple=False
        ),
        html.Div(id='output-data-upload')
    ]),
    
    # Dropdowns para la selección de atributos y tipos de gráfico
    html.Div([
        html.Div([
            html.Label("Seleccionar tipo de gráfico:"),
            dcc.Dropdown(
                id='tipo-grafico-dropdown',
                options=[
                    {'label': 'Diagrama de puntos', 'value': 'scatter'},
                    {'label': 'Histograma (SUMA)', 'value': 'histogram'},
                    {'label': 'Polígono de Promedios', 'value': 'line'},
                    {'label': 'Gráfico de Torta (SUMA)', 'value': 'pie'},
                ],
                value=''
            ),
        ], className='six columns'),
        html.Div([
            html.Label("Seleccionar atributo del eje X:"),
            dcc.Dropdown(
                id='x-axis-dropdown',
                value=''
            ),
        ], className='two columns'),
        html.Div([
            html.Label("Seleccionar atributo del eje Y:"),
            dcc.Dropdown(
                id='y-axis-dropdown',
                value=''
            ),
        ], className='two columns'),
    ], className='row', style={'marginBottom': '50px'}),
    
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
    
    # Sección para el gráfico en 3D
    html.Hr(),  # Línea horizontal para separar las secciones
    
    html.H2("Gráfico en 3D Configurable"),
    
    # Dropdown para seleccionar la dimensión X del gráfico 3D
    html.Label("Seleccionar dimensión X (3D):"),
    dcc.Dropdown(
        id='3d-x-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión Y del gráfico 3D
    html.Label("Seleccionar dimensión Y (3D):"),
    dcc.Dropdown(
        id='3d-y-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar la dimensión Z del gráfico 3D
    html.Label("Seleccionar dimensión Z (3D):"),
    dcc.Dropdown(
        id='3d-z-dropdown',
        value=''
    ),
    
    # Gráfico en 3D
    dcc.Graph(id='3d-plot'),
    
    # Sección para el gráfico de barras agrupadas (bivariado)
    html.Hr(),  # Línea horizontal para separar las secciones
    
    html.H2("Gráfico de Barras Agrupadas (Bivariado)"),
    
    # Dropdown para seleccionar el atributo del eje X en el gráfico de barras agrupadas
    html.Label("Seleccionar atributo del eje X:"),
    dcc.Dropdown(
        id='grouped-bar-x-dropdown',
        value=''
    ),
    
    # Dropdown para seleccionar el atributo del subgrupo en el gráfico de barras agrupadas
    html.Label("Seleccionar atributo del subgrupo del eje X:"),
    dcc.Dropdown(
        id='grouped-bar-subgroup-dropdown',
        value=''
    ),
    
    # Gráfico de Barras Agrupadas (Bivariado)
    dcc.Graph(id='grouped-bar-bi-plot'),
    dcc.Dropdown(
        id='dropdown-column',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=""  # Columna predeterminada
    ),
    html.Div(id='output-container'),
    html.Button('Confirmar cambios', id='confirm-button', n_clicks=0),
    html.Div(id='output-container2'),

    



    html.Button("Abrir Informe", id="abrir-informe-button")


])

# Función para cargar los datos desde un archivo CSV
def parse_contents(contents):
    global checkboxes
    content_type, content_string = contents.split(',')
    #global checkboxes
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
               Output('bubble-label-dropdown', 'options'),
               Output('3d-x-dropdown', 'options'),
               Output('3d-y-dropdown', 'options'),
               Output('3d-z-dropdown', 'options'),
               Output('grouped-bar-x-dropdown', 'options'),
               Output('dropdown-column', 'options'),
               Output('grouped-bar-subgroup-dropdown', 'options')],
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is None:
        raise PreventUpdate
    else:
        options = parse_contents(contents)
        return [html.P(f'Se ha cargado un archivo con {len(df)} filas y {len(df.columns)} columnas.')], options, options, options, options, options, options, options, options,options, options, options, options, options

# Callback para actualizar el gráfico dinámico
@app.callback(
    Output('dynamic-plot', 'figure'),
    [Input('tipo-grafico-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')],
)
def update_dynamic_plot(selected_grafico, selected_x, selected_y):
    if selected_grafico == '' or selected_x=="" or selected_y=="":
        raise PreventUpdate
    
    if selected_grafico == 'scatter':
        fig = px.scatter(df, x=selected_x, y=selected_y)
    elif selected_grafico == 'histogram':
        fig = px.histogram(df, x=selected_x, y=selected_y, marginal="rug", nbins=20)
    elif selected_grafico == 'line':
        df_grouped = df.groupby(selected_x).agg({selected_y: 'mean'}).reset_index()
        fig = px.line(df_grouped, x=selected_x, y=selected_y, markers=True)
    elif selected_grafico == 'pie':
        fig = px.pie(df, names=selected_x, values=selected_y)
    else:
        raise PreventUpdate

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

# Callback para actualizar el gráfico en 3D
@app.callback(
    Output('3d-plot', 'figure'),
    [Input('3d-x-dropdown', 'value'),
     Input('3d-y-dropdown', 'value'),
     Input('3d-z-dropdown', 'value')],
)
def update_3d_plot(selected_x, selected_y, selected_z):
    if not all([selected_x, selected_y, selected_z]):
        raise PreventUpdate

    fig = px.scatter_3d(df, x=selected_x, y=selected_y, z=selected_z, title=f'Gráfico en 3D: {selected_x} vs. {selected_y} vs. {selected_z}')
    return fig
# Callback para actualizar el gráfico de Barras Agrupadas (Bivariado)
@app.callback(
    Output('grouped-bar-bi-plot', 'figure'),
    [Input('grouped-bar-x-dropdown', 'value'),
     Input('grouped-bar-subgroup-dropdown', 'value')],
)
def update_grouped_bar_bi_plot(selected_x, selected_subgroup):
    if not all([selected_x, selected_subgroup]):
        raise PreventUpdate

    # Ordenar los datos por el subgrupo seleccionado
    df_sorted = df.sort_values(by=selected_subgroup)
    
    fig = px.bar(df_sorted, x=selected_x, color=selected_subgroup, barmode='group',
                 labels={selected_x: selected_x, 'count': 'Frecuencia'},
                 title=f'Gráfico de Barras Agrupadas (Bivariado): {selected_x} vs. {selected_subgroup}')
    return fig


@app.callback(
    Output("abrir-informe-button", "n_clicks"),
    Input("abrir-informe-button", "n_clicks")
)
def abrir_informe(n_clicks):
    if n_clicks:
        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile.to_file("tu_informe.html")
        # Abre el informe en una nueva ventana o pestaña
        webbrowser.open_new_tab("tu_informe.html")

    # Si no se ha hecho clic en el botón, no hace nada
    return None
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
        return 'Cambios confirmados y DataFrame actualizado.'
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)


