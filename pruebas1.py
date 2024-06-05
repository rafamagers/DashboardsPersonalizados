
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import io
import plotly.graph_objs as go
import numpy as np
import base64
from dash.exceptions import PreventUpdate
from ydata_profiling import ProfileReport
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import webbrowser
import dash_bootstrap_components as dbc
from collections import defaultdict

# Carga del archivo CSS externo
external_stylesheets=['styles.css']
#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css', 'styles.css']

# Inicialización de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Inicializar df como un DataFrame vacío

df = pd.DataFrame()
filtered_df = pd.DataFrame()


# Diseño de la aplicación



app.layout = html.Div([

    html.H1("Análisis descriptivo y exploratorio con Atributos Seleccionables", style={'textAlign': 'center'}),
   
    # Sección de carga de datos
    html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Span('Arrastra y suelta o ', className='upload-text'),  # Texto mejorado visualmente
                    html.A('Selecciona un archivo CSV', className="upload-button")
                ]),
                multiple=False
            ),
        ], className='upload-container'),  # Aplicar clase de contenedor centrado
        html.Div(id='output-data-upload', children= [html.P(f'Primero carga tu archivo CSV.')],className='upload-instructions')
    ], className='upload-section'),

        # Título y subtítulo
    html.H2("Análisis Descriptivo", className="title"),
    html.H3("Gráficos Univariados", className="subtitle"),
    # Dropdowns para la selección de atributos y tipos de gráfico
    html.Div([
        html.Div([
            html.Label("Seleccionar tipo de gráfico:", className="dropdown-label"),
            dcc.Dropdown(
                id='tipo-grafico-dropdown',
                options=[
                    
                    {'label': 'Diagrama de barra', 'value': 'histogram'},
                    
                    {'label': 'Gráfico de Torta', 'value': 'pie'},
                ],
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
            html.Label("Seleccionar atributo del eje X:", className="dropdown-label"),
            dcc.Dropdown(
                id='x-axis-dropdown',
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
    ], className='dropdown-container', style={'marginBottom': '50px'}),
    # Gráfico dinámico
    dcc.Graph(id='univariado-plot'),
    html.H3("Gráficos Bivariados", className="subtitle"),
    # Dropdowns para la selección de atributos y tipos de gráfico
    html.Div([
        html.Div([
            html.Label("Seleccionar tipo de gráfico:", className="dropdown-label"),
            dcc.Dropdown(
                id='tipo-grafico2-dropdown',
                options=[
                    
                    {'label': 'Diagrama de Polígono (Promedio)', 'value': 'polig'},
                    
                    {'label': 'Gráfico de barra bivariado', 'value': 'bibar'},
                ],
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
            html.Label("Seleccionar atributo del eje X:", className="dropdown-label"),
            dcc.Dropdown(
                id='x-axis-dropdown2',
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
            html.Label("Seleccionar Subrupo del eje X:", className="dropdown-label"),
            dcc.Dropdown(
                id='x-axis-dropdown3',
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
    ], className='dropdown-container', style={'marginBottom': '50px'}),
    # Gráfico de Barras Agrupadas (Bivariado)
    dcc.Graph(id='bi-plot'),
    html.H3("Otros Gráficos", className="subtitle"),
    # Dropdowns para la selección de atributos y tipos de gráfico
    html.Div([
        html.Div([
            html.Label("Seleccionar tipo de gráfico:", className="dropdown-label"),
            dcc.Dropdown(
                id='tipo-grafico3-dropdown',
                options=[
                    
                    {'label': 'Gráfico de radios o araña', 'value': 'araña'},
                    
                    {'label': 'Gráfico de multi barras', 'value': 'multibar'},

                    {'label': 'Gráfico de cajas', 'value': 'cajas'},
                ],
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
            html.Label("Seleccionar el conjunto de atributos:", className="dropdown-label"),
            dcc.Dropdown(
                id='x-axis-dropdown4',
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
    ], className='dropdown-container', style={'marginBottom': '50px'}),
    # Gráfico de Barras Agrupadas (Bivariado)
    dcc.Graph(id='multi-plot'),

    # Sección para el gráfico de barras agrupadas (bivariado)
    html.Hr(),  # Línea horizontal para separar las secciones
    

    html.H2("Análisis Exploratorio", className="title"),
    html.H3("Codificación de datos manual (Solo cadena a numérica)", className="subtitle"),
    dcc.Dropdown(
        id='dropdown-column',
        options=[{'label': col, 'value': col} for col in df.columns],
        value="",  # Columna predeterminada
        className="dropdown"
    ),
    html.Div(id='output-container'),
    html.Button('Confirmar cambios', id='confirm-button', n_clicks=0, className='confirm-button'),
    html.Div(id='output-container2'),

    html.H3("Visualizar y seleccionar columnas", className="subtitle"),
    # Checklist para seleccionar columnas
    html.Div([
        dcc.Checklist(
            id='column-selector',
            options=[{'label': col, 'value': col} for col in df.columns],
            value=df.columns.tolist(),  # Seleccionar todas las columnas por defecto
            inline=True
        ),
    ]),
    html.Div(id='nada'),
    html.Button('Seleccionar todas', id='select-all-button', n_clicks=0),
    html.Button('Deseleccionar todas', id='deselect-all-button', n_clicks=0),
    html.H3("Generar informe de análisis exploratorio para variables seleccionadas", className="subtitle"),


    html.Button("Abrir Informe", id="abrir-informe-button"),
    
    
    html.H3('Visualización de Análisis Factorial Exploratorio', className="subtitle"),
    html.Div([
        html.H3('Elija el número de factores que desea: ', style={'display': 'inline-block', 'margin-right': '10px'}),
        dcc.Input(id='input-number', type='number', value=3, min=1, max=10, step=1, style={'display': 'inline-block'}),
    ]),
    html.Button("Realizar análisis factorial", id="analisisfactorial", className='confirm-button'),
    html.Div([
        dcc.Graph(id='factor_loading_plot'),
        dcc.Graph(id='eigenvalue_plot'),
        dcc.Graph(id='explained_variance_plot')
    ])


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
def agrupar_codigos(codigos):
    grupos = defaultdict(list)

    for codigo in codigos:
        # Obtener la parte común hasta el guion
        parte_comun = codigo['label'].split('-')[0]

        # Agregar el código al grupo correspondiente
        grupos[parte_comun].append(codigo)

    # Generar el arreglo final solo para grupos con más de un código
    arreglo_final = [{'label': clave,
                      'value': '/'.join([cod['value'] for cod in codigos])}
                     for clave,codigos in grupos.items() if len(codigos) > 1]

    return arreglo_final
# Callback para cargar datos cuando se carga un archivo CSV y actualizar los dropdowns
@app.callback([Output('output-data-upload', 'children'),
               Output('x-axis-dropdown4', 'options'),
               Output('x-axis-dropdown', 'options'),
               Output('column-selector', 'options'),
               Output('x-axis-dropdown2', 'options'),
               Output('dropdown-column', 'options'),
               Output('x-axis-dropdown3', 'options')],
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is None:
        raise PreventUpdate
    else:
        options = parse_contents(contents)
        #selected_values = [option['value'] for option in options]
        opcionesespeciales = agrupar_codigos(options)
        return [html.P(f'Se ha cargado un archivo con {len(df)} filas y {len(df.columns)} columnas.')],opcionesespeciales,  options, options, options, options, options

# Callback para actualizar el gráfico dinámico
@app.callback(
    Output('univariado-plot', 'figure'),
    [Input('tipo-grafico-dropdown', 'value'),
     Input('x-axis-dropdown', 'value')],
)
def update_univariado(selected_grafico, selected_x):
    if selected_grafico == '' or selected_x=="":
        raise PreventUpdate
    counts = df[selected_x].value_counts().reset_index()
    counts.columns = ['x', 'count']
    if selected_grafico == 'histogram':
        fig = px.bar(counts, x='x', y='count', labels={'x': selected_x, 'count': 'Cantidad'})
    elif selected_grafico == 'pie':
        fig = px.pie(counts, names='x', values='count', title="Diagrama de Pie", labels={'x': selected_x, 'count': 'Cantidad'})
    else:
        raise PreventUpdate

    fig.update_layout(
        xaxis_title=f'{selected_x}',

        title=f'{selected_grafico.capitalize()}: {selected_x}'
    )
    return fig

# Callback para actualizar el gráfico de Barras Agrupadas (Bivariado)
@app.callback(
    Output('bi-plot', 'figure'),
    [Input('tipo-grafico2-dropdown', 'value'),
     Input('x-axis-dropdown2', 'value'),
     Input('x-axis-dropdown3', 'value')],
)
def update_bivariado(selected_grafico, selected_x, subgroup):
    if selected_grafico == '' or selected_x=="" or subgroup=="":
        raise PreventUpdate
    
    
    if selected_grafico == 'bibar':
        df_sorted = df.sort_values(by=subgroup)
        fig = px.bar(df_sorted, x=selected_x, color=subgroup, barmode='group',
                 labels={selected_x: selected_x, 'count': 'Frecuencia'},
                 title=f'Gráfico de Barras Agrupadas (Bivariado): {selected_x} vs. {subgroup}')
    elif selected_grafico == 'polig':
        df_grouped = df.groupby(selected_x).agg({subgroup: 'mean'}).reset_index()
        fig = px.line(df_grouped, x=selected_x, y=subgroup, markers=True)
    else:
        raise PreventUpdate

    fig.update_layout(
        xaxis_title=f'{selected_x}',

        title=f'{selected_grafico.capitalize()}: {selected_x}'
    )
    return fig
# Función para calcular las medias de las columnas
def calcular_medias(df, nombres_columnas):
    medias = []
    for columna in nombres_columnas:
        media_columna = df[columna].mean()
        media_columna = 100*(media_columna-1)/(df[columna].max()-1)
        medias.append(media_columna)
    return medias

def calcular_porcentajes_ocurrencias(df, nombres_columnas):
    porcentajes = []
    for columna in nombres_columnas:
        porcentajes_columna = []
        # Calcular la cantidad total de valores en la columna
        total_valores = df[columna].count()
        # Calcular el porcentaje de ocurrencia de cada valor (1, 2, 3, 4, 5)
        for valor in range(1, 6):
            porcentaje = round(df[columna].value_counts(normalize=True).get(valor, 0) * 100, 2)
            porcentajes_columna.append(porcentaje)
        porcentajes.append(porcentajes_columna)
    # Calcular el promedio de los porcentajes en cada arreglo de porcentajes
    promedios_porcentajes = [round(sum(columna) / len(columna), 2) for columna in zip(*porcentajes)]
    porcentajes.append(promedios_porcentajes)
    return porcentajes
# Callback para actualizar el gráfico de Barras Agrupadas (Bivariado)
@app.callback(
    Output('multi-plot', 'figure'),
    [Input('tipo-grafico3-dropdown', 'value'),
     Input('x-axis-dropdown4', 'value')],
)
def update_multivariado(selected_grafico, selected_x):
    if selected_grafico == '' or selected_x=="" :
        raise PreventUpdate
    
    columnas = selected_x.split("/")

    medias_columnas = calcular_medias(df, columnas)
    print(medias_columnas)

# Crear DataFrame auxiliar con nombres de columnas y medias
    if selected_grafico == 'araña':
        fig = go.Figure(data=go.Scatterpolar(
        r=medias_columnas,
        theta=columnas,
        fill='toself'
        ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            tickvals=[25,50,75,100],
            range=[0,100]
            ),
        ),
        showlegend=False
        )
    elif selected_grafico == 'cajas':
        fig = go.Figure()
        for media in columnas:
            # Use x instead of y argument for horizontal plot
            fig.add_trace(go.Box(x=df[media], name=media))
        for media in columnas:
            fig.add_trace(go.Scatter(x=[np.mean(df[media])], y=[media], mode='markers', marker=dict(color='red'), name='Media'))




    elif selected_grafico == 'multibar':
        top_labels = ['Mucho peor', 'Peor', 'Neutral', 'Mejor',
              'Mucho mejor']

        colors = ['rgba(255, 51, 51, 0.8)', 'rgba(255, 153, 51, 0.8)',
                'rgba(255, 230, 51, 0.8)', 'rgba(218, 255, 51, 0.85)',
                'rgba(119, 255, 51, 1)']

        x_data = calcular_porcentajes_ocurrencias(df, columnas)
        columnas.append("Promedio: ")
        y_data = columnas

        fig = go.Figure()

        for i in range(0, len(x_data[0])):
            for xd, yd in zip(x_data, y_data):
                fig.add_trace(go.Bar(
                    x=[xd[i]], y=[yd],
                    orientation='h',
                    marker=dict(
                        color=colors[i],
                        line=dict(color='rgb(248, 248, 249)', width=1)
                    )
                ))

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                domain=[0.15, 1]
                
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            barmode='stack',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(l=120, r=10, t=140, b=80),
            showlegend=False,
            
        )

        annotations = []

        for yd, xd in zip(y_data, x_data):
            # labeling the y-axis
            annotations.append(dict(xref='paper', yref='y',
                                    x=0.14, y=yd,
                                    xanchor='right',
                                    text=str(yd),
                                    font=dict(family='Arial', size=14,
                                            color='rgb(67, 67, 67)'),
                                    showarrow=False, align='right'))
            # labeling the first percentage of each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text=str(xd[0]) + '%',
                                    font=dict(family='Arial', size=14,
                                            color='rgb(0, 0, 0)'),
                                    showarrow=False))
            # labeling the first Likert scale (on the top)
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=xd[0] / 2, y=1.1,
                                        text=top_labels[0],
                                        font=dict(family='Arial', size=14,
                                                color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space = xd[0]
            for i in range(1, len(xd)):
                    # labeling the rest of percentages for each bar (x_axis)
                    annotations.append(dict(xref='x', yref='y',
                                            x=space + (xd[i]/2), y=yd,
                                            text=str(xd[i]) + '%',
                                            font=dict(family='Arial', size=14,
                                                    color='rgb(0, 0, 0)'),
                                            showarrow=False))
                    # labeling the Likert scale
                    if yd == y_data[-1]:
                        annotations.append(dict(xref='x', yref='paper',
                                                x=space + (xd[i]/2), y=1.1,
                                                text=top_labels[i],
                                                font=dict(family='Arial', size=14,
                                                        color='rgb(67, 67, 67)'),
                                                showarrow=False))
                    space += xd[i]
        fig.update_layout(annotations=annotations)
    else:
        raise PreventUpdate

    fig.update_layout(
        title=f'{selected_grafico.capitalize()}: {selected_x}'
    )
    return fig

@app.callback(
    Output("abrir-informe-button", "n_clicks"),
    Input("abrir-informe-button", "n_clicks")
)
def abrir_informe(n_clicks):
    if n_clicks:
        profile = ProfileReport(filtered_df, title="Pandas Profiling Report")
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
    if column == "":
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
        updated_df = df.copy()  # Copia del DataFrame para no modificar el original
        for child in children:
            input = child["props"]["children"][0]["props"]["children"]  # Obtener el ID del input
            updated_df.loc[updated_df[columna] == input, columna] = int(child["props"]["children"][1]["props"]["value"])
        # Actualizar el DataFrame original con los cambios
        df.update(updated_df)
        df[columna] = df[columna].astype(int)
        return 'Cambios confirmados y DataFrame actualizado.'
    else:
        return 'No se ha podido actualizar el dataframe (Recuerde que esto es para codificar)'

# Callback para actualizar las opciones del checklist y establecerlas como seleccionadas
@app.callback(
    Output('column-selector', 'value'),
    [
     Input('select-all-button', 'n_clicks'),
     Input('deselect-all-button', 'n_clicks')
    ],
    
)
def update_colmnschacklist(select_all_clicks, deselect_all_clicks):
    # Verificamos cuál botón se ha presionado
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Si se presionó el botón "Seleccionar todos"
    if button_id == 'select-all-button' and select_all_clicks:
        return list(df.columns)
    
    # Si se presionó el botón "Deseleccionar todos"
    elif button_id == 'deselect-all-button' and deselect_all_clicks:
        return []
    
    # Si no se presionó ninguno de los botones, evitamos la actualización
    raise PreventUpdate

# Callback para actualizar el DataFrame filtrado
@app.callback(
    Output("nada", 'children'),
    [Input('column-selector', 'value')]
)
def update_filtered_data(selected_columns):
    global filtered_df
    filtered_df = df[selected_columns]
    return ""

def generate_plots(loadings, eigenvalues, explained_variance):
    # Gráfico de cargas factoriales
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title('Gráfico de Cargas Factoriales')
    plt.xlabel('Factores')
    plt.ylabel('Variables')
    factor_loading_plot = plt.gcf()

    # Gráfico de valores propios
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='--')
    plt.title('Gráfico de Valores Propios')
    plt.xlabel('Número de Factor')
    plt.ylabel('Valor Propio')
    plt.grid(True)
    eigenvalue_plot = plt.gcf()

    # Gráfico de varianza explicada
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance[0])+1), explained_variance[0], marker='o', linestyle='-')
    plt.title('Gráfico de Varianza Explicada')
    plt.xlabel('Número de Factores')
    plt.ylabel('Varianza Explicada')
    plt.grid(True)
    explained_variance_plot = plt.gcf()

    return factor_loading_plot, eigenvalue_plot, explained_variance_plot


@app.callback(
    [Output('factor_loading_plot', 'figure'),
     Output('eigenvalue_plot', 'figure'),
     Output('explained_variance_plot', 'figure')
     ],
    [Input('analisisfactorial', 'n_clicks')],
    [State('input-number', 'value')]
  # Aquí debes especificar el elemento de entrada que desencadena la actualización de los gráficos
)
def update_plots(n_clicks,input_value):
    # Aquí deberías realizar el análisis factorial exploratorio y obtener los resultados
    # Luego, llamar a la función para generar los gráficos
    if n_clicks:
        if filtered_df.empty:
            raise PreventUpdate
        else:
            scaler = StandardScaler()
            
            data_scaled = scaler.fit_transform(filtered_df)
            correlation_matrix = filtered_df.corr()
            fa = FactorAnalyzer(n_factors=input_value, rotation=None) 
            fa.fit(data_scaled)
            print(fa.sufficiency)
            # Obtener valores propios
            eigenvalues = fa.get_eigenvalues()
            print(eigenvalues)
            # Obtener cargas factoriales
            loadings = fa.loadings_
            print(loadings)
            # Obtener varianza explicada
            explained_variance = fa.get_factor_variance()
            print(explained_variance)
            # Crear un gráfico de líneas para la varianza explicada
            # Crear el heatmap
            heatmap_trace = go.Heatmap(z=correlation_matrix.values,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.index,
                                    colorscale='Viridis',
                                    colorbar=dict(title='Correlation'))

            heatmap_layout = go.Layout(title='Correlation Matrix',
                                    xaxis=dict(title='Variables'),
                                    yaxis=dict(title='Variables'))

            heatmap_fig = go.Figure(data=[heatmap_trace], layout=heatmap_layout)

            # Gráfico de cargas factoriales
            loadings_trace = go.Heatmap(z=loadings.T,  # Transponer la matriz de cargas factoriales
                                        x=filtered_df.columns,
                                        y=list(range(1, loadings.shape[1] + 1)),  # Asegurarse de que el eje y tenga el tamaño correcto
                                        colorscale='Viridis',
                                        colorbar=dict(title='Loading'))
            # Crear figuras
            loadings_fig = go.Figure(data=[loadings_trace], layout=dict(title='Factor Loadings Heatmap', xaxis=dict(title='Variables'), yaxis=dict(title='Factor Number')))
            bar_traces = []
            for factor_num in range(loadings.shape[1]):  # Iterar sobre el segundo eje de 'loadings'
                bar_trace = go.Bar(x=filtered_df.columns, y=loadings[:, factor_num], name=f'Factor {factor_num+1}')
                bar_traces.append(bar_trace)

            bar_fig = go.Figure(data=bar_traces, layout=dict(title='Factor Loadings per Variable', xaxis=dict(title='Variable'), yaxis=dict(title='Loading')))

            # Retornar las figuras en tu callback
            return heatmap_fig, loadings_fig, bar_fig
    else:
        # Si no se ha presionado el botón, no se genera ningún gráfico
        raise PreventUpdate

    
if __name__ == '__main__':
    app.run_server(debug=True)


