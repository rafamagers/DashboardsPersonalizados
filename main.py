import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots  # Importar make_subplots
import base64
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from collections import defaultdict
import numpy as np
import plotly.io as pio
from plotly.graph_objs import Figure
from ydata_profiling import ProfileReport
import webbrowser
# Carga del archivo CSS externo
external_stylesheets = ['styles.css']
# Inicialización de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Inicializar df como un DataFrame vacío
df = pd.DataFrame()
filtered_df = pd.DataFrame()
# Diseño de la aplicación
app.layout = html.Div([
    html.H1("Dashboard creator", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Descriptive analysis', children=[
            html.Div([
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.Span('Drag and drop or ', className='upload-text'),
                            html.A('Choose CSV File', className="upload-button")
                        ]),
                        multiple=False
                    ),
                ], className='upload-container'),
                html.Div(id='output-data-upload', children=[html.P('First Upload your CSV File.')], className='upload-instructions')
            ], className='upload-section'),

            html.H3("Various Graphics", className="subtitle"),

            html.Div([
                html.Div([
                    html.Label("Choose your chart:", className="dropdown-label"),
                    dcc.Dropdown(
                        id='tipo-grafico-dropdown',
                        options=[
                            {'label': 'Bar Chart', 'value': 'histogram'},
                            {'label': 'Pie Chart', 'value': 'pie'},
                        ],
                        value='',
                        className="dropdown"
                    ),
                ], className='dropdown-item'),

                html.Div([
                    html.Label("Choose your variable:", className="dropdown-label"),
                    dcc.Dropdown(
                        id='x-axis-dropdown-graficosvarios',
                        value='',
                        className="dropdown"
                    ),
                ], className='dropdown-item'),

                html.Div([
                    html.Label("Choose your filter:", className="dropdown-label"),
                    dcc.Dropdown(
                        id='filter-dropdown-graficosvarios',
                        value='',
                        className="dropdown"
                    ),
                ], className='dropdown-item'),

            ], className='dropdown-container', style={'marginBottom': '50px'}),

            dcc.Graph(id='univariado-plot'),
            html.Button('Add graph to report', id='guardar1', n_clicks=0, className='confirm-button'),
                html.H3("Matrix Charts", className="subtitle"),
    # Dropdowns para la selección de atributos y tipos de gráfico
    html.Div([
        html.Div([
            html.Label("Choose your chart:", className="dropdown-label"),
            dcc.Dropdown(
                id='tipo-grafico3-dropdown',
                options=[
                    
                    {'label': 'Radar Chart', 'value': 'araña'},
                    
                    {'label': 'Multi bar Chart', 'value': 'multibar'},

                ],
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
            html.Label("Choose your group of variables (Matrix):", className="dropdown-label"),
            dcc.Dropdown(
                id='x-axis-dropdown-graficomult',
                value='',
                className="dropdown"
            ),
        ], className='dropdown-item'),
        html.Div([
                    html.Label("Choose your filter:", className="dropdown-label"),
                    dcc.Dropdown(
                        id='filtro-dropdown-matrix',
                        value='',
                        className="dropdown"
                    ),
                ], className='dropdown-item'),
        
    ], className='dropdown-container', style={'marginBottom': '50px'}),
    # Gráfico de Barras Agrupadas (Bivariado)
    dcc.Graph(id='multi-plot'),
    html.Button('Add graph to report', id='guardar2', n_clicks=0, className='confirm-button'),
    html.H3("Custom Heatmap", className="subtitle"),
    html.Div([
            html.Label("Choose your first variable:", className="dropdown-label"),
            dcc.Dropdown(
                id='heat-dropdown1',
                placeholder="Select a variable",
                className="dropdown"
            ),
    ], className='dropdown-item'),
    html.Div([
            html.Label("Choose your aditional variable: ", className="dropdown-label"),
            dcc.Dropdown(
                id='heat-dropdown2',
                className="dropdown",
                disabled=True
            ),
            html.Button('Add', id='add-button', n_clicks=0, disabled=True),
    ], className='dropdown-item'),
    
    dcc.Graph(id='heatmap', style={'display': 'none'}),  # Ocultar el gráfico inicialmente
    html.Button('Add graph to report', id='guardar3', n_clicks=0, className='confirm-button'),
    dcc.Store(id='selected-columns', data=[]),  # Componente oculto para almacenar las columnas seleccionadas
    html.H3("Generate Report", className="subtitle"),
    html.Button('Generar HTML', id='generate-html', n_clicks=0, className='confirm-button'),
    dcc.Download(id='download-html'),
    dcc.Store(id='selected-figures', data=[])
        ]),
        dcc.Tab(label='Exploratory Analysis', children=[
               html.H3("Select variables", className="subtitle"),
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
    html.Button('Select all', id='select-all-button', n_clicks=0),
    html.Button('Deselect all', id='deselect-all-button', n_clicks=0),
    html.H3("Generate automatic report", className="subtitle"),


    html.Button("Abrir Informe", id="abrir-informe-button"),
    
            # Aquí puedes agregar el contenido para la codificación manual de datos
        ]),
        dcc.Tab(label='Data codification', children=[
                html.H3("Manual data codification", className="subtitle"),
                dcc.Dropdown(
                    id='dropdown-column',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value="",  # Columna predeterminada
                    className="dropdown"
                ),
                html.Div(id='output-container'),
                html.Button('Confirm', id='confirm-button', n_clicks=0, className='confirm-button'),
                html.Div(id='output-container2'),
                html.Button('Save updated DF', id='save-button', n_clicks=0),
                html.Div(id='output-container-button')
            # Aquí puedes agregar el contenido para la codificación manual de datos
        ]),
    ])
])


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
@app.callback(
        
    [Output('x-axis-dropdown-graficosvarios', 'options'),
     Output('filter-dropdown-graficosvarios', 'options'),
     Output('filtro-dropdown-matrix', 'options'),
     Output('heat-dropdown1', 'options'),
     Output('dropdown-column', 'options'),
     Output('column-selector', 'options'),
     Output('x-axis-dropdown-graficomult', 'options')],
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename')
)
def update_dropdowns(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        global df
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        options = [{'label': col, 'value': col} for col in df.columns]
        opcionesespeciales = agrupar_codigos(options)
        return options, options, options,options,options, options, opcionesespeciales
    raise PreventUpdate

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return html.Div([
            html.H5(filename),
            html.H6("First 5 rows of the CSV Uploaded:"),
            dash.dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'}
            )
        ])
    raise PreventUpdate

@app.callback(
    Output('univariado-plot', 'figure'),
    Input('tipo-grafico-dropdown', 'value'),
    Input('x-axis-dropdown-graficosvarios', 'value'),
    Input('filter-dropdown-graficosvarios', 'value')
)
def update_varios_plot(tipo_grafico, x_axis, filtro):
    if tipo_grafico and x_axis and filtro:
        filtered_dfs = df.groupby(filtro)
        if tipo_grafico == 'histogram':
            fig = go.Figure()
            for filter_value, filtered_df in filtered_dfs:
                counts = filtered_df[x_axis].value_counts()
                total_counts = counts.sum()
                percentages = (counts / total_counts) * 100
                
                fig.add_trace(go.Bar(
                    x=counts.index,
                    y=counts,
                    name=str(filter_value),
                    text=[f'{p:.1f}%' for p in percentages],
                    textposition='auto'
                ))

            # Opcional: Actualizar el diseño del gráfico
            fig.update_layout(
                barmode='group',
                title=f"{x_axis} Filtered by {filtro}",
                xaxis_title=x_axis,
                yaxis_title='Frecuencia'
            )
        elif tipo_grafico == 'pie':
            unique_values = df[filtro].unique()
            fig = make_subplots(rows=1, cols=len(unique_values), specs=[[{'type': 'domain'}]*len(unique_values)])
            annotations = []
            for i, value in enumerate(unique_values):
                filtered_df = df[df[filtro] == value]
                fig.add_trace(
                    go.Pie(labels=filtered_df[x_axis].value_counts().index, values=filtered_df[x_axis].value_counts().values, name=str(value)),
                    row=1, col=i+1
                )
                annotations.append(dict(
                    x=0.5/len(unique_values) + i*(1.0/len(unique_values)),
                    y=-0.1,  # Ajusta esta posición según sea necesario
                    text=str(value),
                    showarrow=False,
                    xanchor='center',
                    yanchor='top'
                ))
            fig.update_layout(
                title=f"{x_axis} Filtered by {filtro}",
                annotations=annotations
            )
        return fig
    elif tipo_grafico and x_axis:
        if tipo_grafico == 'histogram':
            
            fig = go.Figure()
            counts = df[x_axis].value_counts()
            total_counts = counts.sum()
            percentages = (counts / total_counts) * 100
            fig.add_trace(go.Bar(
                x=counts.index,
                y=counts,
                text=[f'{p:.1f}%' for p in percentages],
                textposition='auto'
            ))
        elif tipo_grafico == 'pie':
            fig = px.pie(df, names=x_axis)
        return fig
    raise PreventUpdate
# Función para calcular las medias de las columnas
def calcular_medias(df, nombres_columnas):
    medias = []
    
    for columna in nombres_columnas:
        mapeo = {
            'Totalmente en desacuerdo': 1,
            'Mucho peor': 1,
            'Muy negativo':1,
            'En desacuerdo': 2,
            'Peor':2,
            'Negativo':2,
            'Neutral': 3,
            'Sin cambios': 3,
            'De acuerdo': 4,
            'Mejor':4,
            'Positivo':4,
            'Totalmente de acuerdo': 5,
            'Mucho mejor':5,
            'Muy positivo':5
        }

        # Aplicar el mapeo a la columna
        df[columna] = df[columna].map(mapeo).fillna(df[columna])
        media_columna = df[columna].mean()
        media_columna = 100*(media_columna-1)/(df[columna].max()-1)
        medias.append(media_columna)
    return medias
def generate_color_palette(n):
    # Definir los colores iniciales y finales en formato RGBA
    start_color = np.array([255, 51, 51, 0.8])  # Rojo
    end_color = np.array([119, 255, 51, 1])    # Verde

    # Crear un array de interpolación
    colors = np.linspace(start_color, end_color, n)

    # Convertir los colores a formato 'rgba(r, g, b, a)'
    rgba_colors = [f'rgba({int(r)}, {int(g)}, {int(b)}, {a:.2f})' for r, g, b, a in colors]

    return rgba_colors
def calcular_porcentajes_ocurrencias(df, nombres_columnas, toplabels):
    porcentajes = []
    for columna in nombres_columnas:
        porcentajes_columna = []
        # Calcular la cantidad total de valores en la columna
        total_valores = df[columna].count()
        # Calcular el porcentaje de ocurrencia de cada valor (1, 2, 3, 4, 5)


        for valor in toplabels:
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
     Input('x-axis-dropdown-graficomult', 'value'),
     Input('filtro-dropdown-matrix', 'value')]
)
def update_multivariado(selected_grafico, selected_x, filtro):
    if selected_grafico == '' or selected_x == '':
        raise PreventUpdate

    columnas = selected_x.split("/")
    titulo = columnas[0].split("-")[0]
    if (filtro == ''):
        unique_filter_values = ["nada"]
    else:
        unique_filter_values = df[filtro].unique()
    num_subplots = len(unique_filter_values)

    if selected_grafico == 'araña':
        
        fig = make_subplots(rows=(num_subplots+1)//2, cols=2, specs=[[{'type': 'polar'}, {'type': 'polar'}] for _ in range((num_subplots+1)//2)])        
        for i, valor in enumerate(unique_filter_values):
            if (valor=='nada'):
                filtered_df=df
            else:
                filtered_df = df[df[filtro] == valor]
            medias_columnas = calcular_medias(filtered_df, columnas)
            y_data = [elemento.split("-")[-1] for elemento in columnas]
            if (valor=='nada'):
                fig.add_trace(go.Scatterpolar(
                r=medias_columnas,
                theta=y_data,
                fill='toself',
                name=f'Radar chart'
               
                ), row=(i+2)//2, col=i%2+1)
            else:
                fig.add_trace(go.Scatterpolar(
                    r=medias_columnas,
                    theta=y_data,
                    fill='toself',
                    name=f'Filtro: {valor}'
                ), row=(i+2)//2, col=i%2+1)

            # Añadir anotación debajo de cada gráfica
      
                fig.add_annotation(
                    x=i / num_subplots + 1 / (2 * num_subplots),
                    y=-0.15,  # Posición y debajo de la gráfica
                    text=f'Filtro: {valor}',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    xanchor='center',
                    yanchor='top'
                )
            fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    tickvals=[25, 50, 75, 100],
                    range=[0, 100]
                ),
            ),
            row=(i+2)//2, col=i%2+1
            )
        fig.update_layout(
        height=(num_subplots+1)//2*500,  # Ajusta la altura total del gráfico
        width=1400,  # Ajusta el ancho total del gráfico
        showlegend=True,
        title_text="Gráficos Radar",
        )
        


    elif selected_grafico == 'multibar':
        fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True)
        annotations = []
        valores_unicos = df[columnas[0]].unique()
        if valores_unicos.dtype == np.int64 or np.array_equal(np.sort(valores_unicos), np.sort(np.array(["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"]))):
            top_labels = ["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"]
            if valores_unicos.dtype == np.int64:
                valoresdecolumna = [1, 2, 3, 4, 5]
            else:
                valoresdecolumna = top_labels
        elif np.array_equal(np.sort(valores_unicos), np.sort(np.array(["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"]))):
            top_labels = ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"]
            valoresdecolumna = top_labels
        elif np.array_equal(np.sort(valores_unicos), np.sort(np.array(['Mucho peor', 'Peor', 'Sin cambios', 'Mejor', 'Mucho mejor']))):
            top_labels = ['Mucho peor', 'Peor', 'Sin cambios', 'Mejor', 'Mucho mejor']
            valoresdecolumna = top_labels
        else:
            top_labels = valores_unicos
            valoresdecolumna = top_labels
        #Por filtro
        colors = generate_color_palette(len(top_labels))
        for i, valor in enumerate(unique_filter_values):
            if (valor=='nada'):
                filtered_df=df
            else:
                filtered_df = df[df[filtro] == valor]

            x_data = calcular_porcentajes_ocurrencias(filtered_df, columnas, valoresdecolumna)
            y_data = [elemento.split("-")[-1] for elemento in columnas]
            y_data.append("Mean")
            #Para cada porcentaje de una sola barra, normalmente serian 5 porcentajes
            for j in range(len(x_data[0])):
                #Ya depende del tamaño de la matrix incluyendo la media
                for xd, yd in zip(x_data, y_data):
                    fig.add_trace(go.Bar(
                        x=[xd[j]], y=["~"+str(yd)],
                        orientation='h',
                        marker=dict(
                            color=colors[j],
                            line=dict(color='rgb(248, 248, 249)', width=1)
                        ),
                       
                    ), row=i+1, col=1)

       
                    # Anotaciones de porcentajes en el eje x
                    annotations.append(dict(
                        xref=f'x{i+1}', yref=f'y{i+1}',
                        x=xd[0] / 2, y="~"+str(yd),
                        text=f'{xd[0]}%',
                        font=dict(family='Arial', size=14, color='rgb(0, 0, 0)'),
                        showarrow=False
                    ))
                    if yd == y_data[-1] and i==0:
                        annotations.append(dict(xref='paper', yref='y1',
                                                x=0 , y=len(y_data)+1,
                                                text=top_labels[0],
                                                font=dict(family='Arial', size=14,
                                                        color='rgb(67, 67, 67)'),
                                                showarrow=False))
                    space = xd[0]
                    for k in range(1, len(xd)):
                        annotations.append(dict(
                            xref=f'x{i+1}', yref=f'y{i+1}',
                            x=space + (xd[k] / 2), y="~"+str(yd),
                            text=f'{xd[k]}%',
                            font=dict(family='Arial', size=14, color='rgb(0, 0, 0)'),
                            showarrow=False
                        ))
                        if yd == y_data[-1] and i==0:
                            annotations.append(dict(xref='paper', yref='y1',
                                                    x=k/len(xd) , y=len(y_data)+1,
                                                    text=top_labels[k],
                                                    font=dict(family='Arial', size=14,
                                                            color='rgb(67, 67, 67)'),
                                                    showarrow=False))
                        space += xd[k]

            # Añadir anotación para el valor del filtro encima de cada subplot
            if (valor!= 'nada'):
                annotations.append(dict(
                    xref= "paper",yref=f'y{i+1}',
                    x=0.5, y=len(y_data),  # Ajustado para posicionar fuera del área de trazado
                    text=f'Filtro: {valor}',
                    font=dict(family='Arial', size=16, color='rgb(67, 67, 67)'),
                    showarrow=False, align='center'
                ))


        base_height = 400  # Altura base para una sola subtrama
        subplot_height = 250  # Altura adicional por cada subtrama adicional
        total_height = base_height + subplot_height * (num_subplots - 1)
        fig.update_layout(
            barmode='stack',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            height=total_height,  
            annotations=annotations, 
            showlegend=False,
            margin=dict(l=120, r=10, t=140, b=80), 
        )
    if (unique_filter_values[0] == "nada"):
        fig.update_layout(
        title=f'{titulo}',
        )
    else:
        fig.update_layout(
            title=f'{titulo} con Filtro: {filtro}',
        )
    return fig
@app.callback(
    Output('selected-figures', 'data'),
    [
        Input('guardar1', 'n_clicks'),
        Input('guardar2', 'n_clicks'),
        Input('guardar3', 'n_clicks')
    ],
    [
        State('selected-figures', 'data'),
        State('univariado-plot', 'figure'),
        State('multi-plot', 'figure'),
        State('heatmap', 'figure')
    ]
)
def add_figure_to_selection(n_clicks_1, n_clicks_2,n_clicks_3, selected_figures, figure_1, figure_2, figure_3):
    ctx = dash.callback_context
    if not ctx.triggered:
        return selected_figures
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'guardar1' and n_clicks_1:
            graph_figure = figure_1
        elif triggered_id == 'guardar2' and n_clicks_2:
            graph_figure = figure_2
        elif triggered_id == 'guardar3' and n_clicks_3:
            graph_figure = figure_3
        else:
            return selected_figures
        selected_figures.append(graph_figure)
        return selected_figures







# Callback para actualizar las opciones del segundo dropdown
@app.callback(
    Output('heat-dropdown2', 'options'),
    Output('heat-dropdown2', 'disabled'),
    Output('add-button', 'disabled'),
    Input('heat-dropdown1', 'value')
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
    Input('heat-dropdown1', 'value'),  # Agregar input para el primer dropdown
    State('heat-dropdown2', 'value'),
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
    State('heat-dropdown1', 'value')
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
    # Función para dividir una cadena en líneas cada n palabras
# Función para dividir una cadena en líneas cada n palabras
    def split_text_by_words(text, words_per_line):
        words = text.split()
        lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
        return '<br>'.join(lines)  # Usar '<br>' para un salto de línea en HTML

    # Definir el número de palabras por línea
    words_per_line = 6

    # Dividir las etiquetas del eje Y en líneas
    y_ticktext = [split_text_by_words(label, words_per_line) for label in heatmap_data.index]

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

    # Permitir saltos de línea en las etiquetas del eje Y
    fig.update_yaxes(tickvals=list(range(len(heatmap_data.index))),
                 ticktext=y_ticktext)


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
        xaxis_title="Heatmap",
        xaxis=dict(side="top"),  # Colocar etiquetas de columnas en la parte superior
        coloraxis_colorbar=dict(
            title="Porcentaje",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]
        )
    )
    
    # Calcular el nuevo tamaño del gráfico en función del número de filas
    new_height = 200 + len(selected_columns) * 100
    
    return fig, {'display': 'block', 'height': new_height}





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
            updated_df.loc[updated_df[columna] == input, columna] = child["props"]["children"][1]["props"]["value"]
        # Actualizar el DataFrame original con los cambios
        df.update(updated_df)
        
        return 'Dataframe updated'
    else:
        return ''


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

# Callback para guardar el DataFrame en un archivo CSV cuando se presiona el botón
@app.callback(
    Output('output-container-button', 'children'),
    Input('save-button', 'n_clicks')
)
def save_dataframe(n_clicks):
    if n_clicks > 0:
        # Nombre del archivo CSV
        filename = 'updated_dataframe.csv'
        # Guardar el DataFrame en un archivo CSV
        df.to_csv(filename, index=False)
        return f'DataFrame saved to {filename}'
    return ''

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












@app.callback(
    Output('download-html', 'data'),
    Input('generate-html', 'n_clicks'),
    State('selected-figures', 'data')
)
def generate_html(n_clicks, selected_figures):
    if n_clicks > 0 and selected_figures:
        html_content = '<html><body>'
        for i,fig_data in enumerate(selected_figures):
            # Convertir el diccionario de datos en un objeto Figure
            fig = Figure(fig_data)
            
            # Convertir la figura a imagen con las mismas dimensiones
            img_bytes = pio.to_image(fig, format="png", engine="kaleido", width=1500, height=800)
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Agregar la imagen al HTML
            html_content += '<h2>Gráfico</h2>'
            html_content += f'<img src="data:image/png;base64,{img_base64}" />'
        
        html_content += '</body></html>'
        return dict(content=html_content, filename="report.html")
    return dash.no_update
if __name__ == '__main__':
    app.run_server(debug=True)
