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
# Carga del archivo CSS externo
external_stylesheets = ['styles.css']

# Inicialización de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Inicializar df como un DataFrame vacío
df = pd.DataFrame()

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
    ], className='dropdown-container', style={'marginBottom': '50px'}),
    # Gráfico de Barras Agrupadas (Bivariado)
    dcc.Graph(id='multi-plot'),

        ]),
        dcc.Tab(label='Exploratory Analysis', children=[
            # Aquí puedes agregar el contenido para la codificación manual de datos
        ]),
        dcc.Tab(label='Data codification', children=[
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
        return options, options, opcionesespeciales
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
                fig.add_trace(go.Histogram(
                    x=filtered_df[x_axis],
                    name=str(filter_value),
                    opacity=0.75
                ))
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
                title=f"Pie Chart of {x_axis} Filtered by {filtro}",
                annotations=annotations
            )
        return fig
    elif tipo_grafico and x_axis:
        if tipo_grafico == 'histogram':
            fig = px.histogram(df, x=x_axis)
        elif tipo_grafico == 'pie':
            fig = px.pie(df, names=x_axis)
        return fig
    raise PreventUpdate
# Función para calcular las medias de las columnas
def calcular_medias(df, nombres_columnas):
    medias = []
    for columna in nombres_columnas:
        media_columna = df[columna].mean()
        media_columna = 100*(media_columna-1)/(df[columna].max()-1)
        medias.append(media_columna)
    return medias

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
     Input('x-axis-dropdown-graficomult', 'value')],
)
def update_multivariado(selected_grafico, selected_x):
    if selected_grafico == '' or selected_x=="" :
        raise PreventUpdate
    
    columnas = selected_x.split("/")
    titulo = columnas[0].split("-")[0]
  

# Crear DataFrame auxiliar con nombres de columnas y medias
    if selected_grafico == 'araña':
        medias_columnas = calcular_medias(df, columnas)
        print(medias_columnas)
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
    elif selected_grafico == 'multibar':
        valores_unicos = df[columnas[0]].unique()
        print(valores_unicos)
        if  valores_unicos.dtype == np.int64 or np.array_equal(np.sort(valores_unicos), np.sort(np.array( ["Muy negativo","Negativo","Neutral","Positivo","Muy positivo"]))):
            top_labels = ["Muy negativo","Negativo","Neutral","Positivo","Muy positivo"]
            if valores_unicos.dtype == np.int64:
                valoresdecolumna = [1,2,3,4,5]
            else:
                valoresdecolumna = top_labels

        elif np.array_equal(np.sort(valores_unicos), np.sort(np.array( ["Totalmente en desacuerdo","En desacuerdo","Neutral","De acuerdo","Totalmente de acuerdo"]))):
            top_labels = ["Totalmente en desacuerdo","En desacuerdo","Neutral","De acuerdo","Totalmente de acuerdo"]
            valoresdecolumna = top_labels
        elif np.array_equal(np.sort(valores_unicos), np.sort(np.array( ['Mucho peor' 'Peor' 'Sin cambios' 'Mejor' 'Mucho mejor']))):
            top_labels = ['Mucho peor' 'Peor' 'Sin cambios' 'Mejor' 'Mucho mejor']
            valoresdecolumna = top_labels
        else:
            top_labels = valores_unicos
            valoresdecolumna = top_labels  
        colors = ['rgba(255, 51, 51, 0.8)', 'rgba(255, 153, 51, 0.8)',
                'rgba(255, 230, 51, 0.8)', 'rgba(218, 255, 51, 0.85)',
                'rgba(119, 255, 51, 1)']

        x_data = calcular_porcentajes_ocurrencias(df, columnas, valoresdecolumna)
        columnas.append("-Mean: ")
        y_data = [elemento.split("-")[-1] for elemento in columnas]


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
        title=f'{selected_grafico.capitalize()}: {titulo}'
    )
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
