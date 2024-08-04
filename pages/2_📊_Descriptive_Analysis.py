import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import io
import numpy as np
import base64
import plotly.figure_factory as ff
import re
import string
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter


if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Descriptive Analysis"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'nested_tab' not in st.session_state:
    st.session_state.nested_tab = "Various Graphics"
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = []
if 'last_fig' not in st.session_state:
    st.session_state['last_fig'] = None
if 'fig_width' not in st.session_state:
    st.session_state['fig_width'] = 1000
if 'fig_height' not in st.session_state:
    st.session_state['fig_height'] = 600  
if 'description' not in st.session_state:
    st.session_state['description'] = ""


st.set_page_config(page_title="Descriptive Analysis", page_icon="📊", layout="wide")

def agrupar_codigos(codigos):
    grupos = defaultdict(list)
    for codigo in codigos:
        parte_comun = codigo.split('_')[0]
        grupos[parte_comun].append(codigo)
    arreglo_final = [{'label': clave, 'value': '|'.join(codigos)}
                     for clave, codigos in grupos.items() if len(codigos) > 1]
    return arreglo_final
def process_text(text):
    # Convertir texto a minúsculas y eliminar signos de puntuación
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    
    # Dividir el texto en palabras
    words = text.split()
    
    # Filtrar las palabras que están en la lista de stopwords
    stop_words = set(stopwords.words('spanish'))  # Cambiar 'spanish' por el idioma de tu preferencia
    words = [word for word in words if word not in stop_words]
    return words

def generate_ngrams(words_list, n):
    return [' '.join(grams) for words in words_list for grams in ngrams(words, n)]

def create_wordcloud(ngram_words):
    word_counts = Counter(ngram_words)
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(word_counts)
    return wordcloud

def generate_wordcloud(text_list):
    if len(text_list) == 0:
        return None

    # Procesar cada frase por separado
    processed_texts = [process_text(text) for text in text_list]
    
    # Monogramas
    monogram_words = [word for words in processed_texts for word in words]
    monogram_wordcloud = create_wordcloud(monogram_words)
    
    # Bigramas
    bigram_words = generate_ngrams(processed_texts, 2)
    if len(bigram_words) > 0:
        bigram_wordcloud = create_wordcloud(bigram_words)
    
    # Trigramas
    trigram_words = generate_ngrams(processed_texts, 3)
    if len(trigram_words) > 0:
        trigram_wordcloud = create_wordcloud(trigram_words)
    
    # Crear la figura y agregar los tres wordclouds
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # Ajustar tamaño según sea necesario
    
    axs[0].imshow(monogram_wordcloud, interpolation='bilinear')
    axs[0].axis('off')
    axs[0].set_title('Monogram')
    
    if len(bigram_words) > 0:
        axs[1].imshow(bigram_wordcloud, interpolation='bilinear')
        axs[1].axis('off')
        axs[1].set_title('Bigram')
    
    if len(trigram_words) > 0:
        axs[2].imshow(trigram_wordcloud, interpolation='bilinear')
        axs[2].axis('off')
        axs[2].set_title('Trigram')
    
    # Convertir la figura a una imagen en base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return image_base64

# Función para generar wordcloud con filtro
def generate_wordcloud_with_filter(df, x_axis, filtro=None):
    if filtro!="No filter":
        unique_values = df[filtro].dropna().unique()
        num_subplots = len(unique_values)
        
        fig = make_subplots(
            rows=num_subplots, cols=1,
            subplot_titles=[f"{filtro}: {val}" for val in unique_values]
        )
        
        for i, val in enumerate(unique_values):
            filtered_df = df[df[filtro] == val]
            text_list = filtered_df[x_axis].dropna().tolist()
            image_base64 = generate_wordcloud(text_list)
            
            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{image_base64}",
                    xref="paper", yref="paper",
                    x=-1.7, y=5,
                    sizex=12, sizey=8,
                    xanchor="left", yanchor="top"
                ),
                row=i + 1, col=1
            )
        fig.update_layout(
            title_text=f"Wordcloud of {x_axis} Filtered by {filtro}",
            height=600 * num_subplots,
            showlegend=False
        )
        
        for i in range(num_subplots):
            fig.update_xaxes(visible=False, row=i + 1, col=1)
            fig.update_yaxes(visible=False, row=i + 1, col=1)
        
    else:
        text_list = df[x_axis].dropna().tolist()
        image_base64 = generate_wordcloud(text_list)
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{image_base64}",
                xref="x domain", yref="y domain",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top"
            )
        )
        fig.update_layout(
            title=f'Wordcloud of {x_axis}',
            xaxis={'visible': False},
            yaxis={'visible': False},
            width=1800,
            height=600
        )
    
    return fig

# Función para calcular los porcentajes en la tabla de contingencia
def calculate_percentages(table, x, y):
    row_percentages = table.div(table['Total'], axis=0)
    col_percentages = table.div(table.loc['Total'], axis=1)
    total_percentage = table / table.loc['Total', 'Total']
    
    output_table = table.copy().astype(str)
    for row in table.index:
        for col in table.columns:
            count = table.loc[row, col]
            if row != 'Total' and col != 'Total':
                row_perc = row_percentages.loc[row, col]
                col_perc = col_percentages.loc[row, col]
                total_perc = total_percentage.loc[row, col]
                #output_table.loc[row, col] = f"{count}  ({row_perc:.3f}, {col_perc:.3f}, {total_perc:.3f})"
                output_table.loc[row, col] = f"{count}"
                if totalper or rowper or colper:
                    output_table.loc[row, col] +=f" ("
                    dos = False
                    if totalper:
                        output_table.loc[row, col] +=f"{total_perc:.3f}"
                        dos = True
                    if rowper:
                        output_table.loc[row, col] += f"{', ' if dos else ''}{row_perc:.3f}"
                        dos=True
                    if colper:
                        output_table.loc[row, col] += f"{', ' if dos else ''}{col_perc:.3f}" 
                    output_table.loc[row, col] +=f")"                  
            else:
                output_table.loc[row, col] = f"{count}"
    return output_table
# Función para dividir texto en palabras
def split_text_by_words(text, num_words):
    words = text.split()
    return ' '.join(words[:num_words]) + '...'


# Función para calcular medias
def calcular_medias(df, nombres_columnas):
    medias = []
    mapeo = {
        'Totalmente en desacuerdo': 1,
        'Mucho peor': 1,
        'Muy negativo': 1,
        'En desacuerdo': 2,
        'Peor': 2,
        'Negativo': 2,
        'Neutral': 3,
        'Sin cambios': 3,
        'De acuerdo': 4,
        'Mejor': 4,
        'Positivo': 4,
        'Totalmente de acuerdo': 5,
        'Mucho mejor': 5,
        'Muy positivo': 5
    }

    for columna in nombres_columnas:
        df[columna] = df[columna].map(mapeo).fillna(df[columna])
        media_columna = df[columna].mean()
        media_columna = 100 * (media_columna - 1) / (df[columna].max() - 1)
        medias.append(media_columna)
    return medias

# Función para generar paleta de colores
def generate_color_palette(n):
    
    end_color = np.array([0, 57, 255, 0.8])  # Rojo
    start_color = np.array([1, 26, 109, 1])    # Verde
    colors = np.linspace(start_color, end_color, n)
    rgba_colors = [f'rgba({int(r)}, {int(g)}, {int(b)}, {a:.2f})' for r, g, b, a in colors]
    return rgba_colors

# Función para calcular porcentajes de ocurrencias
def calcular_porcentajes_ocurrencias(df, nombres_columnas, toplabels):
    porcentajes = []
    for columna in nombres_columnas:
        porcentajes_columna = []
        total_valores = df[columna].count()
        for valor in toplabels:
            porcentaje = round(df[columna].value_counts(normalize=True).get(valor, 0) * 100, 2)
            porcentajes_columna.append(porcentaje)
        porcentajes.append(porcentajes_columna)
    return porcentajes

# Función para generar gráfico de barras horizontales
def generate_horizontal_bar_chart(df, column_names, filtro):
    if filtro!="No filter":
        unique_values = df[filtro].dropna().unique()
        num_subplots = len(unique_values)
        fig = make_subplots(
            rows=num_subplots, cols=1,
            subplot_titles=[f"{filtro}: {val}" for val in unique_values]
        )
        
        for i, val in enumerate(unique_values):
            filtered_df = df[df[filtro] == val]
            frequencies = []
            for col in column_names:
                total_rows = filtered_df.shape[0]
                word_count = filtered_df[col].notna().sum()
                frequency = (word_count / total_rows) * 100
                frequencies.append(frequency)
            fig.add_trace(
                go.Bar(
                    x=frequencies,
                    y=[elemento.split("_")[-1] for elemento in column_names],
                    orientation='h',
                    text=[f'{freq:.1f}%' for freq in frequencies],
                    textposition='auto'
                ),
                row=i + 1, col=1
            )

        fig.update_layout(
            height=150+ 350 * num_subplots * len(column_names) / 8,
            showlegend=False,
            title=f'{column_names[0].split("_")[0]}'
        )

        for i in range(num_subplots):
            fig['layout'][f'yaxis{i + 1}'].update(autorange='reversed')
    else:
        frequencies = []
        for col in column_names:
            total_rows = df.shape[0]
            word_count = df[col].notna().sum()
            frequency = (word_count / total_rows) * 100
            frequencies.append(frequency)
        
        fig = go.Figure(go.Bar(
            x=frequencies,
            y=[elemento.split("_")[-1] for elemento in column_names],
            orientation='h',
            text=[f'{freq:.1f}%' for freq in frequencies],
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title='Proportion (%)',
            yaxis=dict(autorange="reversed"),
            height=150 + 250 * len(column_names) / 8,
            title=f'{column_names[0].split("_")[0]}'
        )

    return fig

# Función para generar gráfico radar
def generate_radar_chart(df, columnas, filtro):
    unique_filter_values = df[filtro].unique() if filtro!="No filter" else ["nada"]
    num_subplots = len(unique_filter_values)
    
    fig = make_subplots(rows=(num_subplots+1)//2, cols=2, specs=[[{'type': 'polar'}, {'type': 'polar'}] for _ in range((num_subplots+1)//2)])
    
    for i, valor in enumerate(unique_filter_values):
        filtered_df = df[df[filtro] == valor] if filtro!="No filter" else df
        medias_columnas = calcular_medias(filtered_df, columnas)
        y_data = ["~"+elemento.split("_")[-1] for elemento in columnas]
        fig.add_trace(go.Scatterpolar(
            r=medias_columnas,
            theta=y_data,
            fill='toself',
            name=f'Filter: {valor}' if filtro!="No filter" else 'Radar chart'
        ), row=(i+2)//2, col=i%2+1)
        
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
        height=(num_subplots+1)//2 * 500,
        width=1400,
        showlegend=True,
        title_text="Radar Chart",
    )
    if filtro=="No filter":
        fig.update_layout(
            showlegend=False,
        )
    return fig
def show_graph_and_table():
    if st.session_state['last_fig']:
        #st.plotly_chart(st.session_state['last_fig'])
        description = st.text_input("Add description", st.session_state['description'], key='description_input')
        st.session_state['description'] = description

        if st.button("Add graph to report"):
            st.session_state['report_data'].append({'figure': st.session_state['last_fig'], 'description': description, 'height': st.session_state['fig_height']})
            st.success("Graph added to report")

# Función para generar gráfico de barras múltiples
def generate_multibar_chart(df, columnas, filtro, likert):
    unique_filter_values = df[filtro].unique() if filtro!="No filter" else ["nada"]
    num_subplots = len(unique_filter_values)
    
    fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True)
    annotations = []
    numerosmode = False
    valores_unicos = df[columnas[0]].unique()
    valores_unicos = np.array(["No answer" if pd.isna(x) else x for x in valores_unicos])
    
    if valores_unicos.dtype == np.int64:
        numerosmode = True
        valoresdecolumna = sorted(valores_unicos)
        if likert=="Agreement (5)":
            top_labels = ["Strongly disagree","Disagree", "Neutral", "Agree", "Strongly agree"]
        elif likert=="Agreement (6)":
            top_labels = ["Disagree Very Strongly", "Disagree Strongly", "Disagree", "Agree","Agree Strongly","Agree Very Strongly"] 
        elif likert=="Quality (5)":
            top_labels = ["Very Poor","Below average", "Average", "Above average", "Excellent"]
        elif likert=="Frequency (6)":
            top_labels = [ "Never", "Very rarely", "Rarely","Occacionally","Very frequently","Always"] 
        elif likert=="Frequency (5)":
            top_labels = [ "Never", "Rarely", "Sometimes","Very Often","Always"] 
        else:
            numerosmode = False
            top_labels = [str(numero) for numero in valoresdecolumna]                       
    elif np.array_equal(np.sort(valores_unicos), np.sort(np.array(["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"]))):
        top_labels = ["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"]
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
        y_data = [elemento.split("_")[-1] for elemento in columnas]
        ##y_data.append("Mean")
        #Para cada porcentaje de una sola barra, normalmente serian 5 porcentajes
        for j in range(len(x_data[0])):
            #Ya depende del tamaño de la matrix incluyendo la media
            for xd, yd in zip(x_data, y_data):
                auxyd = yd
                yd = str(yd)
                fig.add_trace(go.Bar(
                    x=[xd[j]], 
                    y=["~"+yd],
                    orientation='h',
                    marker=dict(
                        color=colors[j],
                        line=dict(color='rgb(248, 248, 249)', width=1)
                    ),
                   
                ), row=i+1, col=1)
                texto = '*' if xd[0] < 4 else f'{xd[0]}%'
                # Anotaciones de porcentajes en el eje x
                annotations.append(dict(
                    xref=f'x{i+1}', yref=f'y{i+1}',
                    x=xd[0] / 2, y="~"+yd,
                    text=texto,
                    font=dict(family='Arial', size=14, color='rgb(255, 255, 255)'),
                    showarrow=False
                ))
                if auxyd == y_data[-1] and i==0:
                    #Determinar cual será la escala si se activó un encabezado custom (Solo primera aparición)
                    if numerosmode:
                        labelito = top_labels[valoresdecolumna[0]-1]
                    else:
                        labelito = top_labels[0]
                    color_square = f"<span style='color:{colors[0]}'>■</span>"
                     # Crear el texto con el cuadrado y el texto de la etiqueta
                    texto_con_color = f"{color_square} {labelito}"
                    annotations.append(dict(xref='x1', yref='y1',
                                            #x=xd[0] / 2 , 
                                            x=10, 
                                            y=len(y_data)+1,
                                            text=texto_con_color,
                                            font=dict(family='Arial', size=14,
                                                    color='rgb(67, 67, 67)'),
                                            showarrow=False))
                space = xd[0]
                for k in range(1, len(xd)):
                    texto = '*' if xd[k] < 4 else f'{xd[k]}%'
                    annotations.append(dict(
                        xref=f'x{i+1}', yref=f'y{i+1}',
                        x=space + (xd[k] / 2), y=f'~{yd}',
                        text=texto,
                        font=dict(family='Arial', size=14, color='rgb(255, 255, 255)'),
                        showarrow=False
                    ))
                    if auxyd == y_data[-1] and i==0:
                        #Determinar cual será la escala si se activó un encabezado custom (Apartir de segunda aparición)
                        if numerosmode:
                            labelito = top_labels[valoresdecolumna[k]-1]
                        else:
                            labelito = top_labels[k]
                        color_square = f"<span style='color:{colors[k]}'>■</span>"
                     # Crear el texto con el cuadrado y el texto de la etiqueta
                        texto_con_color = f"{color_square} {labelito}"
                        
                        annotations.append(dict(xref=f'x{i+1}', yref='y1',
                                                #x=space + (xd[k] / 2) , 
                                                x= 10+ k/len(top_labels)*100 , 
                                                y=len(y_data)+1,
                                                text=texto_con_color,
                                                font=dict(family='Arial', size=14,
                                                        color='rgb(67, 67, 67)'),
                                                showarrow=False))
                    space += xd[k]
        # Añadir anotación para el valor del filtro encima de cada subplot
        if (valor!= 'nada'):
            annotations.append(dict(
                xref= "paper",yref=f'y{i+1}',
                x=0.5, y=len(y_data),  # Ajustado para posicionar fuera del área de trazado
                text=f'Filter: {valor}',
                font=dict(family='Arial', size=16, color='rgb(67, 67, 67)'),
                showarrow=False, align='center'
            ))
    base_height = 210  # Altura base para una sola subtrama
    subplot_height = 150+ 200*(len(columnas)/6)  # Altura adicional por cada subtrama adicional
    total_height = base_height + subplot_height * (num_subplots)
    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        height=total_height,  
        annotations=annotations, 
        showlegend=False,
        margin=dict(l=120, r=10, t=140, b=80), 
    )
    return fig
def update_heatmap(selected_columns, filter_var, topvalues):
    if not selected_columns:
        return go.Figure(), 200
    
    first_col = selected_columns[0]
    if filter_var == "No filter":
        unique_filter_values = ["nada"]
    else:
        unique_filter_values = df[filter_var].unique()
        unique_filter_values_labels = unique_filter_values.astype(str)
    unicos = df[first_col].unique()
    unique_values = sorted(set(unicos))
    num_subplots = len(unique_filter_values)
    selected_columns = selected_columns[::-1]
    if unique_filter_values[0]=="nada":
        fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True)
    else:
        fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True, subplot_titles=unique_filter_values_labels)
    
    for i, valor in enumerate(unique_filter_values):
        filtered_df = df if valor == 'nada' else df[df[filter_var] == valor]
        heatmap_data = pd.DataFrame(columns=unique_values)
        heatmap_text = pd.DataFrame(columns=unique_values)
        
        for col in selected_columns:
            heatmap_data.loc[col] = 0
            for val in unique_values:
                count = (filtered_df[col] == val).sum()
                percentage = count / len(filtered_df) * 100
                heatmap_data.loc[col, val] = percentage
                heatmap_text.loc[col, val] = f"{percentage:.0f}% ({count})"
        
        y_ticktext =  heatmap_data.index
        
        fig.add_trace(go.Heatmap(
            z=heatmap_data.fillna(0).values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text=heatmap_text.fillna("").values,
            colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 0, 255)']],
            zmin=0,
            zmax=100,
            hoverinfo="text",
            showscale=True
        ), row=i+1, col=1)
        fig.update_yaxes(tickvals=list(range(len(heatmap_data.index))),
                         ticktext=y_ticktext, row=i+1, col=1)
        for ii, row in heatmap_data.iterrows():
            for j, val in row.items():
               
                fig.add_annotation(
                    text=f"{heatmap_text.loc[ii, j]}",
                    yref=f'y{i+1}',
                    xref=f'x{i+1}',
                    x=j,
                    y=ii,
                    showarrow=False,
                    font=dict(color="black", size=tamañoletra),
                    xanchor="center",
                    yanchor="middle"
                )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Percent",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]
        )
    )
    mapped_labels = [topvalues.get(val, val) for val in heatmap_data.columns]

    new_height = 200 + len(selected_columns) * 100 * num_subplots
    fig.update_layout(

            height=new_height,  
            font=dict(size = tamañoletra),
            
            yaxis=dict(
                
                tickfont=dict(size = tamañoletra),  # Color de las etiquetas del eje y
                titlefont=dict(size = tamañoletra+2)  # Color del título del eje y
            ),
            xaxis=dict(
                tickvals= heatmap_data.columns,
                ticktext= mapped_labels,
                tickfont=dict(size = tamañoletra),  # Color de las etiquetas del eje x
                titlefont=dict(size = tamañoletra+2)  # Color del título del eje x
            ),
            
            
    )
    return fig, new_height
# Function to update second dropdown options
def update_second_dropdown_options(selected_column):
    if selected_column is None:
        return [], True, True

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        min_base = df[selected_column].min()
        max_base = df[selected_column].max()
        
        matching_columns = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and 
            df[col].min() == min_base and 
            df[col].max() == max_base and 
            col != selected_column
        ]
    else:
        unique_values = set(df[selected_column].unique())
        
        matching_columns = [
            col for col in df.columns
            if set(df[col].unique()) == unique_values and col != selected_column
        ]

    options = [{'label': col, 'value': col} for col in matching_columns]
    return options, False, False
# Lista global para almacenar gráficos y descripciones
# Lista global para almacenar gráficos y descripciones




def generate_mean_polygon_chart(df, selected_x, selected_y, filter_var):
    if filter_var!="No filter":
        df_grouped = df.groupby([filter_var, selected_x]).agg({selected_y: 'mean'}).reset_index()
        fig = px.line(df_grouped, x=selected_x, y=selected_y, color=filter_var, markers=True)
    else:
        df_grouped = df.groupby(selected_x).agg({selected_y: 'mean'}).reset_index()
        fig = px.line(df_grouped, x=selected_x, y=selected_y, markers=True)
    
    fig.update_layout(
        height=500,
        width=1400,
        showlegend=True,
    )
    return fig
def generate_violin_chart(df, selected_x, selected_y, filter_var, show_points):
    if selected_x == "No variable":
        fig = px.violin(df, y=selected_y, box=show_box, points='all' if show_points else False)
        # Ocultar completamente el eje x
        fig.update_layout(xaxis_title=None, xaxis_showticklabels=False, xaxis=dict(showgrid=False, zeroline=False))
    else:
        if filter_var != "No filter":
            fig = px.violin(df, y=selected_y, x=selected_x, color=filter_var, box=True,
                            points='all' if show_points else False,
                            box_visible=show_box,
                            hover_data=df.columns)
        else:
            clases = df[selected_x].unique()
            fig = go.Figure()
            for clase in clases:
                fig.add_trace(go.Violin(x=df[selected_x][df[selected_x] == clase],
                                        y=df[selected_y][df[selected_x] == clase],
                                        name=str(clase),
                                        box_visible=show_box,
                                        meanline_visible=True,
                                        points='all' if show_points else False))
    
    fig.update_layout(
        height=500,
        width=1400,
        showlegend=True,
    )

    return fig
def generate_box_plot(df, selected_x, selected_y, filter_var, show_points):
    if selected_x == "No variable":
        fig = px.box(df, y=selected_y, points='all' if show_points else False)
        # Ocultar completamente el eje x
        fig.update_layout(xaxis_title=None, xaxis_showticklabels=False, xaxis=dict(showgrid=False, zeroline=False))
    else:
        if filter_var != "No filter":
            fig = px.box(df, y=selected_y, x=selected_x, color=filter_var,
                         points='all' if show_points else False,
                         hover_data=df.columns)
        else:
            clases = df[selected_x].unique()
            fig = go.Figure()
            for clase in clases:
                fig.add_trace(go.Box(x=df[selected_x][df[selected_x] == clase],
                                     y=df[selected_y][df[selected_x] == clase],
                                     name=str(clase),
                                     boxpoints='all' if show_points else 'outliers'))
    
    fig.update_layout(
        height=500,
        width=1400,
        showlegend=True,
    )

    return fig
def generate_ridgeline_chart(df, selected_x, selected_y, filter_var):
    if selected_x == "No variable":
        fig = px.violin(df, x=selected_y, box=False, points='all' if show_points else False)
        # Ocultar completamente el eje x
        fig.update_traces(orientation='h', side='positive', width=3, points=False)
        
        fig.update_layout(

            showlegend=False,
            title=f'Ridgeline chart: {selected_y}'
        )
    else:
        
        if filter_var!="No filter":
            unique_values = df[filter_var].unique()
            fig = make_subplots(rows=len(unique_values), cols=1, shared_yaxes=True)
            clases = df[selected_x].unique()
            colors = px.colors.sample_colorscale("Viridis", [n/len(clases) for n in range(len(clases))])

            for i, value in enumerate(unique_values):
                filtered_df = df[df[filter_var] == value]

                for clase, color in zip(clases, colors):
                    fig.add_trace(
                        go.Violin(
                            x=filtered_df[selected_y][filtered_df[selected_x] == clase],
                            line_color=color,
                            name=str(clase)
                        ), 
                        row=i+1, col=1
                    )

                fig.add_annotation(
                    dict(
                        text=f'{value}',
                        xref='paper', yref='paper',
                        x=0.5, y=1 - (i / len(unique_values)*1.06),
                        xanchor='center',
                        yanchor='bottom',
                        showarrow=False,
                        font=dict(size=14)
                    )
                )

                fig.update_xaxes(title_text=selected_y, row=i+1, col=1)
                fig.update_yaxes(title_text=selected_x, row=i+1, col=1)

            fig.update_traces(orientation='h', side='positive', width=3, points=False)

            fig.update_layout(
                height=len(unique_values) * 500,
                width=1400,
                showlegend=False,
                title=f'Ridgeline chart: {selected_x} vs {selected_y}'

            )

        else:
            fig = go.Figure()
            clases = df[selected_x].unique()
            colors = px.colors.sample_colorscale("Viridis", [n/len(clases) for n in range(len(clases))])
            for clase, color in zip(clases, colors):
                fig.add_trace(go.Violin(x=df[selected_y][df[selected_x] == clase], line_color=color, name=str(clase)))
            fig.update_traces(orientation='h', side='positive', width=3, points=False)
            fig.update_layout(xaxis_zeroline=False, yaxis_title=f'{selected_x}', xaxis_title=f'{selected_y}',)
    
    return fig
def generate_displot_chart(df, selected_x, selected_y, filter_var):
    if filter_var!="No filter":
        unique_values = df[filter_var].unique()
        fig = make_subplots(rows=len(unique_values), cols=1, shared_yaxes=True)

        for i, value in enumerate(unique_values):
            filtered_df1 = df[df[filter_var] == value]
            valoresx = filtered_df1[selected_x].unique().astype(str)
            hist_data = []

            for val in filtered_df1[selected_x].unique():
                filtered_df = filtered_df1[filtered_df1[selected_x] == val]
                hist_data.append(filtered_df[selected_y])

            distplot = ff.create_distplot(hist_data, valoresx, show_rug=False)
            for trace in distplot['data']:
                fig.add_trace(trace, row=i+1, col=1)
            fig.update_xaxes(title_text=selected_y, row=i+1, col=1)

            fig.add_annotation(
                dict(
                    text=f'{value}',
                    xref='paper', yref='paper',
                    x=0.5, y=1 - (i / len(unique_values)*1.06),
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(size=14)
                )
            )

        fig.update_layout(
            height=len(unique_values) * 500,
            width=1400,
            showlegend=True,
            xaxis_title=f'{selected_y}'
        )

    else:
        valoresx = df[selected_x].unique().astype(str)
        hist_data = []

        for value in df[selected_x].unique():
            filtered_df = df[df[selected_x] == value]
            hist_data.append(filtered_df[selected_y])

        fig = ff.create_distplot(hist_data, valoresx)
        fig.update_layout(
            height=500,
            width=1400,
            showlegend=True,
            xaxis_title=f'{selected_y}'
        )
    
    return fig



def update_numerico(selected_grafico, selected_x, selected_y, filter_var, show_points):
    if not selected_grafico or not selected_x or not selected_y:
        return go.Figure()

    if selected_grafico == 'Mean polygon chart':
        fig = generate_mean_polygon_chart(df, selected_x, selected_y, filter_var)
    
    elif selected_grafico == 'Violin chart':
        fig = generate_violin_chart(df, selected_x, selected_y, filter_var, show_points)
    
    elif selected_grafico == 'Ridgeline chart':
        fig = generate_ridgeline_chart(df, selected_x, selected_y, filter_var)
   
    elif selected_grafico == 'Displot chart':
        fig = generate_displot_chart(df, selected_x, selected_y, filter_var)
    elif selected_grafico == 'Box plot':
        fig = generate_box_plot(df, selected_x, selected_y, filter_var, show_points)
    else:
        return go.Figure()

    if selected_grafico in ['Mean polygon chart', 'Violin chart']:
        if (selected_x != "No variable"):
            fig.update_layout(
                xaxis_title=f'{selected_x}',
                yaxis_title=f'{selected_y}',
                title=f'{selected_grafico.capitalize()}: {selected_x} vs {selected_y}'
            )
        else:
            fig.update_layout(
                yaxis_title=f'{selected_y}',
                title=f'{selected_grafico.capitalize()}: {selected_y}'
            )     
    return fig



df = st.session_state.df
if df is None:
    st.error("First Upload your CSV File.")
    st.stop()
columns =st.session_state.df.columns.tolist()
opciones_especiales = agrupar_codigos(columns)
st.sidebar.header("Choose the type of graphics")
nested_tab_options = ["Categorical charts", "Matrix Charts", "Custom Heatmaps", "Numeric Charts"]
nested_tab = st.sidebar.radio("Subsections", nested_tab_options, index=nested_tab_options.index(st.session_state.nested_tab) if st.session_state.nested_tab in nested_tab_options else 0)

st.header("📊 Descriptive Analysis")
# Tabs anidados dentro de 'Descriptive Analysis'
if nested_tab == "Categorical charts":
    st.subheader("Categorical charts")
    st.markdown("Use this type of graph to visualise the behaviour between any of your categorical variables.")
    if st.checkbox("Show instructions"):
        st.markdown("**Instructions:**")
        st.markdown("1. Select what chart you want")
        st.markdown("2. Select your first categorical variable (If you selected a word cloud chart, make sure this variable corresponds to an open-ended question).")
        st.markdown("3. If you selected to graph a contingency table, choosing a filter is mandatory to act as a second variable,\n   if you chose another graph it will be an optional variable to visualise how change your first variable according to the chosen filter.")
        st.markdown("4. Click on Sumbit to visualise your graphic")
    container1 = st.container(border=True)
    tipo_grafico = container1.selectbox("Choose your chart:", ['Bar Chart', 'Pie Chart', 'Contingency Table', 'Word Cloud'])
    x_axis = container1.selectbox("Choose your variable:",[""]+ columns)
    filtro = container1.selectbox("Choose your filter:",["No filter"]+ columns)
    if tipo_grafico == 'Bar Chart':
        hori = container1.checkbox("Horizontal bars")
    if tipo_grafico != 'Word Cloud':
        likert = container1.selectbox("Choose your likert scale:", ['Original', 'Agreement (5)', 'Agreement (6)', 'Quality (5)', 'Frequency (5)', 'Frequency (6)'])
        if likert=="Agreement (5)":
            top_labels = {1: "Strongly disagree",2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
        elif likert=="Agreement (6)":
            top_labels = {1: "Disagree Very Strongly", 2: "Disagree Strongly", 3: "Disagree", 4: "Agree",5: "Agree Strongly",6: "Agree Very Strongly"} 
        elif likert=="Quality (5)":
            top_labels = {1: "Very Poor",2: "Below average", 3: "Average", 4: "Above average", 5: "Excellent"}
        elif likert=="Frequency (6)":
            top_labels = {1:  "Never", 2: "Very rarely", 3: "Rarely",4: "Occacionally",5: "Very frequently",6: "Always"}
        elif likert=="Frequency (5)":
            top_labels = {1:  "Never", 2: "Rarely",3:  "Sometimes",4: "Very Often",5: "Always"}
        else:
            top_labels={}
    if (filtro!="No filter" and tipo_grafico != "Wordcloud"):
        container1.markdown("Choose what percentage you want to see (order: N/Table %, N/row %, N/column %):")
        totalper = container1.checkbox("N/Table total %", True)
        rowper = container1.checkbox("N/Row total %")
        colper = container1.checkbox("N/Column total %")
    tamañoletra = container1.number_input(label= "Select the size of the chart's font", value= 16)
    if container1.button("Submit"):
        if tipo_grafico and x_axis:
            if tipo_grafico == 'Bar Chart':
                aux_df = df.copy()
                aux_df[x_axis].fillna("No answer", inplace=True)
                aux_df[x_axis] = aux_df[x_axis].map(lambda x: top_labels[x] if x in top_labels else x)
                fig = go.Figure()
                data_table = []
                
                if filtro != "No filter":
                    filtered_dfs = aux_df.groupby(filtro)
                    for filter_value, filtered_df in filtered_dfs:
                        counts = filtered_df[x_axis].value_counts()
                        total_counts = counts.sum()
                        percentages = (counts / total_counts) * 100
                        if hori:
                            fig.add_trace(go.Bar(
                            x=counts,
                            y=counts.index,
                            name=str(filter_value),
                            text=[f'{p:.1f}%' for p in percentages],
                            textfont=dict(size=tamañoletra),
                            textposition='auto',
                            orientation="h"
                            ))
                        else:
                            fig.add_trace(go.Bar(
                            x=counts.index,
                            y=counts,
                            name=str(filter_value),
                            text=[f'{p:.1f}%' for p in percentages],
                            textfont=dict(size=tamañoletra),
                            textposition='auto',
                            orientation="v"
                            ))
    
                        data_table.extend(filtered_df[[x_axis, filtro]].to_dict('records'))
    
                    fig.update_layout(
                        barmode='group',
                        title=f"{x_axis} Filtered by {filtro}",
                        xaxis_title=x_axis,
                        xaxis_title_font=dict(size=tamañoletra+2),  # Tamaño del título del eje x
                        yaxis_title='Frequency',
                        yaxis_title_font=dict(size=tamañoletra+2),  # Tamaño del título del eje y
                        xaxis=dict(
                            tickfont=dict(size=tamañoletra)  # Tamaño de la fuente de las etiquetas del eje x
                        ),
                        yaxis=dict(
                            tickfont=dict(size=tamañoletra)  # Tamaño de la fuente de las etiquetas del eje y
                        ),
                        height=700
                    )
                    contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                    output_table = calculate_percentages(contingency_table, x_axis, filtro)
                    data_table = output_table.reset_index().to_dict('records')
                else:
                    counts = aux_df[x_axis].value_counts()
                    total_counts = counts.sum()
                    percentages = (counts / total_counts) * 100
                   
                    if hori:
                        fig.add_trace(go.Bar(
                        x=counts,
                        y=counts.index,
                        text=[f'{p:.1f}%' for p in percentages],
                        textfont=dict(size=tamañoletra),
                        textposition='auto',
                        orientation="h"
                        ))
                    else:
                        fig.add_trace(go.Bar(
                        x=counts.index,
                        y=counts,
                        text=[f'{p:.1f}%' for p in percentages],
                        textfont=dict(size=tamañoletra),
                        textposition='auto',
                        orientation="v"
                        ))
                    fig.update_layout(
                        height=700,
                        title="Bar chart of "+str(x_axis),
                        xaxis_title=x_axis,
                        xaxis_title_font=dict(size=tamañoletra+2),  # Tamaño del título del eje x
                        yaxis_title='Frequency',
                        yaxis_title_font=dict(size=tamañoletra+2),  # Tamaño del título del eje y
                        xaxis=dict(
                            tickfont=dict(size=tamañoletra)  # Tamaño de la fuente de las etiquetas del eje x
                        ),
                        yaxis=dict(
                            tickfont=dict(size=tamañoletra)  # Tamaño de la fuente de las etiquetas del eje y
                        ),
                        
                    )
                    counts = aux_df[x_axis].value_counts().reset_index()
                    counts.columns = [x_axis, 'count']
                    data_table = counts.to_dict('records')
    
            elif tipo_grafico == 'Pie Chart':
                aux_df = df.copy()
                aux_df[x_axis].fillna("No answer", inplace=True)
                aux_df[x_axis] = aux_df[x_axis].map(lambda x: top_labels[x] if x in top_labels else x)
                fig = go.Figure()
                data_table = []
    
                if filtro!="No filter":
                    unique_values = aux_df[filtro].unique()
                    fig = make_subplots(rows=1, cols=len(unique_values), specs=[[{'type': 'domain'}]*len(unique_values)],  subplot_titles=[f"{filtro}: {val}" for val in unique_values])
                    
    
                    for i, value in enumerate(unique_values):
                        filtered_df = aux_df[aux_df[filtro] == value]
                        fig.add_trace(
                            go.Pie(labels=filtered_df[x_axis].value_counts().index,textfont=dict(size=tamañoletra), values=filtered_df[x_axis].value_counts().values, name=str(value)),
                            row=1, col=i+1
                        )
    
                    contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                    output_table = calculate_percentages(contingency_table, x_axis, filtro)
                    data_table = output_table.reset_index().to_dict('records')
    
                    fig.update_layout(
                        title=f"{x_axis} Filtered by {filtro}",
                    )
                    fig.update_traces(
                        textinfo='label+percent',  # Muestra etiquetas y porcentajes
                        textfont=dict(size=tamañoletra)  # Cambia este tamaño de letra según sea necesario
                    )
                else:
                    fig = px.pie(aux_df, names=x_axis)
                    fig.update_layout(
                        title={
                            'text': "Pie chart of " + str(x_axis),
                            'font': dict(size=tamañoletra+2)  # Cambia este tamaño de letra según sea necesario
                        },
                        legend=dict(
                            font=dict(size=tamañoletra)  # Cambia este tamaño de letra según sea necesario para la leyenda
                        ),
                        margin=dict(t=50, b=50, l=50, r=50)  # Ajusta los márgenes si es necesario
                    )
                    # Actualizar el diseño de las etiquetas de los segmentos (textos dentro del pastel)
                    fig.update_traces(
                        textinfo='label+percent',  # Muestra etiquetas y porcentajes
                        textfont=dict(size=tamañoletra)  # Cambia este tamaño de letra según sea necesario
                    )
                    counts = aux_df[x_axis].value_counts().reset_index()
                    counts.columns = [x_axis, 'count']
                    data_table = counts.to_dict('records')
    
            elif tipo_grafico == 'Contingency Table':
                fig = go.Figure()
                data_table = []
                if filtro!="No filter":
                    
                    contingency_table = pd.crosstab(df[x_axis], df[filtro])
                    # Calcular los porcentajes
                    total_sum = contingency_table.values.sum()
                    row_sum = contingency_table.sum(axis=1).values[:, None]
                    col_sum = contingency_table.sum(axis=0).values
                    
                    percent_total = contingency_table / total_sum * 100
                    percent_row = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
                    percent_col = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
                    
                    # Crear anotaciones de texto
                    annotations = []
                    for i, row in enumerate(contingency_table.index):
                        for j, col in enumerate(contingency_table.columns):
                            count = contingency_table.loc[row, col]
                            pct_total = percent_total.loc[row, col]
                            pct_row = percent_row.loc[row, col]
                            pct_col = percent_col.loc[row, col]
                            porcentages = f"{count}"
                            cell_value = contingency_table.iloc[i, j]
                            threshold = contingency_table.values.max() / 2  # Ajustar este umbral según sea necesario
                            text_color = 'white' if cell_value > threshold else 'black'
                            if totalper or rowper or colper:
                                porcentages+=f"<br>"
                                dos = False
                                if totalper:
                                    porcentages +=f"{pct_total:.1f}%"
                                    dos = True
                                if rowper:
                                    porcentages += f"{', ' if dos else ''}{pct_row:.1f}%"
                                    dos=True
                                if colper:
                                    porcentages += f"{', ' if dos else ''}{pct_col:.1f}%" 
                                
                           
                            annotations.append(go.layout.Annotation(
                                text=f'<b>{porcentages}</b>',
                                x=col,
                                y=row,
                                xref='x1',
                                yref='y1',
                                showarrow=False,
                                font=dict(color=text_color, size = tamañoletra),
                                
                                
                            ))
                    
                    # Crear el heatmap
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=contingency_table.values,
                        x=contingency_table.columns,
                        y=contingency_table.index,
                        colorscale='Blues',
                        text=[[f'{contingency_table.iloc[i, j]}<br>{percent_total.iloc[i, j]:.1f}%<br>{percent_row.iloc[i, j]:.1f}%<br>{percent_col.iloc[i, j]:.1f}%' for j in range(contingency_table.shape[1])] for i in range(contingency_table.shape[0])],
                        hoverinfo='text'
                    ))
                    #aux_df[x_axis] = aux_df[x_axis].map(lambda x: top_labels[x] if x in top_labels else x)
                    # Agregar anotaciones
                    mapped_labels = [top_labels.get(val, val) for val in contingency_table.index]
                    mapped_labels2 = [top_labels.get(val, val) for val in contingency_table.columns]
                    fig.update_layout(
                        title=f'Contingency table between {x_axis} and {filtro}',
                        xaxis_title=filtro,
                        yaxis_title=split_text_by_words(f'{str(x_axis)}', 9),
                        font=dict(size = tamañoletra),
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(contingency_table.columns),
                            ticktext=mapped_labels2,
                            tickfont=dict(size = tamañoletra),  # Color de las etiquetas del eje x
                            titlefont=dict(size = tamañoletra+2)  # Color del título del eje x
                        ),
                        yaxis=dict(
                            tickmode='array',
                            tickvals=list(contingency_table.index),
                            ticktext=mapped_labels,
                            tickfont=dict(size = tamañoletra),  # Color de las etiquetas del eje y
                            titlefont=dict(size = tamañoletra+2)  # Color del título del eje y
                        ),
                        annotations=annotations
                    )
                    contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                    output_table = calculate_percentages(contingency_table, x_axis, filtro)
                    # Renombrar la primera columna
                    output_table = output_table.reset_index()
                    # Renombrar la segunda columna (que es la primera columna después de reset_index)
                    new_column_name = x_axis+"/"+filtro # Cambia esto al nuevo título que desees
                    output_table.rename(columns={output_table.columns[0]: new_column_name}, inplace=True)
                    # Convertir el DataFrame a un diccionario
                    data_table = output_table.to_dict('records')
                else:
                    st.error("Por favor seleccione un filtro para generar la tabla de contingencia.")
    
            elif tipo_grafico == 'Word Cloud':
                fig = generate_wordcloud_with_filter(df, x_axis, filtro)
                
                data_table = df[[x_axis, filtro]].dropna().to_dict('records') if filtro!="No filter" else df[[x_axis]].dropna().to_dict('records')
    
            else:
                st.error("Tipo de gráfico no reconocido.")
            if fig:
                st.session_state['last_fig'] = fig
                st.session_state['fig_width'] = fig.layout.width
                st.session_state['fig_height'] = fig.layout.height
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(data_table), hide_index=True)
    
    show_graph_and_table()
elif nested_tab == "Matrix Charts":
    st.subheader("Matrix Charts")
    st.markdown("Use this type of graph to visualise the behaviour between a set of categorical variables that are related to a similar question (these sets are automatically recognised).")
    if st.checkbox("Show instructions"):
        st.markdown("**Instructions:**")
        st.markdown("1. Select what chart you want (Tick Bar chart: please note that this type of chart is for multiple choice questions only)")
        st.markdown("2. Select your set of categorical variables ")
        st.markdown("3. You can optionally choose a filter that will allow you to visualise the change of your set of variables according to the chosen filter.")
        st.markdown("4. If you have chosen the Multibar chart option and in addition your set of variables are numerically coded, you can change your scale to Likert.")
        st.markdown("5. Click on Sumbit to visualise your graphic")
    container2 = st.container(border=True)
    chart_type = container2.selectbox("Choose your chart:", [ 'Multi bar Chart', 'Radar Chart','Tick bar Chart'])
    variable_group = container2.selectbox("Choose your group of variables (Matrix):",[""]+ opciones_especiales, format_func=lambda option: option['label'] if option else None)
    filter_var = container2.selectbox("Choose your filter:",["No filter"]+ columns)
    likert_scale = container2.selectbox("Choose your likert scale (Only Multi plot):", ['Original', 'Agreement (5)', 'Agreement (6)', 'Quality (5)', 'Frequency (5)', 'Frequency (6)'])
    tamañoletra = container2.number_input(label= "Select the size of the chart's font", value= 16)
    if container2.button("Submit"):
        if variable_group:
            columnas = variable_group['value'].split("|")
            fig = None
            if chart_type == 'Tick bar Chart':
                fig = generate_horizontal_bar_chart(df, columnas, filter_var)
            elif chart_type == 'Radar Chart':
                fig = generate_radar_chart(df, columnas, filter_var)
            elif chart_type == 'Multi bar Chart':
                st.markdown("Note: the label * mean less than 4%")
                fig = generate_multibar_chart(df, columnas, filter_var, likert_scale)
            if fig:
                st.session_state['last_fig'] = fig
                st.session_state['fig_width'] = fig.layout.width
                st.session_state['fig_height'] = fig.layout.height
                st.plotly_chart(fig)
            if filter_var!="No filter":
                if chart_type == "Tick bar Chart":
                    unique_filter_values = df[filter_var].unique()
                    orden_escala = sorted(df[columnas].stack().unique().tolist())
                    for i, valor in enumerate(unique_filter_values):
                        datos = []
                        st.subheader(f"Table for filter {valor}")
                        filtered_df = df[df[filter_var] == valor]
                        for col in columnas:
                            total_rows = filtered_df.shape[0]
                            word_count = filtered_df[col].notna().sum()
                            frequency = (word_count / total_rows) * 100
                            datos.append([col.split("_")[-1], word_count, frequency])
                        # Crear un DataFrame con los datos calculados
                        df_resultado = pd.DataFrame(datos, columns=[columnas[0].split("_")[0], 'Count', '% Of total'])
                        df_resultado['% Of total'] = df_resultado['% Of total'].apply(lambda x: f"{x:.3f}%")
                        # Mostrar el DataFrame en Streamlit
                        st.dataframe(df_resultado)
                else:
                    unique_filter_values = df[filter_var].unique()
                    orden_escala = sorted(df[columnas].stack().unique().tolist())
                    for i, valor in enumerate(unique_filter_values):
                        st.subheader(f"Crosstab for filter {valor}")
                        filtered_df = df[df[filter_var] == valor]
                        conteos_columnas = []
                        conteos_columnas.append(pd.DataFrame({'Scale': orden_escala}))
                        # Obtener el orden de la escala de la primera columna
                        for columna in columnas:
                            counts = filtered_df[columna].value_counts().reset_index()
                            counts.columns = [columna, 'count']
                            # Calcular el porcentaje
                            total = counts['count'].sum()
                            counts['percentage'] = (counts['count'] / total * 100).round(2)
                            # Ordenar los conteos de acuerdo al orden de la escala determinado previamente
                            counts = counts.set_index(counts.columns[0])
                            # Asegurar que los índices a ordenar estén presentes en la escala y viceversa
                            counts = counts.reindex(orden_escala).fillna(0).reset_index()
                            counts['formatted'] = counts.apply(lambda row: f"{int(row['count'])} ({row['percentage']}%)", axis=1)
                            counts = counts[['formatted']].rename(columns={'formatted': columna})
                            conteos_columnas.append(counts)
                        # Concatenar los conteos en columnas manteniendo el orden
                        data_tables = pd.concat(conteos_columnas, axis=1)
                        data_table = data_tables.to_dict('records')
                        st.dataframe(pd.DataFrame(data_table))
            else:
                if chart_type == 'Tick bar Chart':
                    datos = []
                    for col in columnas:
                        total_rows = df.shape[0]
                        word_count = df[col].notna().sum()
                        frequency = (word_count / total_rows) * 100
                        datos.append([col.split("_")[-1], word_count, frequency])
                    # Crear un DataFrame con los datos calculados
                    df_resultado = pd.DataFrame(datos, columns=[columnas[0].split("_")[0], 'Count', '% Of total'])
                    df_resultado['% Of total'] = df_resultado['% Of total'].apply(lambda x: f"{x:.3f}%")
                    # Mostrar el DataFrame en Streamlit
                    st.dataframe(df_resultado)
                else:
                    orden_escala = sorted(df[columnas].stack().unique().tolist())
                    conteos_columnas = []
                    # Incluir la columna de escala como primer elemento de conteos_columnas
                    conteos_columnas.append(pd.DataFrame({'Scale': orden_escala}))
                    for columna in columnas:
                        counts = df[columna].value_counts().reset_index()
                        counts.columns = [columna, 'count']
                        # Calcular el porcentaje
                        total = counts['count'].sum()
                        counts['percentage'] = (counts['count'] / total * 100).round(2)
                        # Ordenar los conteos de acuerdo al orden de la escala determinado previamente
                        counts = counts.set_index(counts.columns[0])
                        counts = counts.reindex(orden_escala).fillna(0).reset_index()
                        counts['formatted'] = counts.apply(lambda row: f"{int(row['count'])} ({row['percentage']}%)", axis=1)
                        counts = counts[['formatted']].rename(columns={'formatted': columna})
                        conteos_columnas.append(counts)
                    # Concatenar los conteos en columnas manteniendo el orden
                    data_tables = pd.concat(conteos_columnas, axis=1)
                    data_table = data_tables.to_dict('records')
                    st.dataframe(pd.DataFrame(data_table))
    # Llama a la función para mostrar la gráfica y los botones de reportes si ya se ha generado una gráfica
    show_graph_and_table()
elif nested_tab == "Custom Heatmaps":
    st.subheader("Custom Heatmap")
    st.markdown("Use this type of graph to visualise behaviour among a set of categorical variables that you can choose manually, the set of questions must have the same scale.")
    if st.checkbox("Show instructions"):
        st.markdown("**Instructions:**")
        st.markdown("1. Select your first categorical variable (Note that the other variables you can select will depend on the scale of this variable.) ")
        st.markdown("2. You can add all variables that have the same scale, by selecting the variable in the second dropdown and pressing the add button.")
        st.markdown("3. You can optionally choose a filter that will allow you to visualise the change of your set of variables according to the chosen filter.")
        st.markdown("4. Click on Sumbit to visualise your graphic")
    container3 = st.container(border=True)
    var1 = container3.selectbox("Choose your first variable:", [""]+columns)
    selected_columns = st.session_state.get('selected_columns', [var1])
    if var1:
        options, var2_disabled, add_disabled = update_second_dropdown_options(var1)
    else:
        options, var2_disabled, add_disabled = [], True, True
    selected_columns= container3.multiselect("Choose your additional variable:", [opt['label'] for opt in options], disabled=var2_disabled)
    selected_columns.insert(0, var1)
    filter_var = container3.selectbox("Select your filter:", ["No filter"]+columns)
    tamañoletra = container3.number_input(label= "Select the size of the chart's font", value= 16)
    likert = container3.selectbox("Choose your likert scale:", ['Original', 'Agreement (5)', 'Agreement (6)', 'Quality (5)', 'Frequency (5)', 'Frequency (6)'])
    if likert=="Agreement (5)":
        top_labels = {1: "Strongly disagree",2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
    elif likert=="Agreement (6)":
        top_labels = {1: "Disagree Very Strongly", 2: "Disagree Strongly", 3: "Disagree", 4: "Agree",5: "Agree Strongly",6: "Agree Very Strongly"} 
    elif likert=="Quality (5)":
        top_labels = {1: "Very Poor",2: "Below average", 3: "Average", 4: "Above average", 5: "Excellent"}
    elif likert=="Frequency (6)":
        top_labels = {1:  "Never", 2: "Very rarely", 3: "Rarely",4: "Occacionally",5: "Very frequently",6: "Always"}
    elif likert=="Frequency (5)":
        top_labels = {1:  "Never", 2: "Rarely",3:  "Sometimes",4: "Very Often",5: "Always"}
    else:
        top_labels={}
    show_table = container3.button("Sumbit")

    if show_table:
        heatmap_fig, heatmap_height = update_heatmap(selected_columns, filter_var, top_labels)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        if heatmap_fig:
            st.session_state['last_fig'] = heatmap_fig
            st.session_state['fig_width'] = heatmap_fig.layout.width
            st.session_state['fig_height'] = heatmap_fig.layout.height
        if filter_var != "No filter":
            filas_tabla = []
            unique_filter_values = df[filter_var].unique()
            orden_escala = sorted(df[selected_columns].stack().unique().tolist())
            
            for i, valor in enumerate(unique_filter_values):
                st.subheader(f"Crosstab for filter {valor}")
                filtered_df = df[df[filter_var] == valor]
                conteos_columnas = []
                conteos_columnas.append(pd.DataFrame({'Scale': orden_escala}))
                # Obtener el orden de la escala de la primera columna
                for columna in selected_columns:
                    counts = filtered_df[columna].value_counts().reset_index()
                    counts.columns = [columna, 'count']
                    # Calcular el porcentaje
                    total = counts['count'].sum()
                    counts['percentage'] = (counts['count'] / total * 100).round(2)
                    # Ordenar los conteos de acuerdo al orden de la escala determinado previamente
                    counts = counts.set_index(counts.columns[0])
                    # Asegurar que los índices a ordenar estén presentes en la escala y viceversa
                    counts = counts.reindex(orden_escala).fillna(0).reset_index()
                    counts['formatted'] = counts.apply(lambda row: f"{int(row['count'])} ({row['percentage']}%)", axis=1)
                    counts = counts[['formatted']].rename(columns={'formatted': columna})
                    conteos_columnas.append(counts)
                # Concatenar los conteos en columnas manteniendo el orden
                data_tables = pd.concat(conteos_columnas, axis=1)
                data_table = data_tables.to_dict('records')
                st.dataframe(pd.DataFrame(data_table), hide_index=True)
        else:
            orden_escala = sorted(df[selected_columns].stack().unique().tolist())
            
            conteos_columnas = []
            
            # Incluir la columna de escala como primer elemento de conteos_columnas
            conteos_columnas.append(pd.DataFrame({'Scale': orden_escala}))
            
            for columna in selected_columns:
                counts = df[columna].value_counts().reset_index()
                counts.columns = [columna, 'count']
                
                # Calcular el porcentaje
                total = counts['count'].sum()
                counts['percentage'] = (counts['count'] / total * 100).round(2)
                
                # Ordenar los conteos de acuerdo al orden de la escala determinado previamente
                counts = counts.set_index(counts.columns[0])
                counts = counts.reindex(orden_escala).fillna(0).reset_index()
                counts['formatted'] = counts.apply(lambda row: f"{int(row['count'])} ({row['percentage']}%)", axis=1)
                counts = counts[['formatted']].rename(columns={'formatted': columna})
                
                conteos_columnas.append(counts)
            
            # Concatenar los conteos en columnas manteniendo el orden
            data_tables = pd.concat(conteos_columnas, axis=1)
            data_table = data_tables.to_dict('records')
            
            st.dataframe(pd.DataFrame(data_table), hide_index=True)
    show_graph_and_table()
elif nested_tab == "Numeric Charts":
    st.subheader("Numeric charts")
    st.markdown("Use this type of graph to visualise the behaviour between a categorical variable and a numerical variable.")
    if st.checkbox("Show instructions"):
        st.markdown("**Instructions:**")
        st.markdown("1. Select your first categorical variable")
        st.markdown("2. Select your numerical variable")
        st.markdown("3. You can optionally choose a categorical filter that will allow you to visualise the change of your variables according to the chosen filter.")
        st.markdown("4. Click on Sumbit to visualise your graphic")   
    container4 = st.container(border=True)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()        
    chart_type = container4.selectbox("Select your chart:", ['Box plot', 'Mean polygon chart', 'Violin chart', 'Ridgeline chart', 'Displot chart'])
    num_var = container4.selectbox("Select your numerical variable:", [""]+numeric_columns)
    cat_var = container4.selectbox("Select your group variable:", ["No variable"]+columns)
    filter_var = container4.selectbox("Select your filter (categoric):", ["No filter"]+columns)
    if chart_type == 'Violin chart':
        show_points = container4.checkbox("Show Points")
        show_box = container4.checkbox("Show Box")
        #show_ = st.checkbox("Show labels")
    else:
        show_points = False
    tamañoletra = container4.number_input(label= "Select the size of the chart's font", value= 16)
    show_table = container4.button("Sumbit")
    if show_table:
        fig = update_numerico(chart_type, cat_var, num_var, filter_var, show_points)
        st.plotly_chart(fig, use_container_width=True)
        if fig:
            st.session_state['last_fig'] = fig
            st.session_state['fig_width'] = fig.layout.width
            st.session_state['fig_height'] = fig.layout.height
    show_graph_and_table()
st.session_state.nested_tab = nested_tab
