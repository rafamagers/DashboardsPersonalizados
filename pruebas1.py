import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from plotly import graph_objs as go
import io
import numpy as np
import os
import base64
from datetime import datetime
from io import BytesIO
from googletrans import Translator
import plotly.figure_factory as ff
import nltk
import re
from scipy.stats import chi2
import string
import scipy
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer.rotator import Rotator
import pingouin as pg
from semopy import Model, Optimizer, semplot
from semopy.inspector import inspect
import semopy
from scipy.stats import norm, multivariate_normal

from RyStats.common import polychoric_correlation_serial
os.environ['PATH'] = f"{os.path.expanduser('~/R/bin')}:{os.environ['PATH']}"
os.environ['R_HOME'] = os.path.expanduser('~/R')
os.environ["PATH"] += os.pathsep + '/usr/bin'  # Reemplaza con la ruta correcta si es necesario
#from transformers import pipeline
#qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print(scipy.__version__)

# Inicializar el traductor
translator = Translator()
print()


# Estado inicial
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
# Función para cargar y procesar el archivo CSV
# Variables globales para almacenar el número de factores y los ítems seleccionados
# Variables globales para almacenar el número de factores y los ítems seleccionados
# Variables globales para almacenar el número de factores y los ítems seleccionados
if 'num_factors' not in st.session_state:
    st.session_state.num_factors = 2
if 'factor_items' not in st.session_state:
    st.session_state.factor_items = {f'Factor {i+1}': [] for i in range(st.session_state.num_factors)}

if 'models' not in st.session_state:
    st.session_state.models = []
if 'last_selected_factor' not in st.session_state:
    st.session_state.last_selected_factor = None
@st.cache_data
def load_data(uploaded_file,deli):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, keep_default_na=False, na_values=[""], delimiter=deli)
        question = "¿Cuántas filas y columnas tiene este DataFrame?"
        #answer = ask_question_about_df(df, question)
        print("Respuesta:")
        #print(answer)
        return df
    return None
def ask_question_about_df(df, question):
    # Convertir el DataFrame a texto
    df_text = df.to_string()

    # Preguntar sobre el DataFrame
    result = qa_pipeline({
        'question': question,
        'context': df_text
    })
    return result
# Función para traducir una columna a inglés
def translate_column(df, column):
    if df is not None and column in df.columns:
        unique_values = df[column].unique()
        translations = {val: translator.translate(val, src='es', dest='en').text if isinstance(val, str) else val for val in unique_values}
        df[column] = df[column].map(translations)
        new_column_name = translator.translate(column, src='es', dest='en').text
        df.rename(columns={column: new_column_name}, inplace=True)
        return f"Translated column '{column}' to English."
    return "No column selected or dataframe is empty."
def update_factor_items():
    for factor in st.session_state.factor_items.keys():
        st.session_state.factor_items[factor] = st.session_state.get(f'{factor}_selected_items', [])

def add_factor():
    st.session_state.num_factors += 1
    st.session_state.factor_items[f'Factor {st.session_state.num_factors}'] = []

def remove_factor():
    if st.session_state.num_factors > 2:
        del st.session_state.factor_items[f'Factor {st.session_state.num_factors}']
        st.session_state.num_factors -= 1
# Función para traducir todo el dataframe a inglés
def translate_dataframe(df):
    if df is not None:
        for col in df.columns:
            unique_values = df[col].unique()
            translations = {val: translator.translate(val, src='es', dest='en').text if isinstance(val, str) else val for val in unique_values}
            df[col] = df[col].map(translations)
            new_col_name = translator.translate(col, src='es', dest='en').text
            df.rename(columns={col: new_col_name}, inplace=True)
        return "Translated entire dataframe to English."
    return "Dataframe is empty."
def rename_columns(df, old_char, new_char):
    # Get current column names
    columns = df.columns.tolist()
    
    # Rename columns by replacing the first occurrence of old_char with new_char
    renamed_columns = [col.replace(old_char, new_char, 1) for col in columns]
    
    # Assign the new column names to the DataFrame
    df.columns = renamed_columns
    
    return "DataFrame with Renamed Columns"
def combinereplace_columns(df, main_column, other_column, value_to_replace):
    # Iterar sobre las filas del DataFrame
    for index, row in df.iterrows():
        if row[main_column] == value_to_replace:
            df.at[index, main_column] = row[other_column]
    df.drop(columns=[other_column], inplace=True)  # Eliminar la segunda columna después de combinar
    
    return "Completed"

# Función para combinar en una nueva columna sin modificar las columnas originales
def combine_columns(df, main_column, other_column, value_to_replace):
    # Crear una nueva columna con la combinación de las dos columnas
    new_column_name = f'{main_column} (Combined)'
    df[new_column_name] = df.apply(lambda row: row[other_column] if row[main_column] == value_to_replace else row[main_column], axis=1)
    
    return "Completed"
# Función para agrupar códigos
def agrupar_codigos(codigos):
    grupos = defaultdict(list)
    for codigo in codigos:
        parte_comun = codigo.split('_')[0]
        grupos[parte_comun].append(codigo)
    arreglo_final = [{'label': clave, 'value': '|'.join(codigos)}
                     for clave, codigos in grupos.items() if len(codigos) > 1]
    return arreglo_final
# Función para generar el wordcloud
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
def update_heatmap(selected_columns, filter_var):
    if not selected_columns:
        return go.Figure(), 200
    
    first_col = selected_columns[0]
    if filter_var == "No filter":
        unique_filter_values = ["nada"]
    else:
        unique_filter_values = df[filter_var].unique()
        unique_filter_values_labels = unique_filter_values.astype(str)

    
    num_subplots = len(unique_filter_values)
    
    if unique_filter_values[0]=="nada":
        fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True)
    else:
        fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True, subplot_titles=unique_filter_values_labels)
    
    for i, valor in enumerate(unique_filter_values):
        filtered_df = df if valor == 'nada' else df[df[filter_var] == valor]
        unicos = filtered_df[first_col].unique()
        unique_values = sorted(set(unicos))
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
                    font=dict(color="black", size=12),
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
    
    new_height = 200 + len(selected_columns) * 100 * num_subplots
    fig.update_layout(

            height=new_height,  
            
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


# Función para determinar el número de factores a retener
def determine_number_of_factors(eigenvalues, method):
    if method == "Kaiser":
        return np.sum(eigenvalues > 1)
    elif method == "Choose manually with Scree plot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(eigenvalues) + 1)),
            y=eigenvalues,
            mode='lines+markers',
            marker=dict(size=8),
            name='Eigenvalues'
        ))

        fig.add_shape(type="line",
                      x0=1, x1=len(eigenvalues),
                      y0=1, y1=1,
                      line=dict(color="Red", dash="dash"))

        fig.update_layout(
            title='Scree Plot',
            xaxis=dict(title='Factor'),
            yaxis=dict(title='Eigenvalue'),
            showlegend=False
        )

        st.plotly_chart(fig)

        return st.number_input("Select number of factors based on scree plot", min_value=1, max_value=len(eigenvalues), value=1)

    elif method == "Parallel Analysis":
        # Implementación del método de análisis paralelo
        pass
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

# Función para generar el HTML del reporte
def generate_html_report(report_data):
    html_content = '''
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f8ff;
                color: #333;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                padding: 10px;
                background-color: #0073e6;
                color: white;
                border-radius: 10px;
            }}
            .report-title {{
                text-align: center;
                font-size: 24px;
                color: #0073e6;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .toc {{
                border: 1px solid #0073e6;
                border-radius: 10px;
                padding: 10px;
                background-color: #e6f2ff;
            }}
            .toc h2 {{
                text-align: center;
                color: #0073e6;
            }}
            .toc ul {{
                list-style: none;
                padding: 0;
            }}
            .toc li {{
                margin: 10px 0;
            }}
            .figure {{
                border: 1px solid #0073e6;
                border-radius: 10px;
                padding: 10px;
                background-color: white;
                margin-top: 20px;
                text-align: center;
            }}
            .figure img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #0073e6;
                border-radius: 10px;
            }}
            .description {{
                margin-top: 10px;
                font-style: italic;
                color: #555;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background-color: #0073e6;
                color: white;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Descriptive Analysis Report</h1>
            <p>Generated at {date}</p>
        </div>
        <div class="report-title">Content Table</div>
        <div class="toc">
            <h2>Index</h2>
            <ul>
                {toc_items}
            </ul>
        </div>
        {figures}
        <div class="footer">
            <p>Autogenerated Report</p>
        </div>
    </body>
    </html>
    '''
    
    toc_items = ''
    figures = ''
    for i, fig_data in enumerate(report_data):
        fig = go.Figure(fig_data['figure'])
        fig.update_layout(width=1800, height=fig_data['height'])
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        toc_items += f'<li><a href="#figure{i+1}">Chart {i+1}</a></li>'
        description_html = f'<div class="description">{fig_data["description"]}</div>' if fig_data["description"] else ''
        figures += f'''
        <div class="figure" id="figure{i+1}">
            <h2>Chart {i+1}</h2>
            <img src="data:image/png;base64,{img_base64}" />
            {description_html}
        </div>
        '''
    
    return html_content.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        toc_items=toc_items,
        figures=figures
    )



def calculate_kmo2(corr_matrix):
    corr_matrix = np.asarray(corr_matrix)
    inv_corr_matrix = np.linalg.inv(corr_matrix)
    
    # Sum of squares of the correlations
    corr_sq = np.square(corr_matrix)
    np.fill_diagonal(corr_sq, 0)
    corr_sq_sum = np.sum(corr_sq)
    
    # Partial correlations
    partial_corr_matrix = -inv_corr_matrix / np.sqrt(np.outer(np.diag(inv_corr_matrix), np.diag(inv_corr_matrix)))
    partial_corr_sq = np.square(partial_corr_matrix)
    np.fill_diagonal(partial_corr_sq, 0)
    partial_corr_sq_sum = np.sum(partial_corr_sq)
    
    # KMO calculation
    kmo_numerator = corr_sq_sum
    kmo_denominator = corr_sq_sum + partial_corr_sq_sum
    kmo_value = kmo_numerator / kmo_denominator
    
    return kmo_value

# Test de Bartlett
def calculate_bartlett_corr_matrix(corr_matrix, n):
    chi_square_value = -(n - 1 - (2 * corr_matrix.shape[0] + 5) / 6) * np.log(np.linalg.det(corr_matrix))
    df = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2
    p_value = 1 - chi2.cdf(chi_square_value, df)
    return chi_square_value, p_value

# Función para obtener las columnas siguientes
def get_next_columns(current_column, all_columns):
    try:
        idx = all_columns.index(current_column)
        if idx < len(all_columns) - 1:
            return all_columns[idx + 2]
        else:
            return ""
    except ValueError:
        return ""

st.set_page_config(layout="wide")

# Título principal con alineación a la izquierda
st.markdown("<h1 style='text-align: left;'>Dashboard Creator</h1>", unsafe_allow_html=True)

# Configuración de los tabs principales
main_tab = st.sidebar.radio("Main Tabs", ["Descriptive Analysis", "Factorial Analysis", "Data Codification", "Generate Report"], index=["Descriptive Analysis", "Factorial Analysis", "Data Codification", "Generate Report"].index(st.session_state.current_tab))

# Subtabs para 'Descriptive Analysis'
nested_tab = None
if main_tab == "Descriptive Analysis":
    nested_tab_options = ["Various Graphics", "Matrix Charts", "Custom Heatmaps", "Numeric Charts"]
    nested_tab = st.sidebar.radio("Subsections", nested_tab_options, index=nested_tab_options.index(st.session_state.nested_tab) if st.session_state.nested_tab in nested_tab_options else 0)

# Layout para 'Descriptive Analysis'
if main_tab == "Descriptive Analysis":
    st.header("Descriptive Analysis")
    # Carga de archivo
    uploaded_file = st.file_uploader("Drag and drop or Choose CSV File", type="csv")
    deli = st.selectbox("Select the delimiter of your CSV:", ['Comma', 'Semicolon'])
    if deli =="Comma":
        delim=","
    else:
        delim=";"
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = load_data(uploaded_file, delim)
    
    df = st.session_state.df
    
    if df is not None:
        st.success("File uploaded successfully.")
        st.dataframe(df.head())
        columns = df.columns.tolist()
        options = [{'label': col, 'value': col} for col in columns]
        opciones_especiales = agrupar_codigos(columns)
    else:
        st.info("First Upload your CSV File.")
        columns = []
        opciones_especiales = []

    # Tabs anidados dentro de 'Descriptive Analysis'
    if nested_tab == "Various Graphics":
        st.subheader("Various Graphics")
        st.text("Use this type of graph to visualise the behaviour between any of your categorical variables.")
        if st.checkbox("Show instructions"):
            st.text("Instructions:")
            st.text("1: Select what chart you want")
            st.text("2: Select your first categorical variable (If you selected a word cloud chart, make sure this variable corresponds to an open-ended question).")
            st.text("3: If you selected to graph a contingency table, choosing a filter is mandatory to act as a second variable,\n   if you chose another graph it will be an optional variable to visualise how change your first variable according to the chosen filter.")
            st.text("4: Click on Sumbit to visualise your graphic")
        tipo_grafico = st.selectbox("Choose your chart:", ['Bar Chart', 'Pie Chart', 'Contingency Table', 'Word Cloud'])
        x_axis = st.selectbox("Choose your variable:",[""]+ columns)
        filtro = st.selectbox("Choose your filter:",["No filter"]+ columns)
        if tipo_grafico == 'Bar Chart':
            hori = st.checkbox("Horizontal bars")
        if tipo_grafico == 'Bar Chart' or tipo_grafico =='Pie Chart':
            likert = st.selectbox("Choose your likert scale:", ['Original', 'Agreement (5)', 'Agreement (6)', 'Quality (5)', 'Frequency (5)', 'Frequency (6)'])
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
            st.text("Choose what percentage you want to see in the table (order: N/Table %, N/row %, N/column %):")
            totalper = st.checkbox("N/Table total %", True)
            rowper = st.checkbox("N/Row total %")
            colper = st.checkbox("N/Column total %")
        if st.button("Submit"):
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
                                textposition='auto',
                                orientation="h"
                                ))
                            else:
                                fig.add_trace(go.Bar(
                                x=counts.index,
                                y=counts,
                                name=str(filter_value),
                                text=[f'{p:.1f}%' for p in percentages],
                                textposition='auto',
                                orientation="v"
                                ))
        
                            data_table.extend(filtered_df[[x_axis, filtro]].to_dict('records'))
        
                        fig.update_layout(
                            barmode='group',
                            title=f"{x_axis} Filtered by {filtro}",
                            xaxis_title=x_axis,
                            yaxis_title='Frequency',
                            height=700
                        )
                        contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                        output_table = calculate_percentages(contingency_table, x_axis, filtro)
                        data_table = output_table.reset_index().to_dict('records')
                    else:
                        counts = aux_df[x_axis].value_counts()
                        total_counts = counts.sum()
                        percentages = (counts / total_counts) * 100
                        print(counts.index)
                        if hori:
                            fig.add_trace(go.Bar(
                            x=counts,
                            y=counts.index,
                            text=[f'{p:.1f}%' for p in percentages],
                            textposition='auto',
                            orientation="h"
                            ))
                        else:
                            fig.add_trace(go.Bar(
                            x=counts.index,
                            y=counts,
                            text=[f'{p:.1f}%' for p in percentages],
                            textposition='auto',
                            orientation="v"
                            ))
                        fig.update_layout(
                            height=700,
                            title="Bar chart of "+str(x_axis)
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
                        fig = make_subplots(rows=1, cols=len(unique_values), specs=[[{'type': 'domain'}]*len(unique_values)])
                        annotations = []
        
                        for i, value in enumerate(unique_values):
                            filtered_df = aux_df[aux_df[filtro] == value]
                            fig.add_trace(
                                go.Pie(labels=filtered_df[x_axis].value_counts().index, values=filtered_df[x_axis].value_counts().values, name=str(value)),
                                row=1, col=i+1
                            )
                            annotations.append(dict(
                                x=0.5/len(unique_values) + i*(1.0/len(unique_values)),
                                y=-0.1,
                                text=str(value),
                                showarrow=False,
                                xanchor='center',
                                yanchor='top'
                            ))
        
                        contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                        output_table = calculate_percentages(contingency_table, x_axis, filtro)
                        data_table = output_table.reset_index().to_dict('records')
        
                        fig.update_layout(
                            title=f"{x_axis} Filtered by {filtro}",
                            annotations=annotations
                        )
                    else:
                        fig = px.pie(aux_df, names=x_axis)
                        fig.update_layout(
                           
                            title="Pie chart of "+str(x_axis)
                        )
                        counts = aux_df[x_axis].value_counts().reset_index()
                        counts.columns = [x_axis, 'count']
                        data_table = counts.to_dict('records')
        
                elif tipo_grafico == 'Contingency Table':
                    fig = go.Figure()
                    data_table = []
                    if filtro!="No filter":
                        contingency_table = pd.crosstab(df[x_axis], df[filtro])
                        fig = go.Figure(data=go.Heatmap(
                            z=contingency_table.values,
                            x=contingency_table.columns,
                            y=contingency_table.index,
                            colorscale='Blues'
                        ))
        
                        fig.update_layout(
                            title=f'Contingency table between {x_axis} and {filtro}',
                            xaxis_title=filtro,
                            yaxis_title=split_text_by_words(f'{str(x_axis)}', 9),
                            xaxis=dict(tickmode='array', tickvals=list(contingency_table.columns)),
                            yaxis=dict(tickmode='array', tickvals=list(contingency_table.index)),
                        )
                        contingency_table = pd.crosstab(df[x_axis], df[filtro], margins=True, margins_name='Total')
                        output_table = calculate_percentages(contingency_table, x_axis, filtro)
                        data_table = output_table.reset_index().to_dict('records')
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
                st.dataframe(pd.DataFrame(data_table))
        show_graph_and_table()
    elif nested_tab == "Matrix Charts":
        st.subheader("Matrix Charts")
        st.text("Use this type of graph to visualise the behaviour between a set of categorical variables that are related to a similar question (these sets are automatically recognised).")
        if st.checkbox("Show instructions"):
            st.text("Instructions:")
            st.text("1: Select what chart you want (Tick Bar chart: please note that this type of chart is for multiple choice questions only)")
            st.text("2: Select your set of categorical variables ")
            st.text("3: You can optionally choose a filter that will allow you to visualise the change of your set of variables according to the chosen filter.")
            st.text("4: If you have chosen the Multibar chart option and in addition your set of variables are numerically coded, you can change your scale to Likert.")
            st.text("5: Click on Sumbit to visualise your graphic")
        chart_type = st.selectbox("Choose your chart:", [ 'Multi bar Chart', 'Radar Chart','Tick bar Chart'])
        variable_group = st.selectbox("Choose your group of variables (Matrix):",[""]+ opciones_especiales, format_func=lambda option: option['label'] if option else None)
        filter_var = st.selectbox("Choose your filter:",["No filter"]+ columns)
        likert_scale = st.selectbox("Choose your likert scale (Only Multi plot):", ['Original', 'Agreement (5)', 'Agreement (6)', 'Quality (5)', 'Frequency (5)', 'Frequency (6)'])
        if st.button("Submit"):
            if variable_group:
                columnas = variable_group['value'].split("|")
                fig = None

                if chart_type == 'Tick bar Chart':
                    fig = generate_horizontal_bar_chart(df, columnas, filter_var)
                elif chart_type == 'Radar Chart':
                    fig = generate_radar_chart(df, columnas, filter_var)
                elif chart_type == 'Multi bar Chart':
                    st.text("Note: the label * mean less than 4%")
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
        st.text("Use this type of graph to visualise behaviour among a set of categorical variables that you can choose manually, the set of questions must have the same scale.")
        if st.checkbox("Show instructions"):
            st.text("Instructions:")
            st.text("1: Select your first categorical variable (Note that the other variables you can select will depend on the scale of this variable.) ")
            st.text("2: You can add all variables that have the same scale, by selecting the variable in the second dropdown and pressing the add button.")
            st.text("3: You can optionally choose a filter that will allow you to visualise the change of your set of variables according to the chosen filter.")
            st.text("4: Click on Sumbit to visualise your graphic")
        var1 = st.selectbox("Choose your first variable:", [""]+columns)
        selected_columns = st.session_state.get('selected_columns', [var1])
        if var1:
            options, var2_disabled, add_disabled = update_second_dropdown_options(var1)
        else:
            options, var2_disabled, add_disabled = [], True, True

        var2 = st.selectbox("Choose your additional variable:", [opt['label'] for opt in options], disabled=var2_disabled)
        add_button = st.button("Add", disabled=add_disabled)

        if add_button and var2:
            if var2 not in selected_columns:
                selected_columns.append(var2)
                st.session_state['selected_columns'] = selected_columns

        filter_var = st.selectbox("Select your filter:", ["No filter"]+columns)
        show_table = st.button("Sumbit")

        if show_table:
            heatmap_fig, heatmap_height = update_heatmap(selected_columns, filter_var)
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
                    st.dataframe(pd.DataFrame(data_table))
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
                
                st.dataframe(pd.DataFrame(data_table))
        show_graph_and_table()
    elif nested_tab == "Numeric Charts":
        st.subheader("Numeric charts")
        st.text("Use this type of graph to visualise the behaviour between a categorical variable and a numerical variable.")
        if st.checkbox("Show instructions"):
            st.text("Instructions:")
            st.text("1: Select your first categorical variable")
            st.text("2: Select your numerical variable")
            st.text("3: You can optionally choose a categorical filter that will allow you to visualise the change of your variables according to the chosen filter.")
            st.text("4: Click on Sumbit to visualise your graphic")   
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()        
        chart_type = st.selectbox("Select your chart:", ['Box plot', 'Mean polygon chart', 'Violin chart', 'Ridgeline chart', 'Displot chart'])
        num_var = st.selectbox("Select your numerical variable:", [""]+numeric_columns)
        cat_var = st.selectbox("Select your group variable:", ["No variable"]+columns)
        filter_var = st.selectbox("Select your filter (categoric):", ["No filter"]+columns)
        if chart_type == 'Violin chart':
            show_points = st.checkbox("Show Points")
            show_box = st.checkbox("Show Box")
            #show_ = st.checkbox("Show labels")
        else:
            show_points = False

        show_table = st.button("Sumbit")


        if show_table:
            fig = update_numerico(chart_type, cat_var, num_var, filter_var, show_points)
            st.plotly_chart(fig, use_container_width=True)
            if fig:
                st.session_state['last_fig'] = fig
                st.session_state['fig_width'] = fig.layout.width
                st.session_state['fig_height'] = fig.layout.height
        show_graph_and_table()
# Layout para 'Exploratory Analysis'
elif main_tab == "Factorial Analysis":
    df = st.session_state.df
    columns =st.session_state.df.columns
    st.header("Factorial Analysis")
    nested_tab_options = ["Exploratory factor analysis", "Confirmatory factor analysis"]
    nested_tab = st.sidebar.radio("Subsections", nested_tab_options, index=nested_tab_options.index(st.session_state.nested_tab) if st.session_state.nested_tab in nested_tab_options else 0)
    
    if nested_tab == "Exploratory factor analysis":
        st.subheader("Exploratory Factor Analysis")
        
        # Asegurarse de que 'columns' sea una lista
        columns_list = list(columns)
        
        # Selección de ítems
        selected_items = st.multiselect("Select items for EFA", columns_list, default=None)
        
        if len(selected_items)>1:
            df_selected = df[selected_items]
            
            # Tipo de rotación

            correlation_matrix_type = st.selectbox("Select correlation matrix type", ["Pearson", "Polychoric"])

            # Calcular la matriz de correlación
            if correlation_matrix_type == "Pearson":
                correlation_matrix = df_selected.corr()
            else:
                ordinal_data = df_selected.to_numpy()
                ordinal_data_transposed = ordinal_data.transpose()  # o ordinal_data.T

                correlation_matrix = polychoric_correlation_serial(ordinal_data_transposed)

            # Mostrar matriz de correlación al pulsar un botón
            if st.checkbox("Show Correlation Matrix"):
                st.write("Correlation Matrix:")
                st.dataframe(correlation_matrix)

            # Pruebas de Bartlett y KMO
            if st.button("Run Bartlett's test and KMO"):
                n = len(df)
                chi_square_value, p_value = calculate_bartlett_corr_matrix(correlation_matrix, n)
                kmo = calculate_kmo2(correlation_matrix)
                st.write(f"Bartlett's test: Chi-square value = {chi_square_value}, p-value = {p_value}")
                st.write(f"Kaiser-Meyer-Olkin (KMO) test: KMO model = {kmo}")
            # Tipo de matriz de correlación

            # Método de extracción
            extraction_method = st.selectbox("Select extraction method", ["Principal Axis Factoring", "Maximum Likelihood","Unweighted Least Squares (ULS)", "Minimal Residual"])
            rotation = st.selectbox("Select rotation method", ["Varimax (Orthogonal)", "Promax (Oblique)", "Oblimin (Oblique)","Oblimax (Orthogonal)","Quartimin (Oblique)","Quartimax (Orthogonal)","Equamax (Orthogonal)", "None"])

            if extraction_method == "Principal Axis Factoring":
                method = 'principal'
            elif extraction_method == "Maximum Likelihood":
                method = 'ml'
            elif extraction_method == "Unweighted Least Squares (ULS)":
                method = 'uls'
            elif extraction_method == "Minimal Residual":
                method = 'minres'

            fa = FactorAnalyzer(rotation=None, method=method)
            fa.fit(df_selected)
            eigenvalues, _ = fa.get_eigenvalues()


            # Criterios para retención de factores
            retention_method = st.selectbox("Select method to determine number of factors", ["Kaiser", "Choose manually with Scree plot", "Parallel Analysis (NOT IMPLEMENTED)"])
            n_factors = determine_number_of_factors(eigenvalues, retention_method)
                        # Obtener eigenvalues y calcular varianza explicada

            if st.button("Run") and n_factors:
                if rotation != "None":
                    rotation = rotation.split(" ")[0].lower()
                else:
                    rotation = None
                if method == "principal":
                    if correlation_matrix_type != "Pearson":
                        st.write("Note: In the Principal Axis Factoring you only can use the default correlation matrx (Pearson)")
                    corrm = False
                else:
                    corrm = True
                fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, is_corr_matrix=corrm)
                if corrm:
                    fa.fit(correlation_matrix)
                else:
                    fa.fit(df_selected)
                ev, v = fa.get_eigenvalues()
                total_variance_explained = np.cumsum(ev) / np.sum(ev)

                # Crear un DataFrame para mostrar los resultados
                explained_variance_df = pd.DataFrame({
                    'Factor': [f'Factor {i+1}' for i in range(len(ev))],
                    'Eigenvalue': ev,
                    'Proportion of variance explained': ev / np.sum(ev),
                    'Explained cumulative variance': total_variance_explained
                })

                # Visualización de la tabla en Streamlit
                st.write("Total Variance Explained:")
                st.dataframe(explained_variance_df,hide_index=True)


                communalities = fa.get_communalities()
                print(communalities)
                # Mostrar las comunalidades de cada ítem en una tabla separada
                st.write("Comunalities")
                items = df_selected.columns.tolist()
                items_communalities = pd.DataFrame({
                    'Item': items,
                    'Communality (Extraction)': communalities
                })
                st.dataframe(items_communalities, hide_index=True)
                # Mostrar resultados de cargas factoriales
                st.write("Factor Loadings:")
                loadings = pd.DataFrame(fa.loadings_, index=selected_items)
                loadings.columns = [f'Factor {i+1}' for i in range(loadings.shape[1])]
                st.dataframe(loadings)

                                # Graficar cargas factoriales con Plotly
                fig = go.Figure(data=go.Heatmap(
                    z=abs(loadings.values),
                    x=loadings.columns,
                    y=loadings.index,
                    colorscale='Blues',
                    showscale=True,
                    zmin=0,
                    zmax=1
                ))

                fig.update_layout(
                    title='Factor Loadings Heatmap',
                    xaxis_nticks=36
                )

                st.plotly_chart(fig)
                G = nx.DiGraph()

                # Añadir nodos de factores y ítems
                for i, factor in enumerate(loadings.columns):
                    G.add_node(factor, color='green')
                for item in loadings.index:
                    G.add_node(item, color='orange')

                # Añadir aristas con pesos basados en las cargas factoriales
                for item in loadings.index:
                    for i, factor in enumerate(loadings.columns):
                        weight = loadings.loc[item, factor]
                        if abs(weight) > 0.3:  # Puedes ajustar este umbral según sea necesario
                            G.add_edge(factor, item, weight=abs(weight))

                # Posiciones fijas: Factores a la izquierda y ítems a la derecha
                pos = {}
                mitadabajo = len(loadings.index)//2-len(loadings.columns)//2
                for i, factor in enumerate(loadings.columns):
                    pos[factor] = (0,int( (i+1)/(len(loadings.columns))*(len(loadings.index)-1)))
                for i, item in enumerate(loadings.index):
                    print(i)
                    pos[item] = (1, i)

                edges = G.edges(data=True)
                weights = [edge[2]['weight'] for edge in edges]

                colors = [G.nodes[node]['color'] for node in G.nodes]

                fig, ax = plt.subplots(figsize=(12, 8))
                nx.draw(G, pos, with_labels=True, node_color=colors, edge_color=weights,edge_vmin=0.0, edge_vmax=1.0, width=2.0, edge_cmap=plt.cm.Blues, node_size=3000, font_size=10, font_weight='bold', ax=ax)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array(weights)
                fig.colorbar(sm, ax=ax, label='Factor Loading Strength')

                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)
    elif nested_tab == "Confirmatory factor analysis":
        st.subheader("Confirmatory Factor Analysis")
        st.write("Define your model:")

        # Selección de ítems
        columns_list = list(columns)
        selected_items = st.multiselect("Select items for CFA", columns_list, default=None)

        if selected_items:
            df_selected = df[selected_items]

        # Mostrar selectores de ítems para cada factor y almacenar ítems seleccionados temporalmente
        initial_factor_items = st.session_state.factor_items.copy()

        temp_factor_items = {}
        used_items = set()
        update_required = False

        # Iterar sobre los factores y sus ítems
        for factor in st.session_state.factor_items.keys():
            available_items = [item for item in selected_items if item not in used_items]

            # Crear el multiselect para cada factor
            temp_factor_items[factor] = st.multiselect(
                f"Select items for {factor}",
                available_items,
                key=f'{factor}_selected_items'
            )
            used_items.update(temp_factor_items[factor])

        for factor, items in temp_factor_items.items():
            if st.session_state.factor_items[factor] != items:
                st.session_state.factor_items[factor] = items
                # Verificar si el factor actual es diferente al último factor seleccionado
                if st.session_state.last_selected_factor and st.session_state.last_selected_factor != factor:
                    update_required = True
                st.session_state.last_selected_factor = factor

        # Si hubo cambios en el factor seleccionado y es un factor diferente, forzar la recarga de la página
        #if update_required:
            #st.rerun()
        # Botones para agregar o quitar factores
        st.button("Add Factor", on_click=add_factor)
        st.button("Remove Factor", on_click=remove_factor)

        functionmethod = st.selectbox("Select objective function to minimize", 
                                      ["Wishart loglikelihood (MLW)", "Unweighted Least Squares (ULS)", 
                                       "Generalized Least Squares (GLS)", "Weighted Least Squares (WLS)", 
                                       "Diagonally Weighted Least Squares (DWLS)", "Full Information Maximum Likelihood (FIML)"])

        # Construir la definición del modelo
        model_definition = ""
        for factor, items in st.session_state.factor_items.items():
            if items:
                model_definition += f"F{factor[-1]} =~ {' + '.join(items)}\n"
        activatedlav = st.checkbox("Activate manual lavaan writing")
        model_definition= st.text_area("Model Definition (lavaan syntax)", model_definition, height=150, disabled=not activatedlav)
        if st.button("Run CFA (Add model)"):
            try:
                # Verificar si la definición del modelo es válida
                if not model_definition.strip():
                    st.error("The model definition cannot be empty.")
                    raise ValueError("Empty model definition")

                model = Model(model_definition)
                res_opt = model.fit(df_selected, obj=(functionmethod.split(" ")[-1])[1:-1])

                # Añadir el modelo a la lista de modelos en session_state
                st.session_state.models.append((model, res_opt))

                # Comprobaciones adicionales del modelo
                if not model.parameters:
                    st.error("No parameters found in the model.")
                    raise ValueError("No parameters found in the model")

            except Exception as e:
                st.error(f"Error fitting model: {e}")

        # Mostrar resultados de todos los modelos
        if st.session_state.models:
            for idx, (model, res_opt) in enumerate(st.session_state.models):
                st.write(f"Model {idx+1}")
                g = semopy.semplot(model, filename=f"model_{idx}.png", std_ests=True)
                st.image(f"model_{idx}.png")
                st.write("Number of iterations: ")
                st.write(res_opt.n_it)

                estimates = model.inspect(std_est=True)
                estimates = estimates.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
                st.write("Model estimation")
                st.dataframe(estimates, hide_index=True)

                st.write("Statistics")
                stats = semopy.calc_stats(model).round(3)
                st.write(stats)

                cov_estimate, _ = model.calc_sigma() 
                cov = model.mx_cov
                residual = cov - cov_estimate
                std_residual = residual / np.std(residual)
                std_res = pd.DataFrame(
                    std_residual,
                    columns=model.names_lambda[0], index=model.names_lambda[0],
                )
                st.write("Residue of covariance Matrix")
                st.write(std_res)

                loadings = estimates[estimates["op"] == "~"]
                constructs = loadings["rval"].unique().tolist()

                # AVE computation 
                st.write("Average Variance Extracted")
                lods = []
                for cons in constructs:
                    squared_loadings = loadings[loadings["rval"] == cons]["Est. Std"] ** 2
                    ave = squared_loadings.sum() / squared_loadings.size
                    lods.append({'Factor': cons, 'AVE': ave})

                # Crear DataFrame
                df_ave = pd.DataFrame(lods)

                # Mostrar DataFrame en Streamlit
                st.dataframe(df_ave, hide_index=True)
# Layout para 'Data Codification'
elif main_tab == "Data Codification":
    if 'show_manual_change' not in st.session_state:
        st.session_state.show_manual_change = False
    if st.session_state.df is not None:
        st.subheader("Manual data codification")
        column_to_edit = st.selectbox("Select column to edit:", [""]+st.session_state.df.columns)
                # Generar inputs para valores únicos de la columna seleccionada
        new_column_name = st.text_input("Edit column name:")
        if st.button("Confirm name change"):
            if new_column_name:
                st.session_state.df.rename(columns={column_to_edit: new_column_name}, inplace=True)
                st.success(f"Column name changed to '{new_column_name}'")
                st.experimental_rerun()  # Refrescar la interfaz de usuario
 
        if st.checkbox("Change values of column manually"):
            unique_values = st.session_state.df[column_to_edit].unique()
            new_values = {}
            for value in unique_values:
                new_value = st.text_input(f"Change '{value}' to:", value)
                new_values[value] = new_value
    
            if st.button("Confirm changes to values"):
                for old_value, new_value in new_values.items():
                    st.session_state.df[column_to_edit].replace(old_value, new_value, inplace=True)
                st.success("Values updated successfully")
                st.experimental_rerun()


        st.subheader("Automatic data codification")
        if st.button("Translate column to English"):
            result = translate_column(st.session_state.df, column_to_edit)
            st.success(result)
            st.experimental_rerun()
        if st.button("Translate entire dataframe to English"):
            result = translate_dataframe(st.session_state.df)
            st.success(result)
            st.experimental_rerun()
                # Interface to input characters to replace
        if st.checkbox("Show automatic delimiter changer"):
            old_char = st.text_input('Character to replace in column names:', '-')
            new_char = st.text_input('New character:', '_')

            if st.button('Change characters'):
                # Call the function to rename columns
                result = rename_columns(st.session_state.df, old_char, new_char)
                st.success(result)
                st.experimental_rerun()
        st.subheader("Combine 2 columns (one main column and one “other” column)")
        maincolumn = st.selectbox("Select the main column:", [""]+st.session_state.df.columns)
        next_column = get_next_columns(maincolumn, st.session_state.df.columns.tolist())
        if maincolumn:
            unique_valuesf = st.session_state.df[maincolumn].unique()
            tobereplaced = st.selectbox("Select the value to be replaced:", [""]+unique_valuesf)
            
        othercolumn = st.selectbox("Select the 'Other' column", [""] + st.session_state.df.columns.tolist(), index=st.session_state.df.columns.tolist().index(next_column) if next_column else 0)
        if st.button('Combine and replace'):
            # Call the function to rename columns
            result = combinereplace_columns(st.session_state.df, maincolumn, othercolumn, tobereplaced)
            st.success(result)
            st.experimental_rerun()
        if st.button('Combine in a new column'):
            # Call the function to rename columns
            result = combine_columns(st.session_state.df, maincolumn, othercolumn, tobereplaced)
            st.success(result)
            st.experimental_rerun()
        st.subheader("Save changes to CSV")
        if st.button("Save updated DF"):
            csv = st.session_state.df.to_csv(index=False)
            # Agrega un botón de descarga en Streamlit
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='updated_dataframe.csv',
                mime='text/csv',
            )

            # Mensaje de éxito opcional
            st.success('The Dataframe is ready for download.')
    else:
        st.subheader("Please first upload your CSV")
elif main_tab == "Generate Report":
    st.subheader("Download Report")
    if st.button("Generate Report"):
        if st.session_state['report_data']:
            html_report = generate_html_report(st.session_state['report_data'])
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="report.html">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No graphs to generate report")
        
# Guardar el estado actual de las tabs en session_state
st.session_state.current_tab = main_tab
st.session_state.nested_tab = nested_tab
