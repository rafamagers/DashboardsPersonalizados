import streamlit as st



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
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import threading
import uuid
import time

terminar = False
previarespuesta = ""
# Usar las credenciales de tu archivo JSON

os.environ['PATH'] = f"{os.path.expanduser('~/R/bin')}:{os.environ['PATH']}"
os.environ['R_HOME'] = os.path.expanduser('~/R')
os.environ["PATH"] += os.pathsep + '/usr/bin'  # Reemplaza con la ruta correcta si es necesario
#from transformers import pipeline
#qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# Inicializar el traductor




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
if 'num_factors' not in st.session_state:
    st.session_state.num_factors = 2
if 'factor_items' not in st.session_state:
    st.session_state.factor_items = {f'Factor {i+1}': [] for i in range(st.session_state.num_factors)}

if 'models' not in st.session_state:
    st.session_state.models = []
if 'last_selected_factor' not in st.session_state:
    st.session_state.last_selected_factor = None


st.set_page_config(page_title="Home", page_icon="🌍", layout="centered")


st.write("# Welcome to Inter-viewer 👀!")

st.sidebar.success("Upload your data")
st.markdown(
    """
    Inter-viewer 👀 is a web application to perform descriptive 
    and factor analysis of survey data with ease.
    This is a **Storymodelers** tool!
    
    ### How to use it?
    1. First upload your CSV data below
    2. Clean and encode your data in "🛠Data codification" if necessary.
    3. Now you can analyze your data in "📊Descriptive Analysis" or "🧪Factorial Analysis"
    ### See more
    - The source code in [Github](https://github.com/rafamagers/DashboardsPersonalizados)
    - Info about [Storymodelers team](https://www.storymodelers.org/)
"""
)

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

st.header("Upload your CSV")

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
    st.dataframe(df.head(), hide_index=True)
else:
    st.info("First Upload your CSV File.")
    columns = []




