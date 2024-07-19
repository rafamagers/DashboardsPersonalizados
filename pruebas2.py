import streamlit as st
import pandas as pd
import base64

st.title('Generador de Tablas en Formato APA')

# Cargar DataFrame de ejemplo
data = {
    'Nombre': ['Alice', 'Bob', 'Charlie'],
    'Edad': [24, 27, 22],
    'Ciudad': ['Nueva York', 'Los Ángeles', 'Chicago']
}
df = pd.DataFrame(data)

st.write('DataFrame de Ejemplo:')
st.dataframe(df)

def dataframe_to_apa_table(df):
    table = '\\begin{table}[H]\n\\centering\n\\begin{tabular}{' + ' | '.join(['c'] * len(df.columns)) + '}\n\\hline\n'
    table += ' & '.join(df.columns) + ' \\\\\n\\hline\n'
    for _, row in df.iterrows():
        table += ' & '.join(map(str, row.values)) + ' \\\\\n'
    table += '\\hline\n\\end{tabular}\n\\caption{Tabla en formato APA}\n\\end{table}'
    return table

def dataframe_to_latex_display(df):
    table = '\\begin{array}{' + ' | '.join(['c'] * len(df.columns)) + '}\n\\hline\n'
    table += ' & '.join(df.columns) + ' \\\\\n\\hline\n'
    for _, row in df.iterrows():
        table += ' & '.join(map(str, row.values)) + ' \\\\\n'
    table += '\\hline\n\\end{array}'
    return table

if st.button('Generar Tabla en Formato APA'):
    apa_table = dataframe_to_apa_table(df)
    st.write('Tabla en Formato APA:')
    st.text_area('Tabla en Formato APA', apa_table, height=300)
    
    apa_table_display = dataframe_to_latex_display(df)
    st.write('Visualización de la tabla en LaTeX:')
    st.latex(apa_table_display)

def get_table_download_link(tex_content):
    b64 = base64.b64encode(tex_content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="table.tex">Descargar Tabla en Formato APA</a>'
    return href

if st.button('Descargar Tabla'):
    apa_table = dataframe_to_apa_table(df)
    st.markdown(get_table_download_link(apa_table), unsafe_allow_html=True)
