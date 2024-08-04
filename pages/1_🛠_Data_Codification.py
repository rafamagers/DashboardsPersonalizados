import streamlit as st
from googletrans import Translator

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
translator = Translator()
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


st.set_page_config(page_title="Data Codification", page_icon="🛠", layout= "centered")

df = st.session_state.df
if df is None:
    st.error("First Upload your CSV File.")
    st.stop()
columns =st.session_state.df.columns
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
        unique_valuesf = [str(val) for val in unique_valuesf]

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