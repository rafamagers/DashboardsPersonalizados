import os
import streamlit as st
import semopy

# Añadir la ruta de Graphviz al PATH
os.environ["PATH"] += os.pathsep + '/usr/local/bin'  # Reemplaza con la ruta correcta si es necesario

# Tu código Streamlit y Semopy
def main():
    data = semopy.examples.political_democracy.get_data()
    mod = semopy.examples.political_democracy.get_model()
    m = semopy.Model(mod)
    m.fit(data)
    g = semopy.semplot(m, "pd.png")

    st.image("pd.png")

if __name__ == "__main__":
    main()