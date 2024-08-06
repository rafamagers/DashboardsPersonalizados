import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from plotly import graph_objs as go
import numpy as np
from scipy.stats import chi2
from factor_analyzer import FactorAnalyzer
from semopy import Model
import semopy
from RyStats.common import polychoric_correlation_serial

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
# Función para determinar el número de factores a retener
def determine_number_of_factors(eigenvalues, method):
    if method == "Kaiser":
        return np.sum(eigenvalues > 1)
    elif method == "Choose manually with Scree plot":
              # Crear la figura
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

        # Actualizar el layout para ajustar el eje Y y mostrar la cuadrícula
        fig.update_layout(
            title='Scree Plot',
            xaxis=dict(title='Factor'),
            yaxis=dict(title='Eigenvalue', range=[0, max(eigenvalues) + 0.2], dtick=0.2),
            showlegend=False,
            yaxis_tickformat='.3f',  # Formato de los ticks del eje Y
            yaxis_showgrid=True,     # Mostrar la cuadrícula del eje Y
            yaxis_gridcolor='LightGray'  # Color de la cuadrícula del eje Y
        )

        # Mostrar la figura en Streamlit
        st.plotly_chart(fig)

        return st.number_input("Select number of factors based on scree plot", min_value=1, max_value=len(eigenvalues), value=1)

    elif method == "Parallel Analysis":
        # Implementación del método de análisis paralelo
        pass
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

def calculate_kmo2(corr_matrix, item_names):
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

    # Individual KMO calculation
    kmo_individual = []
    for i in range(corr_matrix.shape[0]):
        corr_sq_sum_i = np.sum(corr_sq[:, i]) - corr_sq[i, i]
        partial_corr_sq_sum_i = np.sum(partial_corr_sq[:, i]) - partial_corr_sq[i, i]
        kmo_i_numerator = corr_sq_sum_i
        kmo_i_denominator = corr_sq_sum_i + partial_corr_sq_sum_i
        kmo_individual.append((kmo_i_numerator / kmo_i_denominator).round(4))
    
    # Create the resulting matrix
    kmo_matrix = pd.DataFrame({
        "Item": item_names,
        "KMO Value": kmo_individual
    })
    
    return kmo_value, kmo_matrix

# Test de Bartlett
def calculate_bartlett_corr_matrix(corr_matrix, n):
    chi_square_value = -(n - 1 - (2 * corr_matrix.shape[0] + 5) / 6) * np.log(np.linalg.det(corr_matrix))
    df = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2
    p_value = 1 - chi2.cdf(chi_square_value, df)
    return chi_square_value, p_value







st.set_page_config(page_title="Factorial Analysis", page_icon="🧪", layout="wide")


df = st.session_state.df
if df is None:
    st.error("First Upload your CSV File.")
    st.stop()
st.sidebar.header("Choose")
columns =st.session_state.df.columns
st.header("🧪 Factorial Analysis")
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
            st.dataframe(correlation_matrix, height=correlation_matrix.shape[0] * 35+35)
            # Crear el heatmap
            fig = px.imshow(correlation_matrix,
                    color_continuous_scale='Blues',
                    zmin=0, zmax=1,
                    labels={'color': 'Correlation'},
                    title='Heatmap representation')
            # Mostrar el heatmap en Streamlit
            st.plotly_chart(fig)
        # Pruebas de Bartlett y KMO
        if st.button("Run Bartlett's test and KMO"):
            n = len(df)
            chi_square_value, p_value = calculate_bartlett_corr_matrix(correlation_matrix, n)
            kmooverall,kmoindi = calculate_kmo2(correlation_matrix, selected_items)
            st.markdown(
                f"""
                ### Bartlett's test:
                - **Chi-square value =** {chi_square_value}
                - **p-value =** {p_value}

                ### Kaiser-Meyer-Olkin (KMO) test:
                - **Overall =** {kmooverall}
                - **Individual =** 
            """
            )
            print(kmoindi)
            st.dataframe(kmoindi,hide_index=True)
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
        if rotation != "None":
            rotation = rotation.split(" ")[0].lower()
        else:
            rotation = None
        fa = FactorAnalyzer(rotation=rotation, method=method)
        fa.fit(df_selected)
        eigenvalues, _ = fa.get_eigenvalues()
        # Criterios para retención de factores
        retention_method = st.selectbox("Select method to determine number of factors", ["Kaiser", "Choose manually with Scree plot", "Parallel Analysis (NOT IMPLEMENTED)"])
        n_factors = determine_number_of_factors(eigenvalues, retention_method)
                    # Obtener eigenvalues y calcular varianza explicada
        if st.button("Run") and n_factors:
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