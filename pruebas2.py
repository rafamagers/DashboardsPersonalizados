import streamlit as st
import pandas as pd
from plotnine import ggplot, geom_density, aes, theme_light, geom_point, stat_smooth
from pathlib import Path

infile = Path(__file__).parent / "penguins.csv"
df = pd.read_csv(infile)


def dist_plot(df):
    plot = (
        ggplot(df, aes(x="Body Mass (g)", fill="Species"))
        + geom_density(alpha=0.2)
        + theme_light()
    )
    return plot.draw()


def scatter_plot(df, smoother):
    plot = (
        ggplot(
            df,
            aes(
                x="Bill Length (mm)",
                y="Bill Depth (mm)",
                color="Species",
                group="Species",
            ),
        )
        + geom_point()
        + theme_light()
    )

    if smoother:
        plot = plot + stat_smooth()

    return plot.draw()


with st.sidebar:
    mass = st.slider("Mass", 2000, 8000, 6000)
    smoother = st.checkbox("Add Smoother")

filt_df = df.loc[df["Body Mass (g)"] < mass]

st.pyplot(scatter_plot(filt_df, smoother))
st.pyplot(dist_plot(filt_df))