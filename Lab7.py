#Lab7
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Iris Data Explorer",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Dataset Interactive Visualization App")


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

st.success("Iris dataset loaded successfully!")

st.sidebar.header("Filter Options")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

selected_hist_col = st.sidebar.selectbox(
    "Select column for histogram:", numeric_cols
)

x_axis = st.sidebar.selectbox("Scatter plot X-axis:", numeric_cols)
y_axis = st.sidebar.selectbox("Scatter plot Y-axis:", numeric_cols)

st.subheader("📌 Dataset Preview")
st.dataframe(df.head())

st.subheader("📈 Summary Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Unique Species", df["species"].nunique())

st.header("📊 Visualizations")

st.subheader(f"Histogram of {selected_hist_col}")
fig1, ax1 = plt.subplots()
sns.histplot(df[selected_hist_col], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df["species"], palette="deep", ax=ax2)
st.pyplot(fig2)


