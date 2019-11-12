from sklearn import datasets
import pandas as pd
import streamlit as st

boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df_boston['target'] = pd.Series(boston_data.target)

st.write(df_boston)
