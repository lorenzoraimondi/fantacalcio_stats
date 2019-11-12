import pandas as pd
import streamlit as st
from fantadata import Fantadata

fd = Fantadata()

rank_df = fd.build_rank(fd.get_teams())

st.write(rank_df)
