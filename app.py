import pandas as pd
import streamlit as st
from fantadata import Fantadata

@st.cache(allow_output_mutation=True)
def init_data():
    print("Initializing Fantadata Instance")
    return Fantadata()

fd = init_data()

st.markdown("# ⚽⚽ Nevian Cup Sfiga Stats ⚽⚽")

turn = st.slider("Giornata:", 1, fd.get_played_turns(), value=fd.get_played_turns(), step=1)
rank_df = fd.build_rank(turn=turn)

st.write(rank_df)
