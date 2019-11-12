import numpy as np
import pandas as pd
import streamlit as st
from fantadata import Fantadata
from statsmodels.tsa.ar_model import AR

@st.cache(allow_output_mutation=True)
def init_data():
    print("Initializing Fantadata Instance")
    return Fantadata()

fd = init_data()

teams = fd.get_teams()

st.markdown("# ⚽⚽ Nevian Cup Sfiga Stats ⚽⚽")

turn = st.slider("Giornata:", 1, fd.get_played_turns(), value=fd.get_played_turns(), step=1)
rank_df = fd.build_rank(turn=turn)

st.write(rank_df)

team = st.selectbox("Squadra:", teams)

team_df = fd.get_team_timeseries(team)

data = team_df.pts.values.tolist()
model = AR(data)
model_fit = model.fit()
yhat = model_fit.predict(start=len(data), end=len(data))
row = pd.DataFrame({'turn': len(team_df)+1, 'pts': yhat, 'pts_rank': np.nan})
team_df = team_df.append(row, ignore_index=True)

team_df.plot(x="turn", y="pts", kind="line")
st.pyplot()

