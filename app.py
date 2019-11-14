import functools
import operator
import numpy as np
import pandas as pd
import streamlit as st
from fantadata import Fantadata
from statsmodels.tsa.ar_model import AR

#@st.cache(allow_output_mutation=True)
def init_data():
    print("Initializing Fantadata Instance")
    return Fantadata()

def next_turn_pred(team_df):
    data = team_df.pts.values.tolist()
    model = AR(data)
    model_fit = model.fit()
    yhat = model_fit.predict(start=len(data), end=len(data))
    row = pd.DataFrame({'turn': len(team_df)+1, 'pts': yhat, 'pts_rank': np.nan})
    team_df = team_df.append(row, ignore_index=True)
    
    return team_df, yhat[0]
    
fd = init_data()

teams = fd.get_teams()

st.markdown("# ⚽⚽ Nevian Cup Sfiga Stats ⚽⚽")

st.markdown("""
    
    ### Sfiga Rank
    
    * `norm_pt_tot`: total points normalized on goal scored
    * `Rank_Drift`: difference between _Championship_ and _Total Points_ rankings. Lower means higher sfiga.

""")

turn = st.slider("Giornata:", 1, fd.get_played_turns(), value=fd.get_played_turns(), step=1)
rank_df = fd.build_rank(turn=turn)

st.write(rank_df)

st.markdown("""
    
    ### Next Match Prediction
    
    Let's play a bit.
    
    Here it is a simple AR model estimating team performance for the next match based on past matches.
    Following, next turn predicted results.
    
""")

team = st.selectbox("Squadra:", teams)

team_df = fd.get_team_timeseries(team)
team_df, prediction = next_turn_pred(team_df)
team_df.plot(x="turn", y="pts", kind="line")
st.pyplot()

next_match_df = fd.get_next_turn()

next_match_df.home_pts = next_match_df.apply(lambda x: round(next_turn_pred(fd.get_team_timeseries(x.home))[1]* 2) / 2, axis=1)
next_match_df.away_pts = next_match_df.apply(lambda x: round(next_turn_pred(fd.get_team_timeseries(x.away))[1]* 2) / 2, axis=1)
next_match_df.result = next_match_df.apply(lambda x: str(fd.goal_calc(x.home_pts)) + "-" + str(fd.goal_calc(x.away_pts)), axis=1)

st.write(next_match_df[["home", "result", "away", "home_pts", "away_pts"]])