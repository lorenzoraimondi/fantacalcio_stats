import numpy as np
import pandas as pd


class Fantadata:
    
    N_FANTA_TEAMS = 10
    N_REAL_TEAMS = 20
    N_TURNS = 36
    N_MATCHES = int(N_FANTA_TEAMS / 2)
    PLAYED_TURNS = None
    
    _datafile = "data/Calendario_Nevian-Cup-X.xlsx"
    _data_df = None
    
    def __init__(self):
        df = pd.read_excel(self._datafile, skiprows=2)
        self._data_df, self.PLAYED_TURNS = self._build_matches_df(df)
        self._teams = sorted(self._data_df["home"].unique().tolist())
        
    def get_df(self):
        return self._data_df
        
    def get_teams(self):
        return self._teams
        
    def get_played_turns(self):
        return self.PLAYED_TURNS
    
    def point_calc(self, home_goals, away_goals):
        if home_goals > away_goals:
            return (3, 0)
        elif home_goals < away_goals:
            return (0, 3)
        else:
            return (1, 1)

    def _build_matches_df(self, excel_df):
        col_names = ["home", "home_pts", "away_pts", "away", "result"]

        odd_df = excel_df.iloc[:, :5]
        even_df = excel_df.iloc[:, 6:]

        even_df.rename(dict(zip(even_df.columns.tolist(), col_names)), axis=1, inplace=True)
        odd_df.rename(dict(zip(odd_df.columns.tolist(), col_names)), axis=1, inplace=True)

        matches_df = pd.DataFrame()

        header_row = 1
        n = self.N_MATCHES + header_row
        for i in range(18):
            matches_df = matches_df.append(odd_df.iloc[int(i*n)+1:int((i+1)*n),:])
            matches_df = matches_df.append(even_df.iloc[int(i*n)+1:int((i+1)*n),:])

        turn = list()

        for x in range(self.N_TURNS):
            turn.extend([x+1] * self.N_MATCHES)

        matches_df["turn"] = turn

        goals = matches_df.result.str.split("-", expand=True)
        matches_df["home_goals"] = goals[0]
        matches_df["away_goals"] = goals[1]
        matches_df["home_rank_pts"] = matches_df.apply(lambda x: self.point_calc(x.home_goals, x.away_goals)[0], axis=1)
        matches_df["away_rank_pts"] = matches_df.apply(lambda x: self.point_calc(x.home_goals, x.away_goals)[1], axis=1)
        matches_df = matches_df.replace(r'^\s*$', np.nan, regex=True)

        available_matches_df = matches_df[~matches_df.home_goals.isna()]
        played_turns = int(len(available_matches_df) / self.N_MATCHES)
        
        matches_df["home"] = matches_df["home"].apply(lambda x: x.strip() if x else np.nan)
        matches_df["away"] = matches_df["away"].apply(lambda x: x.strip() if x else np.nan)

        matches_df.reset_index(drop=True, inplace=True)
        
        return matches_df, played_turns

    def build_rank(self, turn=-1):
        data = []
        
        teams = self.get_teams()
        row_filter = int(self.PLAYED_TURNS*self.N_MATCHES)
        available_df = self._data_df.iloc[:row_filter,:]
        
        if 0 < turn < self.PLAYED_TURNS:
            turn_filter = int(turn*self.N_MATCHES)
        else:
            turn_filter = len(available_df)
        
        for t in teams:
            home_matches = available_df.iloc[:turn_filter,:][available_df.home == t]
            away_matches = available_df.iloc[:turn_filter,:][available_df.away == t]

            vh = len(home_matches[home_matches.home_rank_pts == 3])
            nh = len(home_matches[home_matches.home_rank_pts == 1])
            ph = len(home_matches[home_matches.home_rank_pts == 0])
            va = len(away_matches[away_matches.away_rank_pts == 3])
            na = len(away_matches[away_matches.away_rank_pts == 1])
            pa = len(away_matches[away_matches.away_rank_pts == 0])
            gfh = home_matches.home_goals.astype(int).sum()
            gfa = away_matches.away_goals.astype(int).sum()
            gsh = home_matches.away_goals.astype(int).sum()
            gsa = away_matches.home_goals.astype(int).sum()
            drh = gfh - gsh
            dra = gfa - gsa
            pth = home_matches.home_rank_pts.sum()
            pta = away_matches.away_rank_pts.sum()
            pth_tot = home_matches.home_pts.sum()
            pta_tot =away_matches.away_pts.sum()
            v = vh + va
            n = nh + na
            p = ph + pa
            g = v + n + p
            gf = gfh + gfa
            gs = gsh + gsa
            dr = drh + dra
            pt = pth + pta
            pt_tot = pth_tot + pta_tot

            data.append([t, g, v, n, p, gf, gs, dr, pt, pt_tot])

        rank = pd.DataFrame(data, columns=["Team", "g", "v", "n", "p", "gf", "gs", "dr", "pt", "pt_tot"])
        rank.sort_values(by="pt", ascending=False, inplace=True)

        return rank

    def build_ts(self, team):
        df = self._data_df[(self._data_df.home == team) | (self._data_df.away == team)]

        df["goals"] = df.apply(lambda x: x.home_goals if x.home == team else x.away_goals, axis=1)
        df["pts"] = df.apply(lambda x: x.home_pts if x.home == team else x.away_pts, axis=1)
        df["rank_pts"] = df.apply(lambda x: x.home_rank_pts if x.home == team else x.away_rank_pts, axis=1)
        df = df[["turn", "goals", "pts", "rank_pts"]]

        return df
