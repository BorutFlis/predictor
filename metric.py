from sklearn.metrics import brier_score_loss
import pandas as pd
import football

class betting_metrics:
    def test_brier(self,file_location):
        df=pd.read_csv(file_location)
        train_df=df.iloc[:int(len(df)/2),:]
        test_df=df.iloc[int(len(df)/2):,:]
        test_df.insert(len(test_df.columns),"discounter",(1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).sum(axis=1))
        prob_odds= (1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).div(test_df["discounter"],axis=0)
        pm=football.PoissonModel()
        pm.fit(train_df)
        prob_p=pm.predict_proba(test_df)
        return_list=[]
        for name,probs in [["odds",prob_odds],["poisson",prob_p]]:
            brier_score_home=brier_score_loss(test_df["home_team_goal_count"]>test_df["away_team_goal_count"],probs.iloc[:,0])
            brier_score_draw=brier_score_loss(test_df["home_team_goal_count"]==test_df["away_team_goal_count"],probs.iloc[:,1])
            brier_score_away=brier_score_loss(test_df["home_team_goal_count"]<test_df["away_team_goal_count"],probs.iloc[:,2])
            print(f"{name} {brier_score_home} {brier_score_draw} {brier_score_away}")
            return_list.append([name, brier_score_home, brier_score_draw, brier_score_away])
        return return_list

