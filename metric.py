from sklearn.metrics import brier_score_loss, accuracy_score
import pandas as pd
import football

class betting_metrics:

    def hypotetical_profit(self,file_location):
        df = pd.read_csv(file_location)
        train_df = df.iloc[:int(len(df) / 2), :]
        test_df = df.iloc[int(len(df) / 2):, :]
        pm = football.PoissonModel()
        pm.fit(train_df)
        round_len = int(len(test_df["home_team_name"].unique()) / 2)
        # we also make predictions with a Poisson model that is adaptable
        prob_pa = pm.predict_proba(test_df.iloc[:round_len, :])
        for i in range(round_len, len(test_df), round_len):
            pm.fit(pd.concat([train_df, test_df.iloc[:i]]))
            prob_pa = pd.concat([prob_pa, pm.predict_proba(test_df.iloc[i:(i + round_len), :])])
        predict_df=pd.concat([test_df.reset_index(),prob_pa.reset_index()],axis=1)
        results_type = test_df.apply(lambda x: "win" if (x["home_team_goal_count"] > x["away_team_goal_count"]) else "draw" if (x["home_team_goal_count"] == x["away_team_goal_count"]) else "away", axis=1)
        predict_df["results"]=results_type.reset_index(drop=True)
        print(football.Helpers.betting_accuracy(predict_df))

    def test_brier(self,file_location):
        df=pd.read_csv(file_location)
        train_df=df.iloc[:int(len(df)/2),:]
        test_df=df.iloc[int(len(df)/2):,:]
        test_df.insert(len(test_df.columns),"discounter",(1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).sum(axis=1))
        prob_odds= (1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).div(test_df["discounter"],axis=0)
        prob_odds.columns = list(range(len(prob_odds.columns)))
        pm=football.PoissonModel()
        pm.fit(train_df)
        prob_p=pm.predict_proba(test_df)
        round_len=int(len(test_df["home_team_name"].unique())/2)
        #we also make predictions with a Poisson model that is adaptable
        prob_pa=pm.predict_proba(test_df.iloc[:round_len,:])
        for i in range(round_len,len(test_df),round_len):
            pm.fit(pd.concat([train_df,test_df.iloc[:i]]))
            prob_pa=pd.concat([prob_pa,pm.predict_proba(test_df.iloc[i:(i+round_len),:])])
        return_list=[]
        results_type = test_df.apply(lambda x: 0 if (x["home_team_goal_count"] > x["away_team_goal_count"]) else 1 if (x["home_team_goal_count"] == x["away_team_goal_count"]) else 2, axis=1)
        for name,probs in [["odds",prob_odds],["poisson",prob_p],["poisson_adapt",prob_pa]]:
            brier_score_home=brier_score_loss(test_df["home_team_goal_count"]>test_df["away_team_goal_count"],probs.iloc[:,0])
            brier_score_draw=brier_score_loss(test_df["home_team_goal_count"]==test_df["away_team_goal_count"],probs.iloc[:,1])
            brier_score_away=brier_score_loss(test_df["home_team_goal_count"]<test_df["away_team_goal_count"],probs.iloc[:,2])
            probs.columns = list(range(len(probs.columns)))
            ca=accuracy_score(results_type,probs.idxmax(axis=1))
            print(f"{name} {brier_score_home} {brier_score_draw} {brier_score_away} {ca}")
            return_list.append([name, brier_score_home, brier_score_draw, brier_score_away, ca])
        df = pd.DataFrame(return_list, columns=["index", "brier home", "brier draw", "brier away", "classification accuracy"])
        df.set_index("index",inplace=True)
        return df

