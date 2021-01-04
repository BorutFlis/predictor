from sklearn.metrics import brier_score_loss, accuracy_score
import pandas as pd
import models

class betting_metrics:

    def __init__(self):
        self.complete_files=["classifier\\france\\france-ligue-1-matches-2018-to-2019-stats.csv","weekend\\previous_seasons\\england-premier-league-matches-2019-to-2020-stats (1).csv","weekend\\previous_seasons\\italy-serie-a-matches-2019-to-2020-stats.csv","weekend\\previous_seasons\\spain-la-liga-matches-2019-to-2020-stats.csv"]

    def comprehensive_test(self,files):
        predict_df=[]
        probs_p=[]
        probs_pa=[]
        probs_odds=[]
        for f in files:
            df = pd.read_csv(f)
            train_df = df.iloc[:int(len(df) / 2), :]
            test_df = df.iloc[int(len(df) / 2):, :]
            test_df.insert(len(test_df.columns), "discounter",(1 / test_df.loc[:, ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"]]).sum(axis=1))
            prob_odds = (1 / test_df.loc[:, ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"]]).div(test_df["discounter"], axis=0)
            prob_odds_over25=test_df.apply(lambda x: [1/x["odds_ft_over25"],1-1/x["odds_ft_over25"]],axis=1,result_type="expand")
            round_len = int(len(test_df["home_team_name"].unique()) / 2)
            pm = models.PoissonModel()
            pm.fit(train_df)
            # we also make predictions with a Poisson model that is adaptable
            prob_p = pm.predict_proba(test_df)
            prob_p_over25=pm.predict_proba_over(test_df,2)
            probs_p.append(pd.concat([prob_p.reset_index(),prob_p_over25.reset_index()], axis=1))
            prob_pa = pm.predict_proba(test_df.iloc[:round_len, :])
            prob_pa_over25=pm.predict_proba_over(test_df.iloc[:round_len, :],2)
            for i in range(round_len, len(test_df), round_len):
                pm.fit(pd.concat([train_df, test_df.iloc[:i]]))
                prob_pa = pd.concat([prob_pa, pm.predict_proba(test_df.iloc[i:(i + round_len), :])])
                prob_pa_over25 = pd.concat([prob_pa_over25, pm.predict_proba_over(test_df.iloc[i:(i + round_len), :],2)])
            new_predict_df = pd.concat([test_df.reset_index(), prob_pa.reset_index(),prob_pa_over25.reset_index()], axis=1)
            predict_df.append(test_df.reset_index())
            probs_pa.append(pd.concat([prob_pa.reset_index(),prob_pa_over25.reset_index()], axis=1))
            probs_odds.append(pd.concat([prob_odds, prob_odds_over25],axis=1))
        predict_df=pd.concat(predict_df,axis=0,ignore_index=True)
        probs_pa=pd.concat(probs_pa,axis=0,ignore_index=True)
        probs_p=pd.concat(probs_p,axis=0,ignore_index=True)
        probs_odds = pd.concat(probs_odds, axis=0, ignore_index=True)
        probs_odds.columns=["win","draw","away","over","under"]
        return_list=[]
        for name,probs in [["poisson",probs_p],["poisson adapt",probs_pa],["odds",probs_odds]]:
            brier_score_home=brier_score_loss(predict_df["home_team_goal_count"]>predict_df["away_team_goal_count"],probs.loc[:,"win"])
            brier_score_draw=brier_score_loss(predict_df["home_team_goal_count"]==predict_df["away_team_goal_count"],probs.loc[:,"draw"])
            brier_score_away=brier_score_loss(predict_df["home_team_goal_count"]<predict_df["away_team_goal_count"],probs.loc[:,"away"])
            brier_score_25=brier_score_loss(predict_df["total_goal_count"]>2,probs.loc[:,"over"])
            results_type = predict_df.apply(lambda x: "win" if (x["home_team_goal_count"] > x["away_team_goal_count"]) else "draw" if (x["home_team_goal_count"] == x["away_team_goal_count"]) else "away", axis=1)
            over25= predict_df.apply(lambda x: "over" if (x["total_goal_count"] > 2) else "under",axis=1)
            predict_df["results"]=results_type.reset_index(drop=True)
            ca=accuracy_score(results_type,probs.loc[:,["win","draw","away"]].idxmax(axis=1))
            ca_25=accuracy_score(over25, probs.loc[:,["over","under"]].idxmax(axis=1))
            profit=models.Helpers.hypotetical_betting(pd.concat([predict_df,probs],axis=1)) if name!="odds" else None
            profit25=models.Helpers.hypotetical_betting_25(pd.concat([predict_df,probs],axis=1)) if name!="odds" else None
            print(f"{name} {brier_score_home} {brier_score_draw} {brier_score_away} {brier_score_25} {ca} {ca_25} {profit} {profit25}")
            return_list.append([name, brier_score_home, brier_score_draw, brier_score_away, brier_score_25, ca, ca_25, profit, profit25])
        df = pd.DataFrame(return_list,columns=["index", "brier home", "brier draw", "brier away","brier over25","classification accuracy", "classification accuracy 25","profit 1x2","profit25"])
        df.set_index("index", inplace=True)
        return df


    def hypotetical_profit(self,file_location):
        df = pd.read_csv(file_location)
        train_df = df.iloc[:int(len(df) / 2), :]
        test_df = df.iloc[int(len(df) / 2):, :]
        pm = models.PoissonModel()
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
        print(models.Helpers.betting_accuracy(predict_df))

    def test_brier(self,file_location):
        df=pd.read_csv(file_location)
        train_df=df.iloc[:int(len(df)/2),:]
        test_df=df.iloc[int(len(df)/2):,:]
        test_df.insert(len(test_df.columns),"discounter",(1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).sum(axis=1))
        prob_odds= (1/test_df.loc[:,["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]]).div(test_df["discounter"],axis=0)
        prob_odds.columns = list(range(len(prob_odds.columns)))
        pm=models.PoissonModel()
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

