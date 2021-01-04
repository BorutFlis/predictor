import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
import pandas as pd
import numpy as np

class Helpers:

    @staticmethod
    def hypotetical_betting(predict_df):
        predict_df[["bet_win", "bet_draw", "bet_away"]] = \
            predict_df.apply( \
                lambda x: [x["win"] * (x["odds_ft_home_team_win"] - 1) - (1 - x["win"]), \
                           x["draw"] * (x["odds_ft_draw"] - 1) - (1 - x["draw"]), \
                           x["away"] * (x["odds_ft_away_team_win"] - 1) - (1 - x["away"])], axis=1,
                result_type="expand")
        max_ev = predict_df.iloc[:, -3:].max(axis=1)
        bet = predict_df.iloc[:, -3:].idxmax(axis=1)
        predict_df.insert(len(predict_df.columns), "max_ev", max_ev)
        predict_df.insert(len(predict_df.columns), "bet", bet)
        translate_dict = {"bet_win": ["win", "odds_ft_home_team_win"], "bet_draw": ["draw", "odds_ft_draw"],
                          "bet_away": ["away", "odds_ft_away_team_win"]}
        predict_df["odds"] = predict_df.apply(lambda x: x[translate_dict[x["bet"]][1]], axis=1)
        predict_df["bet"] = predict_df.apply(lambda x: x["bet"].split("_")[1], axis=1)
        return predict_df[predict_df.max_ev.gt(0)].apply(lambda x:  x["odds"]-1 if x["results"]==x["bet"] else -1,axis=1).sum()

    @staticmethod
    def hypotetical_betting_25(predict_df):
        predict_df["bet_over"] = \
            predict_df.apply( \
                lambda x: x["over"] * (x["odds_ft_over25"] - 1) - (1 - x["over"]), axis=1,\
                result_type="expand")
        return predict_df[predict_df.bet_over.gt(0)].apply(lambda x: x["odds_ft_over25"] - 1 if x["total_goal_count"] > 2 else -1,axis=1).sum()



class PoissonModel:

    def __init__(self, home_dict={'home_team_name':'team', 'away_team_name':'opponent','home_team_goal_count':'goals'}, away_dict={'away_team_name':'team', 'home_team_name':'opponent','away_team_goal_count':'goals'}):
        self.home_dict=home_dict
        self.away_dict=away_dict
        self.classes_=["win","draw","away"]

    def fit(self,df):
        goal_model_data = pd.concat([
            df[['home_team_name', 'away_team_name', 'home_team_goal_count']].assign(home=1).rename(columns=self.home_dict),
                df[['home_team_name', 'away_team_name', 'away_team_goal_count']].assign(home=0).rename(columns=self.away_dict)])
        self.model=smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,family=sm.families.Poisson()).fit()

    def avg_prediction(self,homeTeam,awayTeam):
        home_goals_avg = self.model.predict(pd.DataFrame(data={'team': homeTeam,
                                                               'opponent': awayTeam, 'home': 1},
                                                         index=[1])).values[0]
        away_goals_avg = self.model.predict(pd.DataFrame(data={'team': awayTeam,
                                                               'opponent': homeTeam, 'home': 0},
                                                         index=[1])).values[0]
        return {"home_pred":home_goals_avg, "away_pred":away_goals_avg}

    def predict_proba(self,df):
        pred_avg=  df.apply(lambda x: self.avg_prediction(x["home_team_name"],x["away_team_name"]),axis=1,result_type="expand")
        df=pd.concat([df,pred_avg],axis=1)
        final_predictions=[]
        for i,row in df.iterrows():
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, 7)] for team_avg in [row["home_pred"], row["away_pred"]]]
            prob_matrix=(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
            h_prob=np.sum(np.tril(prob_matrix, -1))
            draw_prob=np.sum(np.diag(prob_matrix))
            a_prob=np.sum(np.triu(prob_matrix, 1))
            final_predictions.append([h_prob,draw_prob,a_prob])
        return pd.DataFrame(final_predictions,columns=self.classes_)

    def predict_proba_over(self,df,no_of_goals):
        pred_avg=  df.apply(lambda x: self.avg_prediction(x["home_team_name"],x["away_team_name"]),axis=1,result_type="expand")
        df=pd.concat([df,pred_avg],axis=1)
        final_predictions=[]
        for i,row in df.iterrows():
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, 7)] for team_avg in [row["home_pred"], row["away_pred"]]]
            prob_matrix=(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
            test = [prob_matrix[i][:(no_of_goals+ 1 - i)] for i in range(no_of_goals + 1)]
            test2 = [prob_matrix[i][max((no_of_goals+ 1 - i),0):] for i in range(len(prob_matrix))]
            under=sum([sum(row) for row in test])
            over=1-under
            final_predictions.append([under,over])
        return pd.DataFrame(final_predictions,columns=["under","over"])