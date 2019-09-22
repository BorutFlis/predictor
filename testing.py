'''
Created on 8. avg. 2018

@author: Borut
'''
import unittest
import pandas as pd
import football
import math
import numpy as np

class Test(unittest.TestCase):

    def testCreateQueue(self):
        matches = pd.read_csv("classifier/italy-serie-a-matches-2014-to-2015-stats.csv", na_values=[""])
        juve = matches.loc[(matches['home_team_name'] == 'Juventus') | (matches['away_team_name'] == 'Juventus')]
        season_before = 'classifier/italy-serie-a-matches-2014-to-2015-stats.csv'
        seasons = ['classifier/italy-serie-a-matches-2015-to-2016-stats.csv',
                   'classifier/italy-serie-a-matches-2016-to-2017-stats.csv',
                   'classifier/italy-serie-a-matches-2017-to-2018-stats.csv']

        teams=football.make_attribute_queue(['total_goal_count','total_goal_count'], season_before,20)
        self.assertEqual(np.sum(juve[-20:]['total_goal_count']),np.sum(teams['Juventus']) )

    def testProbOver25Minute(self):
        df = pd.read_csv("classifier.csv", na_values=[""])
        #calculating the proballity with the function, 46 is the first minute of the second half, so we test before the 46 minute
        prob=football.prob_over25_minute(46,df)
        #calculating the probability without the function
        df = df[df['total_goals_at_half_time'] == 0]
        len_over25_00 = len(df[df['total_goal_count'] > 2])
        self.assertEqual(len_over25_00 / len(df),prob[0])

    def testLambdaAttrs(self):
        season='classifier/italy-serie-a-matches-2017-to-2018-stats.csv'
        df = pd.read_csv("classifier/italy-serie-a-matches-2017-to-2018-stats.csv", na_values=[""])
        lmbd_points = lambda x, y: 3 if x > y else 0 if x < y else 1
        ppg = ([('home_team_goal_count', 'away_team_goal_count'), lmbd_points],
               [('away_team_goal_count', 'home_team_goal_count'), lmbd_points], 'ppg', 1)
        teams=football.make_attribute_queue(ppg,season)
        self.assertEqual(sum(teams['Napoli']),91)

    def testDataForClassifier(self):

        season_before = 'classifier/ITA/italy-serie-a-matches-2014-to-2015-stats.csv'
        seasons = ['classifier/ITA/italy-serie-a-matches-2015-to-2016-stats.csv',
                   'classifier/ITA/italy-serie-a-matches-2016-to-2017-stats.csv',
                   'classifier/ITA/italy-serie-a-matches-2017-to-2018-stats.csv']
        s=pd.read_csv("classifier/ITA/italy-serie-a-matches-2014-to-2015-stats.csv", na_values=[""])
        s1=pd.read_csv("classifier/ITA/italy-serie-a-matches-2016-to-2017-stats.csv", na_values=[""])
        s2 = pd.read_csv("classifier/ITA/italy-serie-a-matches-2017-to-2018-stats.csv", na_values=[""])
        lmbd_points = lambda x, y: 3 if x > y else 0 if x < y else 1
        new_attrs=[('total_goal_count', 'total_goal_count', 'total_goal_count',2.54,38),
                   ('home_team_goal_count', 'away_team_goal_count', 'goals_scored', 0.9,38),
                   ('away_team_goal_count', 'home_team_goal_count', 'goals_conceded', 1.5,38),
                   ([('home_team_goal_count', 'away_team_goal_count'), lmbd_points],
                    [('away_team_goal_count', 'home_team_goal_count'), lmbd_points], 'ppg', 1,38)
                   ]
        df=football.data_for_classifier(season_before, seasons,new_attrs)
        #testing for the team with data from the season not included in the final output
        self.assertEqual(np.mean(s.loc[(s['home_team_name']=="Juventus")|(s['away_team_name']=="Juventus")]['total_goal_count']),df.iloc[2]['home_total_goal_count_pre_game'])
        #we need to define the data separetely for home/away goal count & goals conceded
        hg=s.loc[(s['home_team_name'] == "Juventus")]['home_team_goal_count']
        ag=s.loc[(s['away_team_name'] == "Juventus")]['away_team_goal_count']
        hc = s.loc[(s['home_team_name'] == "Juventus")]['away_team_goal_count']
        ac = s.loc[(s['away_team_name'] == "Juventus")]['home_team_goal_count']
        self.assertEqual(np.mean(hg.append(ag)),df.iloc[2]['home_goals_scored_pre_game'])
        self.assertEqual(np.mean(hc.append(ac)), df.iloc[2]['home_goals_conceded_pre_game'])
        self.assertEqual((87.0/38.0),df.iloc[2]['home_ppg_pre_game'])
        #testing for the team that got promoted
        self.assertEqual(2.54,df.iloc[387]['away_total_goal_count_pre_game'])
        self.assertEqual(0.9, df.iloc[387]['away_goals_scored_pre_game'])
        self.assertEqual(1.5, df.iloc[387]['away_goals_conceded_pre_game'])
        self.assertEqual(1, df.iloc[387]['away_ppg_pre_game'])
        #testing for the team that got promoted last year and then stayed up
        self.assertEqual(np.mean(s1.loc[(s1['home_team_name']=="Cagliari")|(s1['away_team_name']=="Cagliari")]['total_goal_count']),df.iloc[760]['away_total_goal_count_pre_game'])
        hg = s1.loc[(s1['home_team_name'] == "Cagliari")]['home_team_goal_count']
        ag = s1.loc[(s1['away_team_name'] == "Cagliari")]['away_team_goal_count']
        hc = s1.loc[(s1['home_team_name'] == "Cagliari")]['away_team_goal_count']
        ac = s1.loc[(s1['away_team_name'] == "Cagliari")]['home_team_goal_count']
        self.assertEqual(np.mean(hg.append(ag)), df.iloc[760]['away_goals_scored_pre_game'])
        self.assertEqual(np.mean(hc.append(ac)), df.iloc[760]['away_goals_conceded_pre_game'])
        self.assertEqual((47.0/38.0),df.iloc[760]['away_ppg_pre_game'])
        #testing for the team that continues to play
        self.assertEqual(np.mean(s1.loc[(s1['home_team_name'] == "Juventus") | (s1['away_team_name'] == "Juventus")]['total_goal_count']),df.iloc[760]['home_total_goal_count_pre_game'])
        #testing for second game in season
        #all the games from 201617 appart from the first one
        ps=s1.loc[(s1['home_team_name'] == "Juventus") | (s1['away_team_name'] == "Juventus")].loc[2:]['total_goal_count']
        #changing pandas series to python list
        ps=ps.tolist()
        #appending the first game of 201718
        ps.append(s2.iloc[0]['total_goal_count'])
        self.assertEqual(np.mean(ps),df.iloc[770]['away_total_goal_count_pre_game'])

    def testAddAttribute(self):
        df=pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv",na_values=[""])
        #df=add_result_column(df)
        #df=add_shot_pre_game(df)1
    
        df=football.add_avg_attr(df, 'possession')
        self.assertEqual(70, df.iloc[19]['home_possession_pre_game'])
        self.assertTrue(math.fabs(df.iloc[372]['away_possession_pre_game']-68)<1.0 )
        
        df=football.add_avg_attr(df, 'goal_count')
        self.assertEqual(3, df.iloc[14]['home_goal_count_pre_game'])
        self.assertTrue(math.fabs(df.iloc[372]['away_goal_count_pre_game']-2.789)<0.1 )



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()