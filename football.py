'''
Created on 26. jul. 2018

@author: Borut
'''
import numpy as np
import pandas as pd
import datetime
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import binom_test
import unittest
from sklearn.naive_bayes import GaussianNB
import queue
import collections
import re
#import Orange3
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pickle
import mysql.connector
from sqlalchemy import create_engine


class Helpers:

    @staticmethod
    def expected_value(odds,prob):
        #we assume odds are decimal and our stake is one
        return (odds-1)*prob-(1-prob)

    @staticmethod
    def naive_bayes(prob,prob_array):
        ex=1
        ex_not=1
        ev=1
        for p in prob_array:
            ex*=p[0]/prob[0]
            ex_not*=(p[1]-p[0])/(prob[1]-prob[0])
            ev*=p[1]/prob[1]
        prob_over=(prob[0]/prob[1])*ex/ev
        prob_under=((prob[1]-prob[0])/prob[1])*ex_not/ev
        prob_sum=prob_over+prob_under
        return [prob_over/prob_sum,prob_under/prob_sum]

    @staticmethod
    def get_relegated_teams(teams,ns):
        ns_teams = ns['home_team_name'].unique()
        return set(teams.keys()) - set(ns_teams)

    @staticmethod
    def get_promoted_teams(teams,ns):
        ns_teams = ns['home_team_name'].unique()
        return set(ns_teams) - set(teams.keys())

    @staticmethod
    def fill_up_promoted_teams(teams,attrs,queues,queue_length):
        for attr in attrs:
            for t in teams:
                length_of_queue = attr[4] if len(attr) > 4 else queue_length
                queues[attr[2]][t]=collections.deque(maxlen=length_of_queue)
                for i in range(queue_length):
                    queues[attr[2]][t].append(attr[3])

class Dump:
    @staticmethod
    def go_through_season(ns, new_attrs, queues):
        ns=ns[ns['status']=='complete']
        #the condition status == complete is here because we want to go only through the games that have already been played, but we still need the other games to calucalte length_of_queue
        for index, game in ns.iterrows():#TODO: you could add some safety check which would ensure all the games before current date are set to complete
            for attr in new_attrs:
                teams = queues[attr[2]]
                # we go through an additional for loop to account for the home and away team
                for ha, haattr in [('home', attr[0]), ('away', attr[1])]:
                    if game[ha + '_team_name'] not in teams:
                        # we use the average predefined in attr[3] if the team has yet to compete
                        ns.loc[index, (ha + '_' + attr[2] + '_pre_game')] = attr[3]
                        length_of_queue = attr[4] if len(attr) > 4 else sum(ns['away_team_name']==game[ha + '_team_name']) + sum(ns['home_team_name']==game[ha + '_team_name'])
                        teams[game[ha + '_team_name']] = collections.deque(maxlen=length_of_queue)
                        # for the promoted teams we do not have data so we add the average predefined in attr[3]
                        for i in range(length_of_queue):
                            teams[game[ha + '_team_name']].append(attr[3])
                        #this is a little bit confusing to understand, but it really adds the value of the current game as well to the new queue
                        if type(haattr) is list:
                            lmbd_attr = attr[5](game[haattr[0]], game[haattr[1]])
                            teams[game[ha + '_team_name']].append(lmbd_attr)
                        else:
                            teams[game[ha + '_team_name']].append(game[haattr])
                    else:
                        ns.loc[index, (ha + '_' + attr[2] + '_pre_game')] = np.mean(teams[game[ha + '_team_name']])
                        if type(haattr) is list:
                            lmbd_attr = attr[5](game[haattr[0]], game[haattr[1]])
                            teams[game[ha + '_team_name']].append(lmbd_attr)
                        else:
                            teams[game[ha + '_team_name']].append(game[haattr])
        return ns,queues

    def add_shot_pre_game(df):
        home_shots_avg = []
        away_shots_avg = []
        teams_shots = dict()
        for index, match in df.iterrows():
            if match['home_team_name'] not in teams_shots:
                # we have to setup an entry in the dictionary
                teams_shots[match['home_team_name']] = []
                teams_shots[match['home_team_name']].append(match['home_team_shots'])
                home_shots_avg.append(None)
            else:
                home_shots_avg.append(np.mean(teams_shots[match['home_team_name']]))
                teams_shots[match['home_team_name']].append(match['home_team_shots'])
            if match['away_team_name'] not in teams_shots:
                teams_shots[match['away_team_name']] = []
                teams_shots[match['away_team_name']].append(match['away_team_shots'])
                away_shots_avg.append(None)
            else:
                away_shots_avg.append(np.mean(teams_shots[match['away_team_name']]))
                teams_shots[match['away_team_name']].append(match['away_team_shots'])
        # self.assertEqual(np.sum(teams_shots['Manchester City']),490 )
        # self.assertEqual(home_shots_avg[19],9)
        df['home_shots_pre_game'] = pd.Series(home_shots_avg, index=df.index)
        df['away_shots_pre_game'] = pd.Series(away_shots_avg, index=df.index)
        return df

    def add_avg_attr(df, home_attr, away_attr, attr_name):
        teams_attr = dict()
        home_away_avg = {home_attr: [], away_attr: []}
        for index, match in df.iterrows():
            for name, haattr in [('home_team_name', home_attr), ('away_team_name', away_attr)]:
                if match[name] not in teams_attr:
                    # we have to setup an entry in the dictionary
                    teams_attr[match[name]] = [match[haattr]]
                    home_away_avg[haattr].append(None)
                else:
                    home_away_avg[haattr].append(np.mean(teams_attr[match[name]]))
                    teams_attr[match[name]].append(match[haattr])
        # self.assertEqual(np.sum(teams_shots['Manchester City']),490 )
        # self.assertEqual(home_shots_avg[19],9)
        df['home' + attr_name + '_pre_game'] = pd.Series(home_away_avg[home_attr], index=df.index)
        df['away' + attr_name + '_pre_game'] = pd.Series(home_away_avg[away_attr], index=df.index)
        return df

class Exploration(Helpers):

    def homeaway_analysis(file_name, team_name):
        df = pd.read_csv(file_name, na_values=[""])
        df['home_ratios'] = df['points_per_game_home'] / (df['points_per_game_home'] + df['points_per_game_away'])
        population_ratio = np.mean(df['home_ratios'])
        selected_home = df.loc[df['common_name'] == team_name]
        print(binom_test(40, 80, population_ratio))

    def odds_analysis():
        df = pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv", na_values=[""],
                         parse_dates=['date_GMT'])
        df = df.append(pd.read_csv("europe-uefa-europa-league-matches-2017-to-2018-stats.csv", na_values=[""],
                                   parse_dates=['date_GMT']))
        df = df.append(
            pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv", na_values=[""], parse_dates=['date_GMT']))
        df = df.append(pd.read_csv("england-premier-league-matches-2016-to-2017-stats.csv", na_values=[""],
                                   parse_dates=['date_GMT']))
        df = df.append(pd.read_csv("europe-uefa-champions-league-matches-2017-to-2018-stats.csv", na_values=[""],
                                   parse_dates=['date_GMT']))
        for o in range(13, 30):
            filtered = df[(df['odds_ft_over25'] >= o * 0.1) & ((o + 1) * 0.1 > df['odds_ft_over25'])]
            print("%.1f--%.1f" % (o * 0.1, (o + 1) * 0.1))
            if (len(filtered) > 0):
                # print out how many games with specific odds end up with odds above 2.5
                print(len(filtered[filtered['total_goal_count'] > 2]) / len(filtered),
                      len(filtered[filtered['total_goal_count'] > 2]), len(filtered))
            else:
                print("no game with these odds")

    def exploratory(df, attrs):
        df = df[df["status"] == "complete"]
        df["home_team_possession"].isna()
        r_df = pd.DataFrame()
        for attr in attrs:
            h = df.groupby("home_team_name")[attr[0]].apply(list)
            a = df.groupby("away_team_name")[attr[1]].apply(list)
            for i in h.index:
                h[i].extend(a[i])
            h = h.apply(lambda x: sum(x) / len(x))
            r_df[attr[0][5:]] = h
        return r_df

    def prvaliga_analysis():
        european_games = ['12 July 2017', '19 July 2017', '26 July 2017', '2 August 2017', '16 August 2017',
                          '22 August 2017',
                          '13 September 2017', '26 September 2017', '17 October 2017', '1 November 2017',
                          '21 November 2017', '6 December 2017']
        df = pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv", na_values=[""], parse_dates=['date_GMT'])
        # mask=(df['home_team_name']=='Maribor') | (df['away_team_name']=='Maribor'])
        mb = df[(df['home_team_name'] == 'Maribor') | (df['away_team_name'] == 'Maribor')]
        # mb=mb.append(df.loc[df['away_team_name']=='Maribor'])
        for game in european_games:
            game_date = datetime.strptime(game, '%d %B %Y')
            mask = (mb['date_GMT'] > game_date)
            next_game = mb.loc[mask].iloc[0]
            print(next_game['home_team_name'], next_game['away_team_name'], next_game['home_team_goal_count'],
                  next_game['away_team_goal_count'])
            print(next_game['date_GMT'] - game_date)

    def over25_model():
        df = pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv", na_values=[""], parse_dates=['date_GMT'])
        df['over25'] = df['total_goal_count'] > 2
        gnb = GaussianNB()
        y_pred = gnb.fit([df['home_team_name'], df['away_team_name']], df['over25']).predict(df)
        print("Number of mislabeled points out of a total %d points : %d" % (len(df), (df['over25'] != y_pred).sum()))

    def over25_analysis():
        df = pd.DataFrame()
        slo = pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv", na_values=[""], parse_dates=['date_GMT'])
        eng = pd.read_csv("england-premier-league-matches-2016-to-2017-stats.csv", na_values=[""],
                          parse_dates=['date_GMT'])
        eng18 = pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv", na_values=[""],
                            parse_dates=['date_GMT'])
        df = df.append(slo)
        df = df.append(eng)
        df = df.append(eng18)
        print(df.columns)
        print(np.mean(df['total_goals_at_half_time']))
        print(np.mean(df['total_goal_count'] - df['total_goals_at_half_time']))
        half00 = df.loc[df['total_goals_at_half_time'] == 0]
        print(np.mean(half00['total_goal_count']))


    def cl_qual_analysis():
        df = pd.read_csv("europe-uefa-champions-league-matches-2017-to-2018-stats.csv", na_values=[""],
                         parse_dates=['date_GMT'])
        df = df.append(pd.read_csv("europe-uefa-europa-league-matches-2017-to-2018-stats.csv", na_values=[""],
                                   parse_dates=['date_GMT']))
        df = df[(df['date_GMT'] < '2017-08-25')]

        m = (pd.DataFrame(np.sort(df[['home_team_name', 'away_team_name']], axis=1), index=df.index).duplicated(
            keep='first'))
        n = ~m
        first_leg = df[m]
        second_leg = df[n]

        print(np.mean(first_leg['total_goal_count']))
        print(np.mean(second_leg['total_goal_count']))
        # print(df.iloc[(0:3,20:23),:])

        """
        print(len(df))
        print("home: ", sum(df['home_team_goal_count']>df['away_team_goal_count'])/len(df), "draws: ",sum(df['home_team_goal_count']==df['away_team_goal_count'])/len(df),"away: ",sum(df['home_team_goal_count']<df['away_team_goal_count'])/len(df))
    
        print("0-0 at half-time :",sum(df['total_goals_at_half_time']==0 ))
        print("at end 0-0: ", sum(df['total_goal_count']==0)/sum(df['total_goals_at_half_time']==0 ))
        half00=df[(df['total_goals_at_half_time']==0)]
        print("more then 1 goal: ",sum(half00['total_goal_count']>=1)/len(half00))
        print("more then 2 goal: ",sum(half00['total_goal_count']>=2)/len(half00))
        print("more then 3 goal: ",sum(half00['total_goal_count']>=3)/len(half00))
        """

    def epl_modify():
        teams = pd.read_csv('england-premier-league-teams-2016-to-2017-stats.csv')
        matches = pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv", na_values=[""],
                              parse_dates=['date_GMT'])
        teams = teams.set_index('common_name')
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.expand_frame_repr', False)
        # for cl in matches.columns:
        #   print(cl)
        # 'over25_count_home' 'over25_count_away'

        matches['second_half_total_goal_count'] = matches['total_goal_count'] - matches['total_goals_at_half_time']

        print(len(matches.loc[matches['total_goal_count'] > 2]) / len(matches))
        over_away = len(matches.loc[(matches['away_team_name'] == 'West Ham United') & (matches['total_goal_count'] > 2)])
        away = len(matches.loc[(matches['away_team_name'] == 'West Ham United')])
        over_home = len(matches.loc[(matches['home_team_name'] == 'Newcastle United') & (matches['total_goal_count'] > 2)])
        home = away = len(matches.loc[(matches['home_team_name'] == 'Newcastle United')])
        print(naive_bayes([len(matches.loc[matches['total_goal_count'] > 2]), len(matches)],
                          [(over_away, away), (over_home, home)]))
        # print(teams.loc['Arsenal'])
        # matches.to_csv("england_modified.csv")


    def epl_analysis():
        teams = pd.read_csv("team stats/england-premier-league-teams-2017-to-2018-stats.csv", na_values=[""])
        matches = pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv", na_values=[""],
                              parse_dates=['date_GMT'])
        league = pd.read_csv("england-premier-league-league-2017-to-2018-stats.csv", na_values=[""])
        x = [[], []]

        for index, match in matches.iterrows():
            position_difference = math.fabs(
                teams.loc[teams['common_name'] == match['home_team_name']]['league_position'].item() -
                teams.loc[teams['common_name'] == match['away_team_name']]['league_position'].item())
            average_total_goal_count = (teams.loc[teams['common_name'] == match['home_team_name']][
                                            'total_goal_count'].item() +
                                        teams.loc[teams['common_name'] == match['away_team_name']][
                                            'total_goal_count'].item()) / 2.0

            x[0].append(int((position_difference - 1) / 10))
            x[1].append(average_total_goal_count)

        X = pd.DataFrame(
            {'position_difference': x[0]  # ,
             # 'avg_total_goal_count': x[1]#,
             # 'total_goal_count': matches["total_goal_count"]
             })
        matches['position_difference'] = X['position_difference']
        print(np.mean(matches[matches['position_difference'] == 0]['total_goal_count']))
        print(np.mean(matches[matches['position_difference'] == 1]['total_goal_count']))
        print(np.mean(matches[matches['position_difference'] == 2]['total_goal_count']))
        print(np.mean(matches[matches['position_difference'] == 3]['total_goal_count']))
        plt.scatter(X['position_difference'], matches["total_goal_count"])

        y = matches["total_goal_count"]
        X = sm.add_constant(X)
        print(X)
        # Note the difference in argument order
        model = sm.OLS(y, X)
        res = model.fit()
        # predictions = model.predict(X) # make the predictions by the model
        # Print out the statistics
        x = np.array([min(X['position_difference']), max(X['position_difference'])])
        # fit function
        f = lambda x: res.params[1] * x + res.params[0]
        plt.plot(x, f(x), c="orange")
        plt.show()
        # print(res._results)
        print(res.summary())


class TemporalExploration:

    def prob_over25_minute(minute, df):
        h_goal_minutes = df['home_team_goal_timings'].str.split(',', expand=True)
        a_goal_minutes = df['away_team_goal_timings'].str.split(',', expand=True)
        # accounting for NA values
        h_goal_minutes = h_goal_minutes.fillna("0")
        a_goal_minutes = a_goal_minutes.fillna("0")
        # Handling the minutes in stoppage time like 90'4
        for c in h_goal_minutes.columns:
            h_goal_minutes[c] = h_goal_minutes[c].apply(lambda x: convert_added_time(x))
        for c in a_goal_minutes.columns:
            a_goal_minutes[c] = a_goal_minutes[c].apply(lambda x: convert_added_time(x))
        # goal_minutes = df['home_team_goal_timings'].str.split(',', expand=True)
        # goal_minutes['away_team_goal_timings'] = df['away_team_goal_timings'].str.split(',', expand=True)
        # the games that end with more then 25 goals in a boolean array
        over25 = df[df['total_goal_count'] > 2]

        # the total number of games we save in a variable for further comparisons
        n_games = len(df)
        one_goal_set2 = set()
        to_delete = set()
        for i in range(1, minute):
            # for x in df['home_team_goal_timings']:
            #    print(x)
            h_i = h_goal_minutes.loc[h_goal_minutes.eq(i).any(1)]
            a_i = a_goal_minutes.loc[a_goal_minutes.eq(i).any(1)]

            # sometimes there are two goals in the same minute, such do not get inserted in the one goal set
            same_minute_goals = set(h_i[h_i.eq(i).sum(1).gt(1)].index.values) | set(
                a_i[a_i.eq(i).sum(1).gt(1)].index.values)
            # when they are scored by both teams
            same_minute_goals |= set(h_i.index.values).intersection(set(a_i.index.values))
            # creating a set of indexes we have to delete
            new_goals = set(h_i.index.values) | set(a_i.index.values)
            # new_goals.difference_update(same_minute_goals)
            one_goal_set2 |= (new_goals.difference(same_minute_goals))
            one_goal_set2.difference_update(new_goals.intersection(to_delete))
            to_delete |= new_goals
            to_delete |= same_minute_goals
            # list_one=[int(str(j)) for j in to_delete]

            # one_goal_set1|=to_delete
            # one_goal_set^=to_delete
            # print(one_goal_set)
            # df=df.drop(to_delete)
            # h_goal_minutes=h_goal_minutes.drop(to_delete)
            # a_goal_minutes=a_goal_minutes.drop(to_delete)

        df = df.drop(to_delete)
        prob_ev = len(df) / n_games
        prior_prob = len(over25) / n_games
        bayesian_probability = prior_prob * (sum(df['total_goal_count'] > 2) / len(over25)) / prob_ev
        return [[sum(df['total_goal_count'] > 2) / len(df), bayesian_probability],
                [sum(over25.index.isin(one_goal_set2)) / len(one_goal_set2)]]

class FeatureEngineering:

    def add_result_column(df):
        df["result"] = df.apply(lambda x: 3 if x["home_team_goal_count"] > x["away_team_goal_count"] else 1 if x["home_team_goal_count"] == x["away_team_goal_count"] else 0, axis=1)
        return df

    def modify_csv(filename):
        df = pd.read_csv(filename, na_values=[""])
        # df=add_result_column(df)
        # df=add_shot_pre_game(df)
        df.loc[df['league_position'] == 1, 'type_of_team'] = 'contender'
        df.loc[(df['league_position'] <= 6) & (df['league_position'] > 1), 'type_of_team'] = 'CE'
        df.loc[(df['league_position'] <= 14) & (df['league_position'] > 6), 'type_of_team'] = 'MT'
        df.loc[(df['league_position'] > 14), 'type_of_team'] = 'RB'
        # df=add_avg_attr(df, 'home_team_possession','away_team_possession','avg_possession')
        # df=add_avg_attr(df,'home_team_shots','away_team_shots','avg_shots')
        # df=add_avg_attr(df, 'home_team_goal_count','away_team_goal_count','avg_goals_scored')
        # df=add_avg_attr(df,'away_team_goal_count','home_team_goal_count','avg_goals_conceded')
        # df=add_result_column(df)
        df.to_csv("england_types.csv")

    def convert_added_time(s):
        numbers = re.findall(r'\d+', s)
        if len(numbers)>1:
            #added in the first half we count as minute 45
            if int(numbers[0])==45:
                return 45
            else:
                return (sum(map(int, numbers)))
        else:
            return int(numbers[0])

class CreateDataset(Helpers):
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if CreateDataset.__instance == None:
            CreateDataset()
        return CreateDataset.__instance

    def __init__(self,data_location):
        """ Virtually private constructor. """
        if CreateDataset.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            CreateDataset.__instance = self
            self.data_location=data_location

    @staticmethod
    def make_attribute_queue(attr,season,queue_length=None):
        #TODO: it is probably very inefficient to iterate for every attribute
        teams=dict()
        s=pd.read_csv(season,na_values=[""],parse_dates=['date_GMT'])
        #if the queue length is not set we set it endogeneously as the length of season
        if not queue_length:
            queue_length=len(s)/len(s['home_team_name'].unique())*2
        for index,game in s.iterrows():
            for name,attribute in [('home_team_name',attr[0]),('away_team_name',attr[1])]:
                if game[name] not in teams:
                    #if we require a function to create the attribute it will be presented in a list
                    if type(attribute) is list:
                        teams[game[name]] = collections.deque(maxlen=int(queue_length))
                        lmbd_attr=attr[5](game[attribute[0]],game[attribute[1]])
                        teams[game[name]].append(lmbd_attr)
                    else:
                        teams[game[name]]=collections.deque(maxlen=int(queue_length))
                        teams[game[name]].append(game[attribute])
                else:
                    if type(attribute) is list:
                        lmbd_attr = attr[5](game[attribute[0]], game[attribute[1]])
                        teams[game[name]].append(lmbd_attr)
                    else:
                        teams[game[name]].append(game[attribute])
        return teams



    def data_for_classifier(self,season_before,seasons,new_attrs):
        #we intialize the dictionary of all the attributes that we will keep queues for
        queues=dict()
        #we iterate through all the attributes we want to add, makes attribute queues from previous seasons
        for attr in new_attrs:
            length_of_queue=attr[4] if len(attr)>4 else None
            teams=self.make_attribute_queue(attr,season_before,length_of_queue)
            queues[attr[2]]=teams
        #we intialize the dataframe that will carry games from all of the seasons
        #df=pd.DataFrame()
        all_dfs=[]
        for season in seasons:
            # we read the season in to pandas dataframe with name ns(new season)
            ns = pd.read_csv(season, na_values=[""], parse_dates=['date_GMT'])

            # we delete all the teams that were relegated the previous season. The variable teams is defined in the for loop maybe not the best coding
            relegated_teams =self.get_relegated_teams(teams,ns)
            for attr in new_attrs:
                teams = queues[attr[2]]
                #we delete the relegated teams from each specfic attribute
                for rt in relegated_teams:
                    del teams[rt]

            ns,queues= CreateDataset.go_through_season_apply(ns,new_attrs,queues)

            #we append the modified ns to dataframe
            #df = pd.concat([df,ns],ignore_index=True)
            all_dfs.append(ns)
        df=pd.concat(all_dfs)
        return df

    def attr_iter(home_team_name, away_team_name, home_team_attr, away_team_attr, queues,attr2):
        #we are updating the queue however this will have an effect outside the function, as they are passed by reference=
        pre_game_attr=np.mean(queues[attr2][home_team_name])
        away_pre_game_attr=np.mean(queues[attr2][away_team_name])
        queues[attr2][home_team_name].append(home_team_attr)
        queues[attr2][away_team_name].append(away_team_attr)
        return {'home_' + attr2 + '_pre_game':pre_game_attr, 'away_' + attr2 + '_pre_game':away_pre_game_attr}

    @staticmethod
    def go_through_season_apply(ns, new_attrs, queues):
        ns = ns[ns['status'] == 'complete']
        #attr=('home_team_possession', 'away_team_possession', 'average_possession', 45, 38)
        #teams=queues[attr[2]]
        Helpers.fill_up_promoted_teams(Helpers.get_promoted_teams(queues["total_goal_count"],ns),new_attrs,queues,38)
        for attr in new_attrs:
            #we insert two new columns, where we will insert the new data
            ns.insert(len(ns.columns), 'home_' + attr[2] + '_pre_game', pd.Series())
            ns.insert(len(ns.columns), 'away_' + attr[2] + '_pre_game', pd.Series())
            #normal attributes
            if not type(attr[0]) is list:
                ns.loc[:,['home_' + attr[2] + '_pre_game','away_' + attr[2] + '_pre_game']] = ns.apply(lambda row: CreateDataset.attr_iter(row['home_team_name'],row['away_team_name'],row[attr[0]],row[attr[1]],queues,attr[2]),axis=1,result_type='expand')
            #attributes that use some type of lambda function
            else:
                lmbd=attr[5]
                ns.loc[:, ['home_' + attr[2] + '_pre_game', 'away_' + attr[2] + '_pre_game']] = ns.apply(lambda row: CreateDataset.attr_iter(row['home_team_name'], row['away_team_name'], lmbd(row[attr[0][0]],row[attr[0][1]]), lmbd(row[attr[1][0]],row[attr[1][1]]), queues,attr[2]), axis=1, result_type='expand')
        return ns,queues

    def define_folder_sturcture(self,path=False):
        if path:
            self.data_location=path
        leagues_list=[]
        for fl in os.listdir(self.data_location):
            seasons=sorted(os.listdir(os.path.join(self.data_location,fl)))
            leagues_list.append([os.path.join(self.data_location,fl,s) for s in seasons])
        return leagues_list

    def control_create_dataset(self):
        folder_structure=self.define_folder_sturcture()
        dfs=[]
        for seasons in folder_structure:
            dfs.append(self.data_for_classifier(seasons[0],seasons[1:],game_attributes.attrs))
        df=pd.concat(dfs,ignore_index=True)
        df.to_csv("classifier.csv")
        return df

    def regression_test():
        df=pd.read_csv("classifier.csv", na_values=[""])

        X = pd.DataFrame()
        attrs=['home_total_goal_count_pre_game',
            'away_total_goal_count_pre_game',
            'home_goals_scored_pre_game',
            'away_goals_scored_pre_game',
            'home_goals_conceded_pre_game',
            'away_goals_conceded_pre_game',
               'home_ppg']
        y = df["total_goal_count"]
        #X = sm.add_constant(X)
        #plt.scatter(X['position_difference'], df["total_goal_count"])

        best_rsq=0.0
        #the infinite loop will run until new attributes improve the fit of regression
        while True:
            #we assign the new_attr to None which will be checked in the escape condition
            new_attr=None
            for attr in attrs:
                # we make a copy of the dataframe for one iteration of adding attributes
                i_X = X.copy()
                i_X[attr]=df[attr]
                i_X = sm.add_constant(i_X)

                model = sm.OLS(y, i_X)
                res = model.fit()
                #we check if our new model gives ous a better rsquared value
                if res.rsquared>best_rsq:
                    best_rsq=res.rsquared
                    new_attr=attr
            #escape condition if the new attr is not defined
            if new_attr is None:
                break
            else:
                print("adding ", new_attr, "rsquared now",best_rsq)
                attrs.remove(new_attr)
                X[new_attr]=df[new_attr]
        X=sm.add_constant(X)
        model = sm.OLS(y, X)
        return model.fit()

    def time_analysis_over25(file_name):
        df = pd.read_csv(file_name, na_values=[""])
        len_all_games=len(df)
        df=df[df['total_goals_at_half_time']==0]
        print(len(df)/len_all_games," of games are 0-0 at half time")
        len_over25_00=len(df[df['total_goal_count']>2])
        len_under25_00=len(df[df['total_goal_count']<=2])
        print(len_over25_00/len(df),"of games with 0-0 at half time finis over 2.5")
        print(len_under25_00/len(df),"of games with 0-0 at half time finis under 2.5")

    def teams_goal_timings_distribution(file_name,team):
        df=pd.read_csv(file_name)
        teams= df.dropna(subset=["home_team_goal_timings"]).groupby("home_team_name")["home_team_goal_timings"].apply(lambda x: list(map(convert_added_time, ",".join(x).split(",")))).to_dict()
        teams_a= df.dropna(subset=["away_team_goal_timings"]).groupby("away_team_name")["away_team_goal_timings"].apply(lambda x: list(map(convert_added_time, ",".join(x).split(",")))).to_dict()
        for t in teams:
            teams[t].extend(teams_a[t])
        hsts=pd.Series(teams[team])
        hsts.plot.hist(bins=20)
        plt.show()




    def add_travel_distance():
        #TODO: open data for classifier and go through all games and add attribute distance travelled.
        geolocator = Nominatim(user_agent="specify_your_app_name_here")
        with open('stadiums.pkl', 'rb') as f:
            data = pickle.load(f)
        location = geolocator.geocode(data["Roma"])
        location2=geolocator.geocode(data["Maribor"])
        distance=geodesic((location.latitude,location.longitude),(location2.latitude,location2.longitude))
        print(distance)

    def add_to_mysql():
        engine = create_engine(
            'mysql+mysqlconnector://root:test@127.0.0.1:3306/sandbox', echo=False)
        df = pd.read_csv("classifier.csv")
        df.to_sql(name='test', con=engine, if_exists='replace', index=False)

    def pandas_to_orange(df):
        return

    def prepare_data():
        os.chdir("classifier")
        leagues = os.listdir()
        dfs=[]
        for l in leagues:
            os.chdir(l)
            files = os.listdir()
            season_before = files[0]
            seasons=files[1:]
            ndf=CreateDataset.data_for_classifier(season_before,seasons,game_attributes.attrs)
            dfs.append(ndf)

            #we go back to the classifier folder
            os.chdir('..')
        df=pd.concat(dfs,ignore_index=True)
        df['home_shots_per_goal_pre_game']=df['home_average_shots_pre_game']/df['home_goals_scored_pre_game']
        df['away_shots_per_goal_pre_game']=df['away_average_shots_pre_game']/df['away_goals_scored_pre_game']
        df['home_shots_against_per_goal_pre_game']=df['home_average_shots_against_pre_game']/df['home_goals_conceded_pre_game']
        df['away_shots_against_per_goal_pre_game']=df['away_average_shots_against_pre_game']/df['away_goals_conceded_pre_game']
        #pd.set_option('display.max_columns', None)
        #pd.set_option('display.max_colwidth', -1)
        #pd.set_option('display.width', 2000)
        os.chdir("..")
        df.to_csv("classifier.csv")


#we go through all the minutes 100 is aproximation of max added time, we could do this endogeneously
#prob=prob_over25_minute(45,pd.read_csv("classifier.csv", na_values=[""]))
#print(prob[0],prob[1])
#time_analysis_over25("classifier.csv")
#res= regression_test() #function that returns linear regression model that has the highest rsquraed from a group of attributes.
#epl_analysis()
#print(naive_bayes(prob,prob_array))
#season_before='classifier/italy-serie-a-matches-2014-to-2015-stats.csv'
#seasons=['classifier/italy-serie-a-matches-2015-to-2016-stats.csv','classifier/italy-serie-a-matches-2016-to-2017-stats.csv','classifier/italy-serie-a-matches-2017-to-2018-stats.csv']#if __name__ == '__main__':

class game_attributes:
    attrs = []
    attrs.append(('total_goal_count', 'total_goal_count', 'total_goal_count', 2.54, 38))
    attrs.append(('home_team_goal_count', 'away_team_goal_count', 'goals_scored', 0.9, 38))
    attrs.append(('away_team_goal_count', 'home_team_goal_count', 'goals_conceded', 1.5, 38))#The bundesliga does not have 38 TODO: change so the queues are adapted to length of season
    attrs.append(('home_team_possession', 'away_team_possession', 'average_possession', 45, 38))
    attrs.append(('total_goal_count', 'total_goal_count', 'total_goal_count_5games', 2.54, 5))
    attrs.append(('home_team_shots', 'away_team_shots', 'average_shots', 6, 38))
    attrs.append(('away_team_shots', 'home_team_shots', 'average_shots_against', 12, 38))
    lmbd_points = lambda x, y: 3 if x > y else 0 if x < y else 1
    ppg = (['home_team_goal_count', 'away_team_goal_count'],
           ['away_team_goal_count', 'home_team_goal_count' ], 'ppg', 1, 38, lmbd_points)
    attrs.append(ppg)
    ppg5 = (['home_team_goal_count', 'away_team_goal_count'],
            ['away_team_goal_count', 'home_team_goal_count'], 'ppg_5games', 1, 5, lmbd_points)
    attrs.append(ppg5)




#if __name__ == '__main__':
    #Exploration.teams_goal_timings_distribution("england-premier-league-matches-2017-to-2018-stats.csv","Arsenal")
#print(df)
#matches=pd.read_csv("classifier/italy-serie-a-matches-2014-to-2015-stats.csv",na_values=[""])
#juve=matches.loc[(matches['home_team_name']=='Juventus')| (matches['away_team_name']=='Juventus')]
#teams=data_for_classifier(season_before,seasons)
