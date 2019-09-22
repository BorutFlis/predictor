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



def expected_value(odds,prob):
    #we assume odds are decimal and our stake is one
    return (odds-1)*prob-(1-prob) 

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

def cl_qual_analysis():
    df = pd.read_csv("europe-uefa-champions-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    df = df.append(pd.read_csv("europe-uefa-europa-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT']))
    df=df[(df['date_GMT'] <'2017-08-25')]

    m = (pd.DataFrame(np.sort(df[['home_team_name','away_team_name']], axis=1), index=df.index).duplicated(keep='first'))
    n=~m
    first_leg=df[m]
    second_leg=df[n]

    print(np.mean(first_leg['total_goal_count']))
    print(np.mean(second_leg['total_goal_count']))
    #print(df.iloc[(0:3,20:23),:])
    
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
    teams=pd.read_csv('england-premier-league-teams-2016-to-2017-stats.csv')
    matches=pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    teams=teams.set_index('common_name')
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.expand_frame_repr', False)
    #for cl in matches.columns:
     #   print(cl)
    #'over25_count_home' 'over25_count_away'
    
    matches['second_half_total_goal_count']=matches['total_goal_count']-matches['total_goals_at_half_time']
    
    print(len(matches.loc[matches['total_goal_count']>2])/len(matches))
    over_away=len(matches.loc[(matches['away_team_name']=='West Ham United') & (matches['total_goal_count']>2)])
    away=len(matches.loc[(matches['away_team_name']=='West Ham United')])
    over_home=len(matches.loc[(matches['home_team_name']=='Newcastle United') & (matches['total_goal_count']>2)])
    home=away=len(matches.loc[(matches['home_team_name']=='Newcastle United')])
    print(naive_bayes([len(matches.loc[matches['total_goal_count']>2]),len(matches)],[(over_away,away),(over_home,home)]))
    #print(teams.loc['Arsenal'])
    #matches.to_csv("england_modified.csv")
    

def odds_analysis():
    df=pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    df = df.append(pd.read_csv("europe-uefa-europa-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT']))
    df = df.append(pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT']))
    df = df.append(pd.read_csv("england-premier-league-matches-2016-to-2017-stats.csv",na_values=[""],parse_dates=['date_GMT']))
    df = df.append(pd.read_csv("europe-uefa-champions-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT']))
    for o in range(13,30):
        filtered= df[(df['odds_ft_over25']>=o*0.1) & ((o+1)*0.1>df['odds_ft_over25'])]
        print("%.1f--%.1f" % (o*0.1,(o+1)*0.1))
        if(len(filtered)>0):
            print( len(filtered[filtered['total_goal_count']>2])/len(filtered),len(filtered[filtered['total_goal_count']>2]),len(filtered))
        else:
            print("no game with these odds")
    
def epl_analysis():
    teams=pd.read_csv("team stats/england-premier-league-teams-2017-to-2018-stats.csv",na_values=[""])
    matches=pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    league=pd.read_csv("england-premier-league-league-2017-to-2018-stats.csv",na_values=[""])
    x=[[], []]
    
    for index,match in matches.iterrows():
        position_difference=math.fabs(teams.loc[teams['common_name'] == match['home_team_name']]['league_position'].item()-teams.loc[teams['common_name'] == match['away_team_name']]['league_position'].item())
        average_total_goal_count=(teams.loc[teams['common_name'] == match['home_team_name']]['total_goal_count'].item()+teams.loc[teams['common_name'] == match['away_team_name']]['total_goal_count'].item())/2.0
        
        x[0].append(int((position_difference-1)/10))
        x[1].append(average_total_goal_count)
        
    X = pd.DataFrame(
    {'position_difference': x[0]#,
     #'avg_total_goal_count': x[1]#,
     #'total_goal_count': matches["total_goal_count"]
    })
    matches['position_difference']=X['position_difference']
    print(np.mean(matches[matches['position_difference']==0]['total_goal_count']))
    print(np.mean(matches[matches['position_difference']==1]['total_goal_count']))
    print(np.mean(matches[matches['position_difference']==2]['total_goal_count']))
    print(np.mean(matches[matches['position_difference']==3]['total_goal_count']))
    plt.scatter(X['position_difference'],matches["total_goal_count"])  
    
    y = matches["total_goal_count"]
    X = sm.add_constant(X)
    print(X)
    # Note the difference in argument order
    model = sm.OLS(y,X)
    res = model.fit()
    #predictions = model.predict(X) # make the predictions by the model
        # Print out the statistics
    x = np.array([min(X['position_difference']),max(X['position_difference'])])
    #fit function
    f = lambda x: res.params[1]*x + res.params[0]
    plt.plot(x,f(x), c="orange")
    plt.show()
    #print(res._results)
    print(res.summary())
    
def add_result_column(df):
    result=[]
    for index,match in df.iterrows():
        if match['home_team_goal_count']==match['away_team_goal_count']:
            result.append("draw")
        elif match['home_team_goal_count']>match['away_team_goal_count']:
            result.append("home")
        else:
            result.append("away")
    df['result'] = pd.Series(result, index=df.index) 
    return df   
    
def add_avg_attr(df,home_attr,away_attr,attr_name):
    teams_attr=dict()
    home_away_avg={home_attr: [],away_attr: []}
    for index,match in df.iterrows():
        for name, haattr in [('home_team_name',home_attr),('away_team_name',away_attr)]:
            if match[name] not in teams_attr:
                #we have to setup an entry in the dictionary
                teams_attr[match[name]]=[match[haattr]]
                home_away_avg[haattr].append(None)
            else:
                home_away_avg[haattr].append(np.mean(teams_attr[match[name]]))
                teams_attr[match[name]].append(match[haattr])
    #self.assertEqual(np.sum(teams_shots['Manchester City']),490 )
    #self.assertEqual(home_shots_avg[19],9)    
    df['home'+attr_name+'_pre_game'] = pd.Series(home_away_avg[home_attr], index=df.index)
    df['away'+attr_name+'_pre_game'] = pd.Series(home_away_avg[away_attr],index=df.index)
    return df
    
def add_shot_pre_game(df):
    home_shots_avg=[]
    away_shots_avg=[]
    teams_shots=dict()
    
    for index,match in df.iterrows():
        
        if match['home_team_name'] not in teams_shots:
            #we have to setup an entry in the dictionary
            teams_shots[match['home_team_name']]=[]
            teams_shots[match['home_team_name']].append(match['home_team_shots'])
            home_shots_avg.append(None)
        else:
            home_shots_avg.append(np.mean(teams_shots[match['home_team_name']]))
            teams_shots[match['home_team_name']].append(match['home_team_shots'])
        
        if match['away_team_name'] not in teams_shots:
            teams_shots[match['away_team_name']]=[]
            teams_shots[match['away_team_name']].append(match['away_team_shots'])
            away_shots_avg.append(None)
        else:
            away_shots_avg.append(np.mean(teams_shots[match['away_team_name']])) 
            teams_shots[match['away_team_name']].append(match['away_team_shots'])
    #self.assertEqual(np.sum(teams_shots['Manchester City']),490 )
    #self.assertEqual(home_shots_avg[19],9)    
    df['home_shots_pre_game'] = pd.Series(home_shots_avg, index=df.index)
    df['away_shots_pre_game'] = pd.Series(away_shots_avg,index=df.index)
    return df 

def modify_csv():
    os.chdir("team stats")
    df=pd.read_csv("england-types.csv",na_values=[""])
    #df=add_result_column(df) 
    #df=add_shot_pre_game(df)
    df.loc[df['league_position']==1,'type_of_team']='contender'
    df.loc[(df['league_position']<=6) & (df['league_position']>1),'type_of_team']='CE'
    df.loc[(df['league_position']<=14) & (df['league_position']>6),'type_of_team']='MT'
    df.loc[(df['league_position']>14),'type_of_team']='RB'
    #df=add_avg_attr(df, 'home_team_possession','away_team_possession','avg_possession')
    #df=add_avg_attr(df,'home_team_shots','away_team_shots','avg_shots')
    #df=add_avg_attr(df, 'home_team_goal_count','away_team_goal_count','avg_goals_scored')
    #df=add_avg_attr(df,'away_team_goal_count','home_team_goal_count','avg_goals_conceded')
    #df=add_result_column(df)
    df.to_csv("england_types.csv")
    
def homeaway_analysis():
    os.chdir("team stats")
    df=pd.DataFrame()
    for league in os.listdir():
        print(league)
        newdf=pd.read_csv(league,na_values=[""])
        df=df.append(newdf)
    #print(df.columns)
    
    df['home_ratios'] = df['points_per_game_home']*(df['matches_played']/2) /(df['points_per_game_home']*(df['matches_played']/2)+ df['points_per_game_away']*(df['matches_played']/2))
    
    population_ratio= np.mean(df['home_ratios'])
    aluminij=df.loc[df['common_name']=='Aluminij']
    aluminij_home= aluminij['points_per_game_home']*(aluminij['matches_played']/2)
    
    print(aluminij_home)
    print(aluminij['points_per_game_home']*(aluminij['matches_played']/2)+aluminij['points_per_game_away']*(aluminij['matches_played']/2))
    print(binom_test(40,80 , population_ratio))
    
def prvaliga_analysis():
    european_games=['12 July 2017','19 July 2017','26 July 2017','2 August 2017','16 August 2017','22 August 2017',
                    '13 September 2017','26 September 2017','17 October 2017','1 November 2017','21 November 2017','6 December 2017']
    df=pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    #mask=(df['home_team_name']=='Maribor') | (df['away_team_name']=='Maribor'])
    mb=df[(df['home_team_name'] == 'Maribor') | (df['away_team_name'] == 'Maribor')]
   # mb=mb.append(df.loc[df['away_team_name']=='Maribor'])
    for game in european_games:
        game_date=datetime.strptime(game,'%d %B %Y')
        mask = (mb['date_GMT'] > game_date)
        next_game= mb.loc[mask].iloc[0]
        print(next_game['home_team_name'],next_game['away_team_name'],next_game['home_team_goal_count'],next_game['away_team_goal_count'])
        print(next_game['date_GMT']-game_date)

def over25_model():
    df=pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    df['over25']=df['total_goal_count']>2
    gnb = GaussianNB()
    y_pred = gnb.fit([df['home_team_name'],df['away_team_name']],df['over25']).predict(df)
    print("Number of mislabeled points out of a total %d points : %d" % (len(df),(df['over25'] != y_pred).sum()))

def over25_analysis():
    df=pd.DataFrame()
    slo=pd.read_csv("slovenia-prvaliga-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    eng=pd.read_csv("england-premier-league-matches-2016-to-2017-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    eng18=pd.read_csv("england-premier-league-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
    df=df.append(slo)
    df=df.append(eng)
    df=df.append(eng18)
    print(df.columns)
    print(np.mean(df['total_goals_at_half_time']))
    print(np.mean(df['total_goal_count']-df['total_goals_at_half_time']))
    half00=df.loc[df['total_goals_at_half_time']==0]
    print(np.mean(half00['total_goal_count']))
    
def make_attribute_queue(attr,season,queue_length=None):
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
                    lmbd_attr=attribute[1](game[attribute[0][0]],game[attribute[0][1]])
                    teams[game[name]].append(lmbd_attr)
                else:
                    teams[game[name]]=collections.deque(maxlen=int(queue_length))
                    teams[game[name]].append(game[attribute])
            else:
                if type(attribute) is list:
                    lmbd_attr = attribute[1](game[attribute[0][0]], game[attribute[0][1]])
                    teams[game[name]].append(lmbd_attr)
                else:
                    teams[game[name]].append(game[attribute])
    return teams



def data_for_classifier(season_before,seasons,new_attrs):
    #we intialize the dictionary of all the attributes that we will keep queues for
    queues=dict()
    #we iterate through all the attributes we want to add, makes attribute queues from previous seasons
    for attr in new_attrs:

        length_of_queue=attr[4] if len(attr)>4 else None
        teams=make_attribute_queue([attr[0],attr[1]],season_before,length_of_queue)
        queues[attr[2]]=teams
    #we intialize the dataframe that will carry games from all of the seasons
    #df=pd.DataFrame()
    all_dfs=[]
    for season in seasons:
        # we read the season in to pandas dataframe with name ns(new season)
        ns = pd.read_csv(season, na_values=[""], parse_dates=['date_GMT'])

        # we delete all the teams that were relegated the previous season. The variable teams is defined in the for loop maybe not the best coding
        relegated_teams =get_relegated_teams(teams,ns)
        for attr in new_attrs:
            teams = queues[attr[2]]
            #we delete the relegated teams from each specfic attribute
            for rt in relegated_teams:
                del teams[rt]
            #we define the new columns
            ns.loc[:,'home_'+attr[2]+'_pre_game']=pd.Series(None, index=ns.index)
            ns.loc[:, 'away_'+attr[2]+'_pre_game'] = pd.Series(None, index=ns.index)

        ns,queues= go_through_season(ns,new_attrs,queues)

        #we append the modified ns to dataframe
        #df = pd.concat([df,ns],ignore_index=True)
        all_dfs.append(ns)
    df=pd.concat(all_dfs)
    return df

def go_through_season(ns, new_attrs, queues):
    #the condition status == complete is here because we want to go only through the games that have already been played, but we still need the other games to calucalte length_of_queue
    for index, game in ns[ns['status']=='complete'].iterrows():#TODO: you could add some safety check which would ensure all the games before current date are set to complete
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
                    for i in range(length_of_queue - 1):
                        teams[game[ha + '_team_name']].append(attr[3])
                    #this is a little bit confusing to understand, but it really adds the value of the current game as well to the new queue
                    if type(haattr) is list:
                        lmbd_attr = haattr[1](game[haattr[0][0]], game[haattr[0][1]])
                        teams[game[ha + '_team_name']].append(lmbd_attr)
                    else:
                        teams[game[ha + '_team_name']].append(game[haattr])
                else:
                    ns.loc[index, (ha + '_' + attr[2] + '_pre_game')] = np.mean(teams[game[ha + '_team_name']])
                    if type(haattr) is list:
                        lmbd_attr = haattr[1](game[haattr[0][0]], game[haattr[0][1]])
                        teams[game[ha + '_team_name']].append(lmbd_attr)
                    else:
                        teams[game[ha + '_team_name']].append(game[haattr])
    return ns,queues

def get_relegated_teams(teams,ns):
    ns_teams = ns['home_team_name'].unique()
    return set(teams.keys()) - set(ns_teams)

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


def prob_over25_minute(minute,df):

    h_goal_minutes=df['home_team_goal_timings'].str.split(',', expand=True)
    a_goal_minutes=df['away_team_goal_timings'].str.split(',', expand=True)
    #accounting for NA values
    h_goal_minutes=h_goal_minutes.fillna("0")
    a_goal_minutes=a_goal_minutes.fillna("0")
    #Handling the minutes in stoppage time like 90'4
    for c in h_goal_minutes.columns:
        h_goal_minutes[c] = h_goal_minutes[c].apply(lambda x: convert_added_time(x))
    for c in a_goal_minutes.columns:
        a_goal_minutes[c] = a_goal_minutes[c].apply(lambda x: convert_added_time(x))
    #goal_minutes = df['home_team_goal_timings'].str.split(',', expand=True)
    #goal_minutes['away_team_goal_timings'] = df['away_team_goal_timings'].str.split(',', expand=True)
    #the games that end with more then 25 goals in a boolean array
    over25=df[df['total_goal_count']>2]

    #the total number of games we save in a variable for further comparisons
    n_games=len(df)

    for i in range(1,minute):
        #for x in df['home_team_goal_timings']:
        #    print(x)
        h_i= h_goal_minutes.loc[h_goal_minutes.eq(i).any(1)]
        a_i=a_goal_minutes.loc[a_goal_minutes.eq(i).any(1)]
        #creating a set of indexes we have to delete
        to_delete=set()
        to_delete|=set(h_i.index.values)
        to_delete|=set(a_i.index.values)
        df=df.drop(to_delete)
        h_goal_minutes=h_goal_minutes.drop(to_delete)
        a_goal_minutes=a_goal_minutes.drop(to_delete)

        #mask = [('38' in x) for x in df['home_team_goal_timings']]
        #print(i
        #print(mask)
    prob_ev=len(df)/n_games
    prior_prob=len(over25)/n_games
    bayesian_probability =prior_prob* (sum(df['total_goal_count']>2)/len(over25))/prob_ev
    return [sum(df['total_goal_count']>2)/len(df),bayesian_probability]


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
    attrs.append(('away_team_goal_count', 'home_team_goal_count', 'goals_conceded', 1.5, 38))
    attrs.append(('home_team_possession', 'away_team_possession', 'average_possession', 45, 38))
    attrs.append(('total_goal_count', 'total_goal_count', 'total_goal_count_5games', 2.54, 5))
    attrs.append(('home_team_shots', 'away_team_shots', 'average_shots', 6, 38))
    attrs.append(('away_team_shots', 'home_team_shots', 'average_shots_against', 12, 38))
    lmbd_points = lambda x, y: 3 if x > y else 0 if x < y else 1
    ppg = ([('home_team_goal_count', 'away_team_goal_count'), lmbd_points],
           [('away_team_goal_count', 'home_team_goal_count'), lmbd_points], 'ppg', 1)
    attrs.append(ppg)
    ppg5 = ([('home_team_goal_count', 'away_team_goal_count'), lmbd_points],
            [('away_team_goal_count', 'home_team_goal_count'), lmbd_points], 'ppg_5games', 1, 5)
    attrs.append(ppg5)

if __name__ == '__main__':

    #os.chdir("classifier")
    #add_travel_distance()
    add_to_mysql()
    """"
    leagues = os.listdir()
    dfs=[]
    for l in leagues:
        print(l)
        season_before=files[0]
        os.chdir(l)
        files= os.listdir()
        seasons=files[1:]
        ndf=data_for_classifier(season_before,seasons,game_attributes.attrs)
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
    """
#print(df)
#matches=pd.read_csv("classifier/italy-serie-a-matches-2014-to-2015-stats.csv",na_values=[""])
#juve=matches.loc[(matches['home_team_name']=='Juventus')| (matches['away_team_name']=='Juventus')]
#teams=data_for_classifier(season_before,seasons)
