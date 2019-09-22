'''
Created on 28. jul. 2018

@author: Borut
'''
import os
import sys
import glob
import pandas as pd
import datetime
import football
import collections
import numpy as np
import Orange
import timeit



def expected_value(odds,prob):

    return (odds-1)*prob-(1-prob)


def over_under_value_bet(files):
    df = pd.read_csv(files,na_values=[""],parse_dates=['date_GMT'])
    value_bets=[]
    for index,match in df.iterrows():
        if match['Odds_Over25']!=0:
            ev=expected_value(match['Odds_Over25'],match['Over25 Average']/100)
    
            if(ev>0.1 and match['Over25 Average']<100):
                value_bets.append([match['Home Team'], match['Away Team'],match['Odds_Over25'],match['Over25 Average']/100,ev])
                
    value_bets.sort(key=lambda x: x[4])
     
    for vb in value_bets:
        print(vb)

def analysis_of_predictions(w_pm):
    #TODO make function so you can use it all the time
    condition1 = (w_pm['home_team_goal_count'] - w_pm['away_team_goal_count'] > 0) & (w_pm['predicted_result'] == 1)
    condition2 = (w_pm['home_team_goal_count'] - w_pm['away_team_goal_count'] < 0) & (w_pm['predicted_result'] == 2)
    condition0 = (w_pm['home_team_goal_count'] - w_pm['away_team_goal_count'] == 0) & (w_pm['predicted_result'] == 0)
    correct=sum(condition1)+sum(condition2)+sum(condition0)
    condition25_1=(w_pm['total_goal_count'] > 2) & (w_pm['predicted_over'] == 1)
    condition25_2=(w_pm['total_goal_count'] <= 2) & (w_pm['predicted_over'] == 2)
    correct25=sum(condition25_1)+sum(condition25_2)
    predicted=len(w_pm)
    #denominators for precision
    pr_1=sum(w_pm['predicted_result'] == 1)
    pr_2=sum(w_pm['predicted_result'] == 2)
    pr_0=sum(w_pm['predicted_result'] == 0)
    pr25_1=sum(w_pm['predicted_over'] == 1)
    pr25_2=sum(w_pm['predicted_over'] == 2)
    return (correct, predicted),(correct25,predicted),[(sum(condition1),pr_1),(sum(condition2),pr_2),(sum(condition0),pr_0),(sum(condition25_1),pr25_1),(sum(condition25_2),pr25_2)]

def weekend_prediction(display_attrs,today= datetime.datetime.today(),n_days=4):
    # we define today's date as the default the games in this round are going to be in three days after this date: friday, Saturday, Sunday, Monday
    #we initialize two lists to hold the dataframes of all the leagues all_dfs(all Games) dfs(games playing this weekend)
    all_dfs=[]
    dfs=[]
    os.chdir("weekend")
    files = glob.glob("*stats.csv")

    for f in files:
        ndf=pd.read_csv(f,na_values=[""],parse_dates=['date_GMT'])
        all_dfs.append(ndf)
        ndf=ndf[(ndf['date_GMT']>=today) & (ndf['date_GMT']<(today+ datetime.timedelta(days=n_days)))]
        dfs.append(ndf)
    df = pd.concat(dfs, ignore_index=True)
    all_df=pd.concat(all_dfs, ignore_index=True)
    #we check if the df containing the games we will predict has 34 games(serie A 10, EPL 10, Bundesliga 9, Prvaliga 5)
    if len(df)!=34:
        print("there is an unusual number of games:",len(df))
        #ask if we wish to continue we can check in this time what was so unusual
        p_result = input("do you wish to continue y/n:")
        if p_result =='n':
            exit()
    # TODO we load the queues from pickle
    # if queues are available in pickle

    #else get queues
    queues=dict()
    #we have to define dictionary keys before because we have multiple leagues
    for attr in display_attrs:
        queues[attr[2]]=dict()
    os.chdir(os.path.dirname(sys.argv[0])+"\\weekend\\previous_seasons")
    previous_seasons = glob.glob("*stats.csv")
    for season in previous_seasons:
        #we have to deal with all the attributes, they come in tuples (home attr, away attr, name attr, lenght of queue)
        for attr in display_attrs:
            #we have multiple leagues so we have to merge the queues in one dictionary
            queue_length=attr[4] if len(attr)>4 else None
            league_queue=football.make_attribute_queue(attr,season,queue_length)
    #relegated_teams=football.get_relegated_teams(teams,eue([attr[0],attr[1]],season,queue_length)
            queues[attr[2]].update(league_queue) #I am not sure I even need to take care of relegated teams
    for attr in display_attrs:
        # we define the new columns
        all_df.loc[:, 'home_' + attr[2] + '_pre_game'] = pd.Series(None, index=all_df.index)
        all_df.loc[:, 'away_' + attr[2] + '_pre_game'] = pd.Series(None, index=all_df.index)
    football.go_through_season(all_df,display_attrs,queues)
    """"
    #we go through all the games that all ready happened this season to fill up the queues
    for index, game in all_df[all_df['status']=="complete"].iterrows():
        #we go through all the attributes we want to update
        for attr in display_attrs:
            teams = queues[attr[2]]
            # we go through an additional for loop to account for the home and away team
            for ha, haattr in [('home', attr[0]), ('away', attr[1])]:
                if game[ha + '_team_name'] not in teams:
                    #we calculate the queue length, we have to do this because different leagues have a different number of games
                    queue_length = sum(all_df['away_team_name']==game[ha + '_team_name']) + sum(all_df['home_team_name']==game[ha + '_team_name'])
                    teams[game[ha + '_team_name']] = collections.deque(maxlen=queue_length)
                    #for the promoted teams we do not have data so we add the average predefined in attr[3]
                    for i in range(queue_length-1):
                        teams[game[ha + '_team_name']].append(attr[3])
                    teams[game[ha + '_team_name']].append(game[haattr])
                else:
                    teams[game[ha + '_team_name']].append(game[haattr])
    """
    os.chdir('..')
    if False:#TODO: I need to code in such a way that it does not execute if the file weekly predictions does not exists, even better make your own function so you can check accuracy anytime you want
        #we open the file that has the predictions of the previous week
        w_p=pd.read_csv("weekly_predictions.csv",na_values=[""],parse_dates=['date_GMT'])
        #we go through the predictions of the previous week where we fill the queues
        #for i,game in w_p.iterrows():
            #difference=game['home_total_goal_count']-game['away_total_goal_count']
        w_p=w_p[['home_team_name', 'away_team_name','predicted_result','predicted_over']].copy()
        #merging previous week predictions with all the games
        w_pm = w_p.merge(all_df, on=['home_team_name', 'away_team_name'])
        #this code is only used for the slovenian league because it has 4 games between teams, in second half of season I will have to change to keep="last"
        w_pm = w_pm.drop_duplicates(subset=['home_team_name', 'away_team_name'],keep='first')
        accuracy,accuracy_over25,precisions= analysis_of_predictions(w_pm)
        #TODO: this calculations can throw division by zero error I need to handle this
        print("Our prediction accuracy: ", accuracy[0],accuracy[1],accuracy[0]/accuracy[1])
        print("Our over 2.5 accuracy: ", accuracy_over25[0],accuracy_over25[1],accuracy_over25[0]/accuracy_over25[1])
        print("predicted 1 precision",precisions[0][0],precisions[0][1],precisions[0][0]/precisions[0][1])
        print("predicted 2 precision", precisions[1][0], precisions[1][1], precisions[1][0] / precisions[1][1])
        print("predicted 0 precision", precisions[2][0], precisions[2][1], precisions[2][0] / precisions[2][1])
        print("predicted over 25 precision", precisions[3][0], precisions[3][1], precisions[3][0] / precisions[3][1])
        print("predicted under 25 precision", precisions[4][0], precisions[4][1], precisions[4][0] / precisions[4][1])
        #we do not need the dataframe that has all the games anymore
        del all_df

        # we open the file with all predictions to which we will add last weeks' predictions
        try:
            all_p = pd.read_csv("all_predictions.csv", na_values=[""], parse_dates=['date_GMT'])
        #if there is no file with this name we try to catch the exception and create a new dataframe.
        except FileNotFoundError:
            all_p=pd.DataFrame()

        all_p = all_p.append(w_pm)
        all_p.to_csv('all_predictions.csv')

    predicted_result=[]
    predicted_over=[]
    #we reset the index because our df is made from multiple dataFrames
    df=df.reset_index(drop=True)
    #we load the data on which we will train the classifiers         \
    classifier_data = Orange.data.Table(os.path.dirname(sys.argv[0])+"/test/classifier.tab")
    #we create and train the two best models
    learner_nb = Orange.classification.NaiveBayesLearner()
    learner_rf= Orange.classification.RandomForestLearner()
    classifier_nb = learner_nb(classifier_data)
    classifier_rf = learner_rf(classifier_data)

    for i,game in df.iterrows():
        print(game['date_GMT'])
        print(game['home_team_name'],game['away_team_name'])

        print("{0:45s} {1:8.2f}".format("odds over 25:", game['odds_ft_over25']))
        print("{0:45s} {1:8.2f} {2:8.2f} {3:8.2f}".format("odds 1 0 2",game['odds_ft_home_team_win'],game['odds_ft_draw'],game['odds_ft_away_team_win']))
        #dictionary I will use to create Orange data instance
        game_attrs={}
        for attr in display_attrs:
            teams = queues[attr[2]]
            game_attrs['home_'+attr[2]+'_pre_game']=np.mean(teams[game['home_team_name']])
            game_attrs['away_'+attr[2]+'_pre_game']=np.mean(teams[game['away_team_name']])
            print("{0:45s} {1:8.2f} {2:8.2f}".format(attr[2],game_attrs['home_'+attr[2]+'_pre_game'],game_attrs['away_'+attr[2]+'_pre_game']))
        # predictions of Orange classifiers
        #we define a new Orange instance which will be used to fill with data from the game we want to predict
        new_inst = Orange.data.Instance(classifier_data.domain)
        #we enumerate attributes from the domain so we can assign the values from the game_attrs dictionary
        for i,attr in enumerate(classifier_data.domain.attributes):
            new_inst.x[i]=game_attrs[attr.name]
        # we print the predictions of Orange classifier which we previously determined as the best ones.

        classifier_nb(new_inst)
        #TODO: Change so that it will be 1,0,2
        print("{0:45s} {1:8d} ".format("Naive Bayes predicted class: ", classifier_nb(new_inst)[0]))
        print("{0:45s} {1:8.2f} {2:8.2f} {3:8.2f}".format("Naive Bayes predictions (0,1,2): ",*classifier_nb(new_inst,1)[0]))
        print("{0:45s} {1:8.0f} ".format("Random Forest predicted class: ", classifier_rf(new_inst)[0]))
        print("{0:45s} {1:8.2f} {2:8.2f} {3:8.2f}".format("Random Forest predictions (0,1,2): ",*classifier_rf(new_inst,1)[0]))
        p_result = input('Who will win: ')
        predicted_result.append(p_result)
        p_over = input('1- Over 2.5 / 2- Under 2.5: ')
        predicted_over.append(p_over)

    df['predicted_result'] = pd.Series(predicted_result,index=df.index)
    df['predicted_over']= pd.Series(predicted_over,index=df.index)
    df.to_csv('weekly_predictions.csv')
    return df


def daily_odds():
    #TODO: use of python api
    os.chdir("matchday")
    file = os.listdir()[0]
    df = pd.read_csv(file, na_values=[""], parse_dates=['date_GMT'])
    for i,game in df.iterrows():
        if game['Odds_Draw']>(8.097 -2.121*game['Odds_Over25']) and game['Odds_Over25']>2:
            print(game['Home Team'],game['Away Team'],game['Odds_Draw'],game['Odds_Over25'])
#files = glob.glob("*.csv")
#files.sort(key=os.path.getmtime)
#  matchday_file=files[-1]

if __name__ == '__main__':
    df=weekend_prediction(football.game_attributes.attrs,today=datetime.datetime.today(), n_days=6)
    #daily_odds()

#df=pd.read_csv("classifier/italy-serie-a-matches-2017-to-2018-stats.csv",na_values=[""],parse_dates=['date_GMT'])
#df1=pd.read_csv("classifier/italy-serie-a-matches-2016-to-2017-stats.csv",na_values=[""],parse_dates=['date_GMT'])
