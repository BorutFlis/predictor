'''
Created on 28. jun. 2018

@author: Borut
'''
import numpy as np
import pandas as pd  
import sys 
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from scipy.stats import pearsonr

os.chdir('top10scorers')
player_std=dict()
player_fga_result=dict()
player_previous_game=dict()
full_df = pd.DataFrame()
full_data=[]
positions=dict()

positions['guards']=['Russell Westbrook','Victor Oladipo','Kyrie Irving','James Harden','Damian Lillard']
positions['forwards']=['Anthony Davis','Giannis Antetokounmpo','Kevin Durant','LaMarcus Aldridge','Lebron James']
guards=pd.DataFrame()
forwards=pd.DataFrame()
for player in os.listdir():
    name_of_player = player.rsplit( ".", 1 )[ 0 ]
    data = pd.read_csv(player, sep=',', na_values=['Did Not Dress','Did Not Play','Inactive','Not With Team'], index_col=0)
    full_data.append(data)
    #player_std[name_of_player]=data['PTS'].describe()['std']
    data['Unnamed: 7'] = data['Unnamed: 7'].str.extract(r"\((.*?)\)").apply(pd.to_numeric)
    data['previous_PTS']=pd.Series(index=data.index)
    data.loc[2:,'previous_PTS']=[pt for pt in data['PTS'][:-1]]
    percentiles=np.array([2.5,25,50,75,97.5])
    ptiles_vers=np.nanpercentile(data['PTS'],percentiles)
    player_previous_game[name_of_player]=data.corr()['PTS']['previous_PTS']
    player_fga_result[name_of_player]= data.corr()['FGA']['+/-']
    if name_of_player in positions['guards']:
        guards=guards.append(data)
    else:
        forwards=forwards.append(data)
        
#print(data['PTS'])
#wc=pd.read_csv("World Cup 2018 Dtaset.csv")
sorted_d = sorted(player_fga_result.items(), key=lambda x: x[1])
for player in sorted_d:
    print(player[0], player[1])
print("__________________________________")    
sorted_p = sorted(player_previous_game.items(), key=lambda x: x[1])
for player in sorted_p:
    print(player[0], player[1])

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    #drops all the NA values and all the non numeric data
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

#guards_p=calculate_pvalues(guards)
#forwards_p=calculate_pvalues(forwards)
mask = ~np.isnan(guards['3PA'].values) * ~np.isnan(guards['DRB'].values)
#the product of the negated values of isnan, becuase if one value is Na the other
# is useless to, but we won't find such a case
guards_pear_3pa=guards['3PA'].values[mask]
guards_pear_drb=guards['DRB'].values[mask]

mask = ~np.isnan(forwards['3PA'].values) * ~np.isnan(forwards['DRB'].values)
forwards_pear_3pa=forwards['3PA'].values[mask]#._get_numeric_data()
forwards_pear_drb=forwards['DRB'].values[mask]

print(pearsonr(guards_pear_3pa,guards_pear_drb),pearsonr(forwards_pear_3pa,forwards_pear_drb))
print(guards.corr()['3PA']['DRB'],forwards.corr()['3PA']['DRB'])

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_correlations(df, n=20):
    au_corr = df.corr()
    labels_to_drop = get_redundant_pairs(au_corr)
    au_corr= au_corr.unstack()
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n], au_corr[-n:]

full_df=pd.concat(full_data)

full_df= full_df.drop(["GS"], axis=1)
#plt.matshow(full_df)
p_values=calculate_pvalues(full_df)
top_corr,bottom_corr=get_top_correlations(full_df)
for label,corr in top_corr.items():
    print(label,corr,p_values[label[0]][label[1]])

for label,corr in bottom_corr.items():
    print(label,corr,p_values[label[0]][label[1]])
#print(data.groupby(['Year']).filter(lambda x: x['Age'] > 35))
#print(data.loc[lambda x : x.Age>35].groupby(['Year'])['Player'].count())
