#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import glob, os  
from datetime import datetime


# In[2]:


import warnings #warnings regarding too many rows were popping up constantly when outputting dataframe values so I put this here for testing purposes
warnings.filterwarnings('ignore')


# In[3]:


pathprem = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\premierleague' 
pathchamp = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\championship' 
pathleagueone = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\leagueone' 
pathleaguetwo = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\leaguetwo' 


prem = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathprem, "*.csv")))) #Assaigning the dataframe for all of the premier league datasets , reading all csv's from filepath using glob to access the filepath. Using concat and map to join all of the opened files through pandas
champ = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathchamp, "*.csv"))))
leagueone = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathleagueone, "*.csv"))))
leaguetwo = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathleaguetwo, "*.csv"))))


# In[4]:


pd.options.display.max_rows = 200 #display setting for using jupyter notebook
prem #prem seasons have 38 matches over 20 teams, our 11 seasons of data give us 4180 matches.


# In[5]:


prem.drop(prem.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
prem


# In[6]:


champ


# In[7]:


champ.drop(champ.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
champ #important to note championship ,league one and two fixtures do not include playoff fixtures. Only league matches.


# In[8]:


leagueone


# In[9]:


leagueone.drop(leagueone.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
leagueone


# In[10]:


leaguetwo


# In[11]:


leaguetwo.drop(leaguetwo.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
leaguetwo


# In[12]:


print(leaguetwo.loc[[0]]) # checking the rows of index 0 to see whether the index needs to be reset after importing data.


# In[13]:


prem.dtypes # checking dtypes to see if the date column is a datetime object


# In[14]:


prem["Date"] = pd.to_datetime(prem["Date"],).dt.strftime('%d-%m-%Y') #converting the date to datetime in a specific format as there's two formats in the date column throughout the years
champ["Date"] = pd.to_datetime(champ["Date"]).dt.strftime('%d-%m-%Y')
leagueone["Date"] = pd.to_datetime(leagueone["Date"]).dt.strftime('%d-%m-%Y')
leaguetwo["Date"] = pd.to_datetime(leaguetwo["Date"]).dt.strftime('%d-%m-%Y')

prem["Date"] = pd.to_datetime(prem["Date"],)#.dt.strftime('%d-%m-%Y') #converting the date to datetime in a specific format as there's two formats in the date column throughout the years
champ["Date"] = pd.to_datetime(champ["Date"])#.dt.strftime('%d-%m-%Y')
leagueone["Date"] = pd.to_datetime(leagueone["Date"])#.dt.strftime('%d-%m-%Y')
leaguetwo["Date"] = pd.to_datetime(leaguetwo["Date"])#.dt.strftime('%d-%m-%Y')


# In[15]:


prem.sort_values(by=['Date'])
champ.sort_values(by=['Date'])
leagueone.sort_values(by=['Date']) # sorting by date to ensure that all matches are aligned
leaguetwo.sort_values(by=['Date'])


# In[16]:


prem.reset_index(drop = True)
champ.reset_index(drop = True)
leagueone.reset_index(drop = True)
leaguetwo.reset_index(drop = True)


# In[17]:


#leaguetwo.isnull().any()
#nan_values = champ[champ.isna().any(axis=1)]

#print (nan_values)
#temp = leaguetwo[leaguetwo['HomeTeam'].str.contains('Bristol')]
#print(temp)


# In[18]:


#nan_values[nan_values.isna().any(1)]


# In[19]:


#print(nan_values.iloc[:,9:15])


# In[20]:


#nan_values.Referee = nan_values.Referee.fillna('No Ref')
#nan_values.HF = nan_values.HF.fillna(0)
#nan_values = nan_values.fillna(0)
#nan_values = nan_values.dropna()
#nan_values[nan_values.isna().any(1)]


# In[21]:


prem = prem.dropna() # Dropping NaN values that occur within the ref columns, rows completely filled and one section of hf and af in league two


# In[22]:


champ.Referee = champ.Referee.fillna('No Ref')
#champ = champ.fillna(0)
champ = champ.dropna()


# In[23]:


leaguetwo.Referee = leaguetwo.Referee.fillna('No Ref')
leaguetwo.HF = leaguetwo.HF.fillna(0)
leaguetwo.AF = leaguetwo.AF.fillna(0)
leaguetwo = leaguetwo.dropna()


# In[24]:


leagueone.Referee = leagueone.Referee.fillna('No Ref')
leagueone = leagueone.dropna()


# In[ ]:





# ---------------------------------------------------------------------------------------------------------------------------
#  
# ---------------------------------------------------------------------------------------------------------------------------
# Maher Method Code Below (Poisson Regression)

# In[ ]:





# In[ ]:





# ---------------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------------------------------------------------------
# Logistic Regression Code Below

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[26]:


x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values # columns from hs to ar of match statistics

y = prem.iloc[:, 6].values # main variable FTR 


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# In[28]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[29]:


y_pred = classifier.predict(X_test)


# In[30]:


print ("Logistic Regression Models Accuracy : ", accuracy_score(y_test, y_pred)) # test run to see if the model will work


# In[31]:


regression_results = []


# In[32]:


def leagueone_linear_regression(z):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = z)
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        leagueone_results = ["LeagueOne",z,(accuracy_score(y_test, y_pred)*100)]
        regression_results.append(leagueone_results)
        print ("leagueone Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
        
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
z = 0
c = 0
for i in range(5):
    leagueone_linear_regression(z)
    z=z+1


# In[ ]:





# In[33]:


def prem_linear_regression(z):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values # match statistics columns
        y = prem.iloc[:, 6].values # FTR column
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = z) #70/30 train test split
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100)]
        regression_results.append(prem_results)
        print ("Premier League Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        

z = 0 # have z in our random state so we can run through the model 5 different times with different random states
for i in range(5):
    prem_linear_regression(z) # calling our logistic regression method for the premier league dataframe
    z=z+1


# In[ ]:





# In[34]:


def leaguetwo_linear_regression(z):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = z)
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100)]
        regression_results.append(leaguetwo_results)
        print ("leaguetwo Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        
z = 0
c = 0
for i in range(5):
    leaguetwo_linear_regression(z)
    z=z+1


# In[35]:


def champ_linear_regression(z):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = z)
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100)]
        regression_results.append(champ_results)
        print ("Championship Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        

z = 0
for i in range(5):
    champ_linear_regression(z)
    z=z+1


# In[67]:


regression_resultsdf = pd.DataFrame(regression_results)
#regression_resultsdf.rename(columns = {'0':'League', '1':'Random State',
#                              '2':'Accuracy Score'}, inplace = True)
regression_resultsdf


# In[68]:


for col in regression_resultsdf.columns:
    print(col)


# In[80]:


fig = px.line(regression_resultsdf, x=1, y=2, color = 0,text=0,title='Accuracy Scores For All Leagues',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division"
                 },)
fig.update_traces(textposition='bottom center')
fig.show()


# ---------------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------------------------------------------------------
# Statistical Analysis Code Here:

# In[37]:


prem


# In[38]:


prem['year'] = prem['Date'].apply(lambda x: x.strftime('%Y'))
prem['month'] = prem['Date'].apply(lambda x: x.strftime('%m')) #creating the year and month columns in order to use them for the season column later

prem['year'] = prem['year'].astype('int')
prem['month'] = prem['month'].astype('int')

champ['year'] = champ['Date'].apply(lambda x: x.strftime('%Y'))
champ['month'] = champ['Date'].apply(lambda x: x.strftime('%m'))

champ['year'] = champ['year'].astype('int')
champ['month'] = champ['month'].astype('int')

leagueone['year'] = leagueone['Date'].apply(lambda x: x.strftime('%Y'))
leagueone['month'] = leagueone['Date'].apply(lambda x: x.strftime('%m'))

leagueone['year'] = leagueone['year'].astype('int')
leagueone['month'] = leagueone['month'].astype('int')

leaguetwo['year'] = leaguetwo['Date'].apply(lambda x: x.strftime('%Y'))
leaguetwo['month'] = leaguetwo['Date'].apply(lambda x: x.strftime('%m'))

leaguetwo['year'] = leaguetwo['year'].astype('int')
leaguetwo['month'] = leaguetwo['month'].astype('int')


# In[39]:


prem.dtypes #checking whether date is a datetime object


# In[40]:


prem['season'] = np.where(prem['month'] <= 6,(prem['year'] - 1).astype(str) + '/' + prem['year'].astype(str), 
                          prem['year'].astype(str) + '/' + (prem['year'] + 1).astype(str))
# here i am checking the month and year, if they are less than 6/july it means the matchday is in the year columns season and the season before
champ['season'] = np.where(champ['month'] <= 6,(champ['year'] - 1).astype(str) + '/' + champ['year'].astype(str), 
                          champ['year'].astype(str) + '/' + (champ['year'] + 1).astype(str))
#otherwise the season is attributed to the years current year value and the next year as its season
leagueone['season'] = np.where(leagueone['month'] <= 6,(leagueone['year'] - 1).astype(str) + '/' + leagueone['year'].astype(str), 
                          leagueone['year'].astype(str) + '/' + (leagueone['year'] + 1).astype(str))


leaguetwo['season'] = np.where(leaguetwo['month'] <= 6,(leaguetwo['year'] - 1).astype(str) + '/' + leaguetwo['year'].astype(str), 
                          leaguetwo['year'].astype(str) + '/' + (leaguetwo['year'] + 1).astype(str))


# In[ ]:





# In[41]:


prem


# In[42]:


import plotly.express as px


# In[43]:


#Referees


# In[44]:


prem['total_fouls'] = prem['HF'] + prem['AF']
prem['total_yellows'] = prem['HY'] + prem['AY']
prem['total_reds'] = prem['HR'] + prem['AR']


# In[45]:


prem


# In[46]:


prem11 = prem[prem['season'] == "2011/2012"]
prem11
refs = prem11.groupby(['Referee']).sum()
refs['matches'] =  prem11['Referee'].value_counts()
refs.reset_index(inplace = True)
refs


# In[47]:


fig = px.scatter(refs, x="matches", y="total_fouls", color = 'Referee',text="Referee",title='Referee Fouls Given in 2011/12')
fig.update_traces(textposition='top center')
fig.show()


# In[48]:


fig = px.scatter(refs, x="matches", y="total_yellows", color = 'Referee',text="Referee",title='Referee Yellow Cards Given in 2011/12')
fig.update_traces(textposition='top center')
fig.show()


# In[49]:


fig = px.scatter(refs, x="matches", y="total_reds", color = 'Referee',text="Referee",title='Referee Red Cards Given in 2011/12')
fig.update_traces(textposition='bottom right')
fig.show()


# In[50]:


prem11 = prem[prem['season'] == "2011/2012"]
prem11
prem11['homewin'] = np.where(prem11['FTHG'] > prem11['FTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['homelosses'] = np.where(prem11['FTHG'] < prem11['FTAG'], 1, 0)#1 equals a Away win, 0 is a loss
prem11['home_draw'] = np.where(prem11['FTHG'] == prem11['FTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['home_matches'] = prem11['homewin'] + prem11['homelosses'] + prem11['home_draw']

prem11['halftime_homewin'] = np.where(prem11['HTHG'] > prem11['HTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['halftime_homelosses'] = np.where(prem11['HTHG'] < prem11['HTAG'], 1, 0)#1 equals a Away win, 0 is a loss
prem11['halftime_draw'] = np.where(prem11['HTHG'] == prem11['HTAG'], 1, 0)#1 equals a win, 0 is a loss


hometeam = prem11.groupby(['HomeTeam']).sum()
hometeam.reset_index(inplace = True)
hometeam['winratio'] = hometeam['homewin'] / hometeam['home_matches'] * 100
hometeam['lossratio'] = hometeam['homelosses'] / hometeam['home_matches'] * 100
hometeam['halftime_winratio'] = hometeam['halftime_homewin'] / hometeam['home_matches'] * 100
hometeam['halftime_DRAWratio'] = hometeam['halftime_draw'] / hometeam['home_matches'] * 100
hometeam['halftime_lossratio'] = hometeam['halftime_homelosses'] / hometeam['home_matches'] * 100

hometeam['Shots_on_target%'] = hometeam['HST'] / hometeam['HS'] * 100
hometeam['goal_conversion'] = hometeam['FTHG'] / hometeam['HST'] * 100

hometeam


# In[51]:


prem11['awaywin'] = np.where(prem11['FTHG'] < prem11['FTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['awaylosses'] = np.where(prem11['FTHG'] > prem11['FTAG'], 1, 0)#1 equals a Away win, 0 is a loss
prem11['away_draw'] = np.where(prem11['FTHG'] == prem11['FTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['away_matches'] = prem11['awaywin'] + prem11['awaylosses'] + prem11['away_draw']

prem11['halftime_awaywin'] = np.where(prem11['HTHG'] < prem11['HTAG'], 1, 0)#1 equals a win, 0 is a loss
prem11['halftime_awaylosses'] = np.where(prem11['HTHG'] > prem11['HTAG'], 1, 0)#1 equals a Away win, 0 is a loss
prem11['halftime_away_draw'] = np.where(prem11['HTHG'] == prem11['HTAG'], 1, 0)#1 equals a win, 0 is a loss



awayteam = prem11.groupby(['AwayTeam']).sum()
awayteam.reset_index(inplace = True)
awayteam['winratio'] = awayteam['awaywin'] / awayteam['away_matches'] * 100
awayteam['lossratio'] = awayteam['awaylosses'] / awayteam['away_matches'] * 100
awayteam['halftime_winratio'] = awayteam['halftime_awaywin'] / awayteam['away_matches'] * 100
awayteam['halftime_DRAWratio'] = awayteam['halftime_away_draw'] / awayteam['away_matches'] * 100
awayteam['halftime_lossratio'] = awayteam['halftime_awaylosses'] / awayteam['away_matches'] * 100

awayteam['Shots_on_target%'] = awayteam['AST'] / awayteam['AS'] * 100
awayteam['goal_conversion'] = awayteam['FTAG'] / awayteam['AST'] * 100

awayteam


# In[52]:


fig = px.scatter(hometeam, x="winratio", y="halftime_winratio", color = 'HomeTeam',title='Fulltime over Halftime Win Ratios 11/12')
fig.show()


# In[53]:


fig = px.bar(hometeam, x="HomeTeam", y="homewin", title='Premier League Teams Wins At Home 11/12')
fig.show()


# In[54]:


fig = px.bar(awayteam, x="AwayTeam", y="awaywin", title='Premier League Teams Wins Away 11/12')
fig.show()


# In[55]:


fig = px.scatter(hometeam, x="homewin", y="halftime_homelosses", color = 'HomeTeam',text = 'HomeTeam',size ='halftime_homelosses',color_discrete_sequence =px.colors.qualitative.Dark24,
                 title='Home Wins when losing at halftime 11/12')
fig.update_traces(textposition='top center')
fig.show()


# In[56]:


fig = px.scatter(hometeam, x="homewin", y="halftime_homewin", color = 'HomeTeam',text = 'HomeTeam',size ='halftime_homewin',color_discrete_sequence =px.colors.qualitative.Dark24,
                 title='Home Wins when winning at halftime 11/12')
fig.update_traces(textposition='top center')
fig.show()


# In[ ]:




