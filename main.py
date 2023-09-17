#!/usr/bin/env python
# coding: utf-8

# In[47]:


#All Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics

import glob, os  
from datetime import datetime
from IPython.display import display
import plotly.express as px
import lime
import lime.lime_tabular


# In[48]:


import warnings #warnings regarding too many rows were popping up constantly when outputting dataframe values so I put this here for testing purposes
warnings.filterwarnings('ignore')


# In[49]:


#Filepaths for the Datasets
pathprem = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\premierleague' 
pathchamp = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\championship' 
pathleagueone = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\leagueone' 
pathleaguetwo = r'C:\Users\shaqu\OneDrive\Desktop\Masters Course\Dissertation\Project\Dissertation\Data\leaguetwo' 

#Assaigning the dataframe for all of the premier league datasets , reading all csv's from filepath using glob to access the filepath. Using concat and map to join all of the opened files through pandas
prem = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathprem, "*.csv")))) 
champ = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathchamp, "*.csv"))))
leagueone = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathleagueone, "*.csv"))))
leaguetwo = pd.concat(map(pd.read_csv, glob.glob(os.path.join(pathleaguetwo, "*.csv"))))


# In[50]:


pd.options.display.max_rows = 200 #display setting for using jupyter notebook
prem #prem seasons have 38 matches over 20 teams, our 11 seasons of data give us 4180 matches.


# In[51]:


prem.drop(prem.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
prem


# In[52]:


champ


# In[53]:


champ.drop(champ.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
champ #important to note championship ,league one and two fixtures do not include playoff fixtures. Only league matches.


# In[54]:


leagueone


# In[55]:


leagueone.drop(leagueone.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
leagueone


# In[56]:


leaguetwo


# In[57]:


leaguetwo.drop(leaguetwo.iloc[:, 23:140], inplace=True, axis=1) #dropping the accompined betting data that we wont be using.
leaguetwo


# In[58]:


print(leaguetwo.loc[[0]]) # checking the rows of index 0 to see whether the index needs to be reset after importing data.


# In[59]:


prem.dtypes # checking dtypes to see if the date column is a datetime object


# In[60]:


#converting the date to datetime in a specific format as there's two formats in the date column throughout the years
prem["Date"] = pd.to_datetime(prem["Date"],).dt.strftime('%d-%m-%Y') 
champ["Date"] = pd.to_datetime(champ["Date"]).dt.strftime('%d-%m-%Y')
leagueone["Date"] = pd.to_datetime(leagueone["Date"]).dt.strftime('%d-%m-%Y')
leaguetwo["Date"] = pd.to_datetime(leaguetwo["Date"]).dt.strftime('%d-%m-%Y')

#converting the date to datetime in a specific format as there's two formats in the date column throughout the years
prem["Date"] = pd.to_datetime(prem["Date"],)#.dt.strftime('%d-%m-%Y') 
champ["Date"] = pd.to_datetime(champ["Date"])#.dt.strftime('%d-%m-%Y')
leagueone["Date"] = pd.to_datetime(leagueone["Date"])#.dt.strftime('%d-%m-%Y')
leaguetwo["Date"] = pd.to_datetime(leaguetwo["Date"])#.dt.strftime('%d-%m-%Y')


# In[61]:


# sorting by date to ensure that all matches are aligned
prem.sort_values(by=['Date'])
champ.sort_values(by=['Date'])
leagueone.sort_values(by=['Date']) 
leaguetwo.sort_values(by=['Date'])


# In[62]:


#Resetting index here abd dropping the old one indexes
prem.reset_index(drop = True)
champ.reset_index(drop = True)
leagueone.reset_index(drop = True)
leaguetwo.reset_index(drop = True)


# In[63]:


#Depreciated code checking for NaN values and locating specific rows where they were present

#leaguetwo.isnull().any()
#nan_values = champ[champ.isna().any(axis=1)]

#print (nan_values)
#temp = leaguetwo[leaguetwo['HomeTeam'].str.contains('Bristol')]
#print(temp)


# In[64]:


#nan_values[nan_values.isna().any(1)]


# In[65]:


#print(nan_values.iloc[:,9:15])


# In[66]:


#nan_values.Referee = nan_values.Referee.fillna('No Ref')
#nan_values.HF = nan_values.HF.fillna(0)
#nan_values = nan_values.fillna(0)
#nan_values = nan_values.dropna()
#nan_values[nan_values.isna().any(1)]


# In[67]:


# Dropping NaN values that occur within the ref columns, rows completely filled and one section of hf and af in league two
prem = prem.dropna() 


# In[68]:


champ.Referee = champ.Referee.fillna('No Ref')
#champ = champ.fillna(0)
champ = champ.dropna()


# In[69]:


leaguetwo.Referee = leaguetwo.Referee.fillna('No Ref')
leaguetwo.HF = leaguetwo.HF.fillna(0)
leaguetwo.AF = leaguetwo.AF.fillna(0)
leaguetwo = leaguetwo.dropna()


# In[70]:


leagueone.Referee = leagueone.Referee.fillna('No Ref')
leagueone = leagueone.dropna()


# In[ ]:





# 

# In[71]:


#creating the year and month columns in order to use them for the season column later

prem['year'] = prem['Date'].apply(lambda x: x.strftime('%Y'))
prem['month'] = prem['Date'].apply(lambda x: x.strftime('%m')) 

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


# In[72]:


# here i am checking the month and year, if they are less than 6/july it means the matchday is in the year columns season and the season before
#otherwise the season is attributed to the years current year value and the next year as its season.

prem['season'] = np.where(prem['month'] <= 6,(prem['year'] - 1).astype(str) + '/' + prem['year'].astype(str), 
                          prem['year'].astype(str) + '/' + (prem['year'] + 1).astype(str))

champ['season'] = np.where(champ['month'] <= 6,(champ['year'] - 1).astype(str) + '/' + champ['year'].astype(str), 
                          champ['year'].astype(str) + '/' + (champ['year'] + 1).astype(str))

leagueone['season'] = np.where(leagueone['month'] <= 6,(leagueone['year'] - 1).astype(str) + '/' + leagueone['year'].astype(str), 
                          leagueone['year'].astype(str) + '/' + (leagueone['year'] + 1).astype(str))


leaguetwo['season'] = np.where(leaguetwo['month'] <= 6,(leaguetwo['year'] - 1).astype(str) + '/' + leaguetwo['year'].astype(str), 
                          leaguetwo['year'].astype(str) + '/' + (leaguetwo['year'] + 1).astype(str))


# In[73]:


#Depreciated Code: Dividing seasons for initial use in trying to have only the last season as a testing variable
prem_train = prem[prem['season'] != "2021/2022"]
prem_train


# In[74]:


prem_test = prem[prem['season'] == "2021/2022"]
prem_test


# ---------------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------------------------------------------------------
# Log-Linear Classifier (sklearn classifier) Code Below

# In[75]:


#Initial Depreciated code to test the model and ensure all individual sections of the model works


# In[76]:


#x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values # columns from hs to ar of match statistics

#y = prem.iloc[:, 6].values # main variable FTR 


# In[77]:


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# In[78]:


#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)


# In[79]:


#y_pred = classifier.predict(X_test)


# In[80]:


#print ("Logistic Regression Models Accuracy : ", accuracy_score(y_test, y_pred)) # test run to see if the model will work


# In[81]:


#Storing the Accuracy results in an empty series that I can append the results and league identifier alongside other variables
logistic_results = []


# In[82]:


#Initially coded to work inside as a call function
#Now this function is Depreciated as the test size split is defined in the overall function call
def assaign_traintest_split(test_size_split,counter):
    #test_size_split = 0.30
    if counter >= 5 < 10:
        test_size_split = 0.20
        #print("IF:", test_size_split)
        return(test_size_split)
    else:
        #print("ELSE:", test_size_split)
        return(test_size_split)
    


# In[83]:


#Function Call for the Log-Linear Classification Classifier(Otherwise known as Logistic Regression)
#Function takes in a interger called 'z' which is used as an increasing counter for the random state over multiple iterations in a for loop
#The function also takes in the test_size_split which is a float defined as either 0.30 or 0.20 to split our train test split as either 70/30 or 80/20 respectively
def leagueone_logistic(z,test_size_split):        
        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values #Training Variables
        y = leagueone.iloc[:, 6].values #Target variable 'FTR'
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        #Our predications come out as either 'H','A','D'
        
        #Here I designate the results that we are appending to our series , accuracy score is *100 to show full percentage and make it easier to graph
        leagueone_results = ["LeagueOne",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        logistic_results.append(leagueone_results)
        
        #print ("leagueone Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
        #print(y_pred)


# In[84]:


#this same process for the functions is repeated for all of the 3 remaining leagues.


# In[85]:


def prem_logistic(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values # match statistics columns
        y = prem.iloc[:, 6].values # FTR column
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z) #70/30 train test split
        
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        logistic_results.append(prem_results)
        
        #print ("Premier League Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        


# In[ ]:





# In[86]:


def leaguetwo_logistic(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        logistic_results.append(leaguetwo_results)
        
        return(test_size_split)
        
        #print ("leaguetwo Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       


# In[87]:


def champ_logistic(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = LogisticRegression(random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        logistic_results.append(champ_results)
        
        #print ("Championship Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        


# In[170]:


#This is the main function for the models
#Here I define the test size split the will be passed into the model functions along with the z counter
#For testing purposes the for loops are at a range of 5 to ensure the functions work properly but can be increased to higher ranges with no issue
def logistic_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_logistic(z,test_size_split)
        champ_logistic(z,test_size_split)
        leagueone_logistic(z,test_size_split)
        leaguetwo_logistic(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_logistic(z,test_size_split)
        champ_logistic(z,test_size_split)
        leagueone_logistic(z,test_size_split)
        leaguetwo_logistic(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

#This function below is to show the results from the series that we have appended to in all of our model calls 
#First we convert the series into a dataframe so that we can rename and graph our results, also a dataframe is easier to use for the future.
def show_logistic_results():
    logistic_resultsdf = pd.DataFrame(logistic_results)
    logistic_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    
    #I use display here instead of print or calling the dataframe as neither worked in testing.
    display(logistic_resultsdf)
    
    #I use plotly here to construct a scatter plot showing our results
    #The plot shows the random state as a way to compare accuracies score over the different leagues and the train test split is shown as two different sets of symbols for the datapoints
    fig = px.scatter(logistic_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using Log-Linear Classifier',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[89]:


#Here I call both functions to complete and show the log-linear classifier results
logistic_function_calls()


# In[171]:


show_logistic_results()


# In[199]:


logistic_resultsdf = pd.DataFrame(logistic_results)
logistic_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[200]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = logistic_resultsdf.loc[logistic_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = logistic_resultsdf.loc[logistic_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[201]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[210]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(Log-Linear)',
            text_auto=True )
fig.show()


# In[91]:


#This function here is to demonstrate the LIME explanations, the functions operates as the previous function calls
#However we define the test size split and z counter inside the function instead.
#We also now take in the lime entry to target specific rows to explain in the lime call
def champ_logistic1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = LogisticRegressionCV(cv = 2 , random_state = z)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        #We dont append the champ results here as this is not part of the main model calls, instead we print to show initial accuraccy and explainations for lime in jupyter notebook
        print(champ_results)
        
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','D','A'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)
        #logistic_results.append(champ_results)


# In[92]:


lime_entry = 10
champ_logistic1(lime_entry)


# In[93]:


lime_entry = 15
champ_logistic1(lime_entry)


# CROSS PARAMETER GRID SEARCH TESTING

# In[94]:


C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

scores = []

#Much like the Lime function this function was used as a testing call for grid search 
#The function works upon its call and the specific accuracy scores are appended in the scores series

def champ_linear_regression(z):    
    
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = z)
        
        classifier = LogisticRegression(random_state = z)
        
        for choice in C:
            classifier.set_params(C=choice)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            scores.append(classifier.score(X_train, y_train))
            champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100)]
            #regression_results.append(champ_results)
        
        
        print ("Championship Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        

#z = 0
#for i in range(5):
    #champ_linear_regression(z)
    #z=z+1
    
#print(scores)


# In[ ]:





# KNN CODE

# In[305]:


knn_results = []


# In[ ]:





# In[306]:


#Like the previous Log-linear functions, I have coded the KNN functions to operate on a similar base code structure
#The key difference is now our KNeighborsClaissifier that takes in the n_neigbors arg for how many nearest neighbors the model should look for in classifying
def champ_KNN(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = KNeighborsClassifier(n_neighbors=7)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        knn_results.append(champ_results)
        #print ("Championship Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       


# In[307]:


def prem_KNN(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = prem.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = KNeighborsClassifier(n_neighbors=7)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        knn_results.append(prem_results)


# In[308]:


def leagueone_KNN(z,test_size_split):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = KNeighborsClassifier(n_neighbors=7)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leagueone_results = ["League One",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        knn_results.append(leagueone_results)


# In[309]:


def leaguetwo_KNN(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = KNeighborsClassifier(n_neighbors=7)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        knn_results.append(leaguetwo_results)


# In[310]:


#Once again here I created the base strucure code for the main function calls
#And then changed the specific functions that we are calling to suit the KNN functions and variables
def knn_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_KNN(z,test_size_split)
        champ_KNN(z,test_size_split)
        leagueone_KNN(z,test_size_split)
        leaguetwo_KNN(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_KNN(z,test_size_split)
        champ_KNN(z,test_size_split)
        leagueone_KNN(z,test_size_split)
        leaguetwo_KNN(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

def show_knn_results():
    knn_resultsdf = pd.DataFrame(knn_results)
    knn_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    display(knn_resultsdf)
    
    fig = px.scatter(knn_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using KNN',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[311]:


knn_function_calls()


# In[312]:


show_knn_results()


# In[313]:


knn_resultsdf = pd.DataFrame(knn_results)
knn_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[314]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = knn_resultsdf.loc[knn_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = knn_resultsdf.loc[knn_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[315]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[316]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(KNN, N=7)',
            text_auto=True )
fig.show()


# In[ ]:





# In[103]:


#Lime explainer function operates on the same basis as the log-linear function but now for KNN
def champ_knn1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        
        
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        print(champ_results)
        
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','D','A'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)


# In[104]:


lime_entry = 10
champ_knn1(lime_entry)


# In[105]:


lime_entry = 15
champ_knn1(lime_entry)


# RANDOM FOREST CODE

# In[276]:


forest_results = []


# In[277]:


from sklearn.ensemble import RandomForestClassifier


# In[278]:


#Like the previous functions above , the base structure is the same.
#Key difference is now the random forest classifier which operates off of the number of estimators/ trees to operate in the forest
def champ_RF(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        forest_results.append(champ_results)
        #print ("Championship Models Accuracy : ", accuracy_score(y_test, y_pred), "Random State was :" , z ,"\n")
# here ive adapated the earlier model and made a function so that we can call it in a loop without issue to output the results of the model with different random states       
        


# In[279]:


def prem_RF(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = prem.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        forest_results.append(prem_results)


# In[280]:


def leagueone_RF(z,test_size_split):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leagueone_results = ["League One",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        forest_results.append(leagueone_results)


# In[281]:


def leaguetwo_RF(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        forest_results.append(leaguetwo_results)


# In[282]:


def forest_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_RF(z,test_size_split)
        champ_RF(z,test_size_split)
        leagueone_RF(z,test_size_split)
        leaguetwo_RF(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_RF(z,test_size_split)
        champ_RF(z,test_size_split)
        leagueone_RF(z,test_size_split)
        leaguetwo_RF(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

def show_forest_results():
    forest_resultsdf = pd.DataFrame(forest_results)
    forest_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    display(forest_resultsdf)
    
    fig = px.scatter(forest_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using Random Forest',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[283]:


forest_function_calls()


# In[284]:


show_forest_results()


# In[285]:


forest_resultsdf = pd.DataFrame(forest_results)
forest_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[286]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = forest_resultsdf.loc[forest_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = forest_resultsdf.loc[forest_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[287]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[288]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(Random Forest Estimators = 100)',
            text_auto=True )
fig.show()


# In[ ]:





# In[115]:


def champ_RF1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        print(champ_results)
        print(y_pred[0])
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','D','A'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)


# In[116]:


lime_entry = 0
champ_RF1(lime_entry)


# Naive Bayes CODE

# In[117]:


naivebayes_results = []


# In[118]:


from sklearn.naive_bayes import GaussianNB


# In[119]:


#Much like the functions created above for the other classifiers, the base structure of the code is the same
#The classifier is now the GaussianNB which operates under the gaussian naive assumption of bayes theorem to classify our models results
def champ_NB(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        naivebayes_results.append(champ_results)


# In[120]:


def prem_NB(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = prem.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        naivebayes_results.append(prem_results)


# In[121]:


def leagueone_NB(z,test_size_split):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leagueone_results = ["League One",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        naivebayes_results.append(leagueone_results)


# In[122]:


def leaguetwo_NB(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        naivebayes_results.append(leaguetwo_results)


# In[322]:


def nb_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_NB(z,test_size_split)
        champ_NB(z,test_size_split)
        leagueone_NB(z,test_size_split)
        leaguetwo_NB(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_NB(z,test_size_split)
        champ_NB(z,test_size_split)
        leagueone_NB(z,test_size_split)
        leaguetwo_NB(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

def show_nb_results():
    naivebayes_resultsdf = pd.DataFrame(naivebayes_results)
    naivebayes_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    display(naivebayes_resultsdf)
    
    fig = px.scatter(naivebayes_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using Naive Bayes(GaussianNB)',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[124]:


nb_function_calls()


# In[323]:


show_nb_results()


# In[325]:


naivebayes_resultsdf = pd.DataFrame(naivebayes_results)
naivebayes_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[326]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = naivebayes_resultsdf.loc[naivebayes_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = naivebayes_resultsdf.loc[naivebayes_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[327]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[328]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(GaussianNB)',
            text_auto=True )
fig.show()


# In[ ]:





# In[126]:


def champ_NB1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        print(champ_results)
        print(y_pred[0])
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','D','A'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)


# In[127]:


lime_entry = 0
champ_NB1(lime_entry)


# SVM CODE HERE

# In[128]:


svm_results = []


# In[129]:


from sklearn import svm


# In[130]:


#Akin to the functions that i created before the base coding structure is the same and operates the same way
#The key difference in these sets of functions is now we use the support vector machines SVC classifier .
#Due to the way that the svc classifers operate on kernal operations these sets of functions take a longer time than our other functions to complete
def champ_SVM(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        svm_results.append(champ_results)
        
        


# In[131]:


def prem_SVM(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = prem.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        svm_results.append(prem_results)


# In[132]:


def leagueone_SVM(z,test_size_split):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leagueone_results = ["League One",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        svm_results.append(leagueone_results)


# In[133]:


def leaguetwo_SVM(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = svm.SVC()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        svm_results.append(leaguetwo_results)


# In[134]:


def svm_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_SVM(z,test_size_split)
        champ_SVM(z,test_size_split)
        leagueone_SVM(z,test_size_split)
        leaguetwo_SVM(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_SVM(z,test_size_split)
        champ_SVM(z,test_size_split)
        leagueone_SVM(z,test_size_split)
        leaguetwo_SVM(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

def show_svm_results():
    svm_resultsdf = pd.DataFrame(svm_results)
    svm_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    display(svm_resultsdf)
    
    fig = px.scatter(svm_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using SVM',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[135]:


svm_function_calls()


# In[136]:


show_svm_results()


# In[317]:


svm_resultsdf = pd.DataFrame(svm_results)
svm_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[318]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = svm_resultsdf.loc[svm_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = svm_resultsdf.loc[svm_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[319]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[321]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(SVM.SVC)',
            text_auto=True )
fig.show()


# In[ ]:





# In[137]:


def champ_SVM1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = svm.SVC(probability=True)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        print(champ_results)
        print(y_pred[0])
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','D','A'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)


# In[138]:


lime_entry = 0
champ_SVM1(lime_entry)


# DECISION TREE CODE HERE

# In[233]:


dt_results = []


# In[234]:


#Like all of our other previous functions these decision tree functions operate on the same basic structure
#Now with the decision tree classifier using a max depth to define how far the tree should base its classifications on its cascading decisions
def champ_DT(z,test_size_split):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = champ.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = DecisionTreeClassifier(max_depth= 3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        dt_results.append(champ_results)


# In[235]:


def prem_DT(z,test_size_split):        
        x = prem.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = prem.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = DecisionTreeClassifier(max_depth= 3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        prem_results = ["Premier League",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        dt_results.append(prem_results)


# In[236]:


def leagueone_DT(z,test_size_split):        
        x = leagueone.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leagueone.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = DecisionTreeClassifier(max_depth= 3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leagueone_results = ["League One",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        dt_results.append(leagueone_results)


# In[237]:


def leaguetwo_DT(z,test_size_split):        
        x = leaguetwo.iloc[:, [11, 12,13,14,15,16,17,18]].values
        y = leaguetwo.iloc[:, 6].values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = DecisionTreeClassifier(max_depth= 3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        leaguetwo_results = ["League Two",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        dt_results.append(leaguetwo_results)


# In[238]:


def dt_function_calls():
    test_size_split = 0.30
    z = 0
    for i in range(5):
        prem_DT(z,test_size_split)
        champ_DT(z,test_size_split)
        leagueone_DT(z,test_size_split)
        leaguetwo_DT(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1
        
    test_size_split = 0.20
    z = 0
    for i in range(5):
        prem_DT(z,test_size_split)
        champ_DT(z,test_size_split)
        leagueone_DT(z,test_size_split)
        leaguetwo_DT(z,test_size_split)
        
        #assaign_traintest_split(test_size_split,z)
        #print(test_size_split)
        z=z+1

def show_dt_results():
    dt_resultsdf = pd.DataFrame(dt_results)
    dt_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)
    display(dt_resultsdf)
    
    fig = px.scatter(dt_resultsdf, x='Random State', y='Accuracy Score', color = 'League',title='Accuracy Scores For All Leagues Using Decision Tree',
                labels={
                     "1": "Random State",
                     "2": "Accuracy Score %",
                     "0": "Football Division",
                     "3": "TrainTestSplit",
                 },symbol="TrainTestSplit",
                 symbol_map={0.3: "circle", 0.2: "x"})
    fig.update_traces(textposition='bottom center')
    fig.show()


# In[239]:


dt_function_calls()


# In[240]:


show_dt_results()


# In[241]:


dt_resultsdf = pd.DataFrame(dt_results)
dt_resultsdf.rename(columns = {0:'League', 1:'Random State',
                                  2:'Accuracy Score',3:'TrainTestSplit'}, inplace = True)


# In[242]:


#splitscoresdf = pd.DataFrame(columns = ['70/30' , '80/20'])
seventy_thirty_split = dt_resultsdf.loc[dt_resultsdf['TrainTestSplit'] == 0.3, 'Accuracy Score'].mean()
eighty_twenty_split = dt_resultsdf.loc[dt_resultsdf['TrainTestSplit'] == 0.2, 'Accuracy Score'].mean()


# In[243]:


print(seventy_thirty_split)
print(eighty_twenty_split)
splitscores = [seventy_thirty_split,eighty_twenty_split]
traintestsplitgraph = ['70 / 30','80 / 20']
splitscoresdf = pd.DataFrame(splitscores)
splitscoresdf.rename(columns = {0:'Accuracy'}, inplace = True)
splitscoresdf.insert(0,'TrainTestSplit', traintestsplitgraph )
splitscoresdf


# In[244]:


fig = px.bar(splitscoresdf, x ='TrainTestSplit' ,y='Accuracy',color = 'TrainTestSplit',title='Mean Accuracy Scores For Train Test Split(Decision Tree, Depth 3)',
            text_auto=True )
fig.show()


# In[147]:


def champ_DT1(lime_entry):        
        x = champ.iloc[:, [11, 12,13,14,15,16,17,18]]#.values
        y = champ.iloc[:, 6]#.values
        
        z = 0
        test_size_split = 0.30
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_split, random_state = z)
        
        classifier = DecisionTreeClassifier(max_depth= 5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        champ_results = ["Championship",z,(accuracy_score(y_test, y_pred)*100),test_size_split]
        print(champ_results)
        print(y_pred[0])
        
        
        #plt.figure(figsize=(12,12))
        #tree.plot_tree(classifier, fontsize=10)
        #plt.savefig('tree_high_dpi', dpi=200)
        
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data= np.array(X_train), 
                                                           feature_names= X_train.columns,
                                                  class_names=['H','A','D'], mode='classification')
        
        
        exp = explainer.explain_instance(data_row= X_test.iloc[lime_entry],predict_fn= classifier.predict_proba)
        exp.show_in_notebook(show_table=True)


# In[148]:


lime_entry = 0
champ_DT1(lime_entry)


# In[149]:


lime_entry = 15
champ_DT1(lime_entry)


# In[ ]:





# ---------------------------------------------------------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------------------------------------------------------
# Statistical Analysis Code Here:

# In[150]:


prem


# In[ ]:





# In[151]:


prem.dtypes #checking whether date is a datetime object


# In[ ]:





# In[ ]:





# In[152]:


prem


# In[ ]:





# In[153]:


#Referees


# In[154]:


prem['total_fouls'] = prem['HF'] + prem['AF']
prem['total_yellows'] = prem['HY'] + prem['AY']
prem['total_reds'] = prem['HR'] + prem['AR']


# In[155]:


prem


# In[156]:


prem11 = prem[prem['season'] == "2011/2012"]
prem11
refs = prem11.groupby(['Referee']).sum()
refs['matches'] =  prem11['Referee'].value_counts()
refs.reset_index(inplace = True)
refs


# In[157]:


fig = px.scatter(refs, x="matches", y="total_fouls", color = 'Referee',text="Referee",title='Referee Fouls Given in 2011/12')
fig.update_traces(textposition='top center')
fig.show()


# In[158]:


fig = px.scatter(refs, x="matches", y="total_yellows", color = 'Referee',text="Referee",title='Referee Yellow Cards Given in 2011/12')
fig.update_traces(textposition='top center')
fig.show()


# In[159]:


fig = px.scatter(refs, x="matches", y="total_reds", color = 'Referee',text="Referee",title='Referee Red Cards Given in 2011/12')
fig.update_traces(textposition='bottom right')
fig.show()


# In[160]:


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


# In[161]:


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


# In[162]:


fig = px.scatter(hometeam, x="winratio", y="halftime_winratio", color = 'HomeTeam',title='Fulltime over Halftime Win Ratios 11/12')
fig.show()


# In[163]:


fig = px.bar(hometeam, x="HomeTeam", y="homewin", title='Premier League Teams Wins At Home 11/12')
fig.show()


# In[164]:


fig = px.bar(awayteam, x="AwayTeam", y="awaywin", title='Premier League Teams Wins Away 11/12')
fig.show()


# In[165]:


fig = px.scatter(hometeam, x="homewin", y="halftime_homelosses", color = 'HomeTeam',text = 'HomeTeam',size ='halftime_homelosses',color_discrete_sequence =px.colors.qualitative.Dark24,
                 title='Home Wins when losing at halftime 11/12')
fig.update_traces(textposition='top center')
fig.show()


# In[166]:


fig = px.scatter(hometeam, x="homewin", y="halftime_homewin", color = 'HomeTeam',text = 'HomeTeam',size ='halftime_homewin',color_discrete_sequence =px.colors.qualitative.Dark24,
                 title='Home Wins when winning at halftime 11/12')
fig.update_traces(textposition='top center')
fig.show()


# ---------------------------------------------------------------------------------------------------------------------------

# In[167]:


prem


# In[168]:


# Poisson Regression Code , Abandoned for classification.
#Code left in to show work done.


def assaign_matches(league):
    poiss_home = league[['HomeTeam','FTHG','FTAG']]
    poiss_away =  league[['AwayTeam','FTHG','FTAG']]

    poiss_home = poiss_home.rename(columns = {'HomeTeam' : 'Team', 'FTHG' : 'Goals Scored', 'FTAG' : 'Goals Conceded'})
    poiss_away = poiss_away.rename(columns = {'AwayTeam' : 'Team', 'FTHG' : 'Goals Conceded', 'FTAG' : 'Goals Scored'})

    #poiss_team_power = pd.concat([poiss_prem_home,poiss_prem_away], ignore_index= True).groupby('Team').mean()
    #poiss_team_power

    #league average number of matches / compared to average goals scored by teams only present in 1-2 seasons, possible restructure



    league_home = poiss_home.groupby(['Team']).sum()
    league_home.reset_index(inplace = True)
    
    
    league_home_goals_scored = poiss_home['Goals Scored'].sum() # amount of total home goals scored in the league


    matchesplayedhome = poiss_home['Team'].value_counts()
    matchesplayedhome.sort_index()


    matcheshome = matchesplayedhome.to_frame()
    matcheshome.reset_index(inplace= True)
    matcheshome.rename(columns = {'index' : 'Team', 'Team' : 'Matches Played Home'},inplace= True)
    matcheshome.sort_values('Team', inplace= True)
    matcheshome.reset_index(inplace= True, drop = True)
    #matches
    #prem_home['Matches Played'] = matches['Matches Played']
    #print (pd.merge(prem_home, matches, on='Team'))
    league_home = pd.merge(league_home, matcheshome, on='Team')
    league_home
    
    #away matches
    league_away = poiss_away.groupby(['Team']).sum()
    league_away.reset_index(inplace = True)
    
    
    league_away_goals_scored = poiss_away['Goals Scored'].sum() # amount of total home goals scored in the league


    matchesplayedaway = poiss_away['Team'].value_counts()
    matchesplayedaway.sort_index()


    matchesaway = matchesplayedaway.to_frame()
    matchesaway.reset_index(inplace= True)
    matchesaway.rename(columns = {'index' : 'Team', 'Team' : 'Matches Played Away'},inplace= True)
    matchesaway.sort_values('Team', inplace= True)
    matchesaway.reset_index(inplace= True, drop = True)
    #matches
    #prem_home['Matches Played'] = matches['Matches Played']
    #print (pd.merge(prem_home, matches, on='Team'))
    league_away = pd.merge(league_away, matchesaway, on='Team')
    league_away
    
    league_matches = pd.merge(league_home, league_away, on='Team')
    league_matches = league_matches.rename(columns = {'Goals Scored_x' : 'Goals_Scored_Home', 'Goals Conceded_x' : 'Goals_Conceded_Home', 'Goals Conceded_y' : 'Goals_Conceded_Away','Goals Scored_y' : 'Goals_Scored_Away'})
    return(league_matches)


#prem_power = assaign_matches(prem)
#prem_power


# In[169]:


# Poisson Regression Code , Abandoned for classification.
#Code left in to show work done.
def assaign_team_strength(league):
    league_home_goals = league['Goals_Scored_Home'].sum()
    league_away_goals = league['Goals_Scored_Away'].sum()
    league_total_goals = league_home_goals + league_away_goals

    #prem_power['Avg_Home_Goals'] = [prem_power['Goals_Scored_Home'] / prem_power['Matches Played Home'] * 100 for x in prem_power['Goals_Scored_Home']]
    league['Avg_Home_Goals'] = league['Goals_Scored_Home'] / league['Matches Played Home']
    league['Avg_Away_Goals'] = league['Goals_Scored_Away'] / league['Matches Played Away']

    league['Avg_Concede_Home_Goals'] = league['Goals_Conceded_Home'] / league['Matches Played Home']
    league['Avg_Concede_Away_Goals'] = league['Goals_Conceded_Away'] / league['Matches Played Away']

    avg_num_of_home_goals = league['Avg_Home_Goals'].mean()
    avg_num_of_conceded_home_goals = league['Avg_Concede_Home_Goals'].mean()
    #print(avg_num_of_home_goals)

    avg_num_of_away_goals = league['Avg_Away_Goals'].mean()
    avg_num_of_conceded_away_goals = league['Avg_Concede_Away_Goals'].mean()
    #print(avg_num_of_away_goals)

    league['Home_Attack_Strength'] = league['Avg_Home_Goals'] / avg_num_of_home_goals 
    league['Away_Attack_Strength'] = league['Avg_Away_Goals'] / avg_num_of_away_goals 

    league['Home_Defence_Strength'] = league['Avg_Concede_Home_Goals'] / avg_num_of_conceded_home_goals
    league['Away_Defence_Strength'] = league['Avg_Concede_Away_Goals'] / avg_num_of_conceded_away_goals

    
    
    return(league)
#assaign_team_strength(prem_power)
#prem_power


# 

# In[ ]:





# In[ ]:




