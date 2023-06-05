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


# In[2]:


import warnings
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


print(leaguetwo.loc[[0]])


# In[13]:


prem.dtypes


# In[14]:


prem["Date"] = pd.to_datetime(prem["Date"]).dt.strftime('%d-%m-%Y') #converting the date to datetime in a specific format as there's two formats in the date column throughout the years
champ["Date"] = pd.to_datetime(champ["Date"]).dt.strftime('%d-%m-%Y')
leagueone["Date"] = pd.to_datetime(leagueone["Date"]).dt.strftime('%d-%m-%Y')
leaguetwo["Date"] = pd.to_datetime(leaguetwo["Date"]).dt.strftime('%d-%m-%Y')


# In[15]:


champ


# In[16]:


prem.reset_index(drop = True)
champ.reset_index(drop = True)
leagueone.reset_index(drop = True)
leaguetwo.reset_index(drop = True)


# In[17]:


#leaguetwo.isnull().any()
#nan_values = prem[prem.isna().any(axis=1)]

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


prem = prem.dropna()


# In[22]:


champ.Referee = champ.Referee.fillna('No Ref')
champ = champ.fillna(0)
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




