#!/usr/bin/env python
# coding: utf-8

# # What's the Score - Project 4 - Data Science
# 
# ### Project Description
# #### The project is to evaluate the provided data soccer matches data set and predict the score for a team given the other features.
# 
# ### Dataset source
# #### https://github.com/fivethirtyeight/data/tree/master/soccer-spi

# In[1]:


# import required libraries
from pandas import DataFrame, read_csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# ### Read the data into dataframes for further manipulation

# In[2]:


matchesRaw = pd.read_csv('/Major Leagues - Project 4/data/raw/spi_matches.csv')


# In[3]:


matchesRaw.head(5)


# ### Data Cleaning
# 
# #### There are a lot of values that have NaN values, which really do not add value, so dropping those.
# #### Also, teams can be one hot encoded.

# In[4]:


matches= matchesRaw.dropna()


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
matches['team1ID']=encoder.fit_transform(matches['team1'].astype('str'))
matches['team2ID']=encoder.fit_transform(matches['team2'].astype('str'))


# In[6]:


matches.head(5)


# ### Data Exploration

# #### Data processing for some data exploration.

# In[7]:


processedData = matchesRaw[['team1','team2','score1','score2','league_id','league','season']]
processedData = processedData.dropna()


# #### Assuming that team1 is the home team and team 2 is the away team 

# In[8]:


TeamScoresHome = processedData.groupby('team1')['score1'].mean().to_frame().reset_index()
TeamScoresAway = processedData.groupby('team2')['score2'].mean().to_frame().reset_index()


# In[9]:


CompareTeamScoresHomeAndAway = TeamScoresHome[['team1','score1']]
CompareTeamScoresHomeAndAway['score2'] = TeamScoresAway['score2']


# #### Identifying, A team scores more at home or away

# In[10]:


CompareTeamScoresHomeAndAway = CompareTeamScoresHomeAndAway.rename(columns={"team1":  'Team' ,"score1" : 'Mean Home Scores' ,'score2' : "Mean Away Scores"})
CompareTeamScoresHomeAndAway


# ### Understanding the Dataset for Regression
# #### My Understanding of the dataset and how the predictions can work are as follows
# #### Dataset has a lot of columns, I'm assuming that all the columns have some impact on the match between the two teams and hence keeping them in. 
# 
# #### The primary columns that give a logical sense in calculating the score of for a team are, 
# 
# #### 1. The season in which the match is happening
# #### 2. The opponent teamID against which the target team is playing.
# #### 3. The league_id, different teams can excel in different leagues
# #### 4. The score of the opponent team
# 
# #### Date Column, team Names (teamIDs will be used), League Name (league ID will be used), don't really add value, so dropping that column.

# In[11]:


#matches.loc[(matches['league_id']==1843) & (matches['date']=='2016-08-12')]


# In[12]:


MatchesForAnalysis = matches.drop(['date', 'team1','team2','league'], axis = 1)


# ### Preparing data for regression analysis
# #### 1. Get the score column as the prediction column.
# #### 2. Get the matches data without the score1 column.
# #### 3. Prepare the train and test data accordingly. 

# In[13]:


Score = MatchesForAnalysis[['score1']]
MatchesForAnalysisWithoutScore = MatchesForAnalysis.drop(['score1'], axis = 1)

# Split Total Data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_validation, Y_train, Y_validation = train_test_split(MatchesForAnalysisWithoutScore, Score, test_size=0.20)


# ### Linear Regression

# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

linearReg = LinearRegression()
linearReg.fit(X_train,Y_train)

predictionsOfTheValidationSet = linearReg.predict(X_validation)
predictionsOfTheTrainSet = linearReg.predict(X_train)
print(predictionsOfTheValidationSet)


# ### Processing the results 
# #### The results need to be modiied a bit to round them to nearest integer, as the soccer score cannot be in decimals.
# ### Accuracy calculation of the predicted score against the actual score.

# In[15]:


predictionsOfTheValidationSet = np.rint(predictionsOfTheValidationSet); # Rounding to nearest integer 
predictionsOfTheValidationSet[predictionsOfTheValidationSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheValidationSet[predictionsOfTheValidationSet == -0] = 0;

predictionsOfTheTrainSet = np.rint(predictionsOfTheTrainSet); # Rounding to nearest integer 
predictionsOfTheTrainSet[predictionsOfTheTrainSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheTrainSet[predictionsOfTheTrainSet == -0] = 0;

print("Validation Data Accuracy : "+ str(accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)))
print("Train Data Accuracy : " + str(accuracy_score(Y_train['score1'].tolist(), predictionsOfTheTrainSet)))


# In[16]:


### Plotting the results
#### Plotting to compare multiple Regressors


# In[17]:


PlottingData = pd.DataFrame(columns=['Regressor','Accuracy'])
temp = {'Regressor': "Linear Regressor", 'Accuracy': accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)}
PlottingData = PlottingData.append(temp, ignore_index=True)


# ###### Similar analysis can be performed for score2 column

# ### Neural Network Regressor

# In[18]:


from sklearn.neural_network import MLPRegressor
nnReg = MLPRegressor().fit(X_train, Y_train)

predictionsOfTheValidationSet = nnReg.predict(X_validation)
predictionsOfTheTrainSet = nnReg.predict(X_train)

print(predictionsOfTheValidationSet)


# ### Processing the results 
# #### The results need to be modiied a bit to round them to nearest integer, as the soccer score cannot be in decimals.
# ### Accuracy calculation of the predicted score against the actual score.

# In[19]:


predictionsOfTheValidationSet = np.rint(predictionsOfTheValidationSet); # Rounding to nearest integer 
predictionsOfTheValidationSet[predictionsOfTheValidationSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheValidationSet[predictionsOfTheValidationSet == -0] = 0;

predictionsOfTheTrainSet = np.rint(predictionsOfTheTrainSet); # Rounding to nearest integer 
predictionsOfTheTrainSet[predictionsOfTheTrainSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheTrainSet[predictionsOfTheTrainSet == -0] = 0;

print("Validation Data Accuracy : "+ str(accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)))
print("Train Data Accuracy : " + str(accuracy_score(Y_train['score1'].tolist(), predictionsOfTheTrainSet)))


# In[20]:


temp = {'Regressor': "Neural network Regressor", 'Accuracy': accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)}
PlottingData = PlottingData.append(temp, ignore_index=True)


# ### Random Forest Regressor

# In[21]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

randomReg = RandomForestRegressor()
randomReg.fit(X_train, Y_train)
predictionsOfTheValidationSet = randomReg.predict(X_validation)
predictionsOfTheTrainSet = randomReg.predict(X_train)

print(predictionsOfTheValidationSet)


# ### Processing the results 
# #### The results need to be modiied a bit to round them to nearest integer, as the soccer score cannot be in decimals.
# ### Accuracy calculation of the predicted score against the actual score.

# In[22]:


predictionsOfTheValidationSet = np.rint(predictionsOfTheValidationSet); # Rounding to nearest integer 
predictionsOfTheValidationSet[predictionsOfTheValidationSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheValidationSet[predictionsOfTheValidationSet == -0] = 0;

predictionsOfTheTrainSet = np.rint(predictionsOfTheTrainSet); # Rounding to nearest integer 
predictionsOfTheTrainSet[predictionsOfTheTrainSet < 0] = 0;   # If rounding set the value to less than 0, set it to 0.
predictionsOfTheTrainSet[predictionsOfTheTrainSet == -0] = 0;

print("Validation Data Accuracy : "+ str(accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)))
print("Train Data Accuracy : " + str(accuracy_score(Y_train['score1'].tolist(), predictionsOfTheTrainSet)))


# In[23]:


temp = {'Regressor': "Random Forest Regressor", 'Accuracy': accuracy_score(Y_validation['score1'].tolist(), predictionsOfTheValidationSet)}
PlottingData = PlottingData.append(temp, ignore_index=True)
PlottingData


# In[24]:


settings = PlottingData.plot(kind='bar')
settings.set(ylabel='Accuracy of score prediction')
plt.title("Accuracy vs Regressor used")
plt.xticks(np.arange(3) + .1, ('Linear Regressor', 'Neural Network Regressor', 'Random Forest Regressor'))


# ### Lets now predict scores for both teams in a match.

# ### Preparing data for regression analysis
# #### 1. Get the score1 and score2 columns as the prediction columns.
# #### 2. Get the matches data without the score1 and score 2 columns.
# #### 3. Prepare the train and test data accordingly. 

# In[25]:


Score = MatchesForAnalysis[['score1','score2']]
MatchesForAnalysisWithoutScore = MatchesForAnalysis.drop(['score1','score2'], axis = 1)

# Split Total Data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_validation, Y_train, Y_validation = train_test_split(MatchesForAnalysisWithoutScore, Score, test_size=0.20)


# ### Using the Random Forest Regressor for predicting both the scores (as it was the best from the 3 regressors used above)

# In[26]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

randomReg = RandomForestRegressor()
randomReg.fit(X_train, Y_train)
predictionsOfTheValidationSet = randomReg.predict(X_validation)
predictionsOfTheTrainSet = randomReg.predict(X_train)


# ### Processing the predicitions
# #### The results need to be modiied a bit to round them to nearest integer, as the soccer score cannot be in decimals.
# 
# ### Accuracy calculation in Multidimension predictions scenario
# 
# #### I calculated by comparing both the two dimensional arrays (predictions and actuals).
# #### Whenever matched, I incremented the success counter by 1 and finally divided the success counter 
# #### by the length of the predictions set.

# In[27]:


# Process the predictions
for values in predictionsOfTheValidationSet: 
    values[0]=np.rint(values[0])
    values[1]=np.rint(values[1])
    if (values[0] < 0): values[0] = 0
    if (values[1] < 0): values[1] = 0    

# Accuracy score calculation        
successful = 0
index = 0
actuals=Y_validation.values
for predictions in predictionsOfTheValidationSet: 
    if(predictions[0]==actuals[index][0] and predictions[1]==actuals[index][1]): successful = successful + 1
    index = index + 1            

print("Test Data Accuracy : " + str(successful/len(predictionsOfTheValidationSet)))                 


# ### Conclusions and further improvements:
# ### Better accuracy for Neural networks could have been achieved if we knew and understood all the features in the data set.
# ### Future work can be understanding the features that actually impact the analysis and remove the ones that have less impact, to increase the accuracy.

# In[ ]:




