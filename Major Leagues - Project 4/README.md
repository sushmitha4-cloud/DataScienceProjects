# What's the score - Project 4 - Regression

The project is to evaluate the provided data soccer matches data set and predict the score for a team given the other features.
# Input data and Output Information

This project takes raw data related to soccer, 
loads the data into data frames and then performs regression on the same
using Linear Regression, Neural networks and Random Forest Regression
to predict the score of a team. 

# Source of Raw/Processed Data

Raw data has been taken from : https://github.com/fivethirtyeight/data/tree/master/soccer-spi

# Data Cleaning

There are a lot of values that have NaN values, which really do not add value, so dropping those.
Also, team is one hot encoded.

# Data Exploration, Processing and Feature Engineering

We cleaned and processed the data to select required columns and 
then explored the data to assume that "team1" is a "Home" team and "team2"
is "Away" team.
Based on this we performed a group by on team1 and team2 to find the mean
scores of each team, in played both as Home team and an Away team.
 
A quick report about the same is available under reports. 
 
# Understanding the Dataset for Regression
My Understanding of the dataset and how the predictions can work are as follows:

Dataset has lot of columns, I'm assuming that all the columns have some impact on the match between the two teams and hence keeping them in.
The primary columns that give a logical sense in calculating the score of for a team are,
1. The season in which the match is happening
2. The opponent teamID against which the target team is playing.
3. The league_id, different teams can excel in different leagues
4. The score of the opponent team
Date Column, team Names (teamIDs will be used), League Name (league ID will be used), don't really add value, so dropping that colum
 
The Data set is also used to predict both team scores in a match at once
using other features.

I used Random Forest Regressor for the same 
and in order to calculate the accuracy
I was getting errors when I used accuracy score method, so
I calculated by comparing both the two dimensional arrays from predictions and actuals.
Whenever matched, I incremented the success counter by 1 and finally divided the success counter 
by the length of the predictions set.
 
# Data Regression

I used Regression algorithms like,
1.Linear Regression
2.Neural Network
3. Random Forest Regression

to predict the scores of the team.

# Plotting and exploration results

Prediction accuracies against the Regression algorithm chosen is 
plotted and saved under reports.

# Conclusions and further improvements:
Better accuracy for Neural networks could have been achieved if we knew and understood all the features in the data set.
Future work can be understanding the features that actually impact the analysis and remove the ones that have less impact, to increase the accuracy.
