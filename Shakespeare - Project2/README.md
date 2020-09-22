# Shakespeare Plays - Project 2 - To be, or not to be

DataScience project that identifies the Player's name based on other features of the data set.

# Input data and Output Information

This project takes raw input data from the Shakespeare plays 
The Information it provides is about the accuracy for different classifiers like
Gaussian NB, KNN and Decision tree.

# Source of Raw Data

Shakespeare Plays data "https://www.kaggle.com/kingburrito666/shakespeare-plays"
Raw data is not available in project folder structure under the raw data, because of the size of the files.

# Raw Data Processing

Dropping rows that have NAN values and truncating all data sets to least possible size among all the datasets to keep the datasets of same size.

# Data exploration

Step 1: Exploring the number of players per play
Step 2: Exploring number of lines per play and player

# Plotting and exploration results

All the results are plotted and available in reports folder.

# Feature Engineering

1. Wanted to use the tag cloud or the word cloud for classification
 -Operations performed on feature 'PlayerLine'
 a. eliminate common words
 b. eliminate punctuations
 c. convert all words to lower case
Idea was to identify the frequency per word per player and use that information to train the ML model.
(Incomplete -- need more knowledge to explore it.)

2. Wanted to use the ActSceneLine column to classify the data.
 -Operations performed on feature 'ActSceneLine'
 a. split by '.'
 b. add columns that are split.
 c. drop other columns
 
# Splitting Data in test and train sets

Split the dataset into 0.2 size test content and 0.8 size training content.
Use features like Act, Scene and Line to identify class Player
Used LabelEncoder for encoding text like class to integer.

# Classification

Classified Using 
1. Gaussian Naive Baeysian
2. KNNeighbours
3. Decision tree classifier.