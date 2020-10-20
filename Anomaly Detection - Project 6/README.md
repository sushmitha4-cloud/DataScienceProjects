# Anomaly Detection - Project 6

#Project Description
The project tries to identify anomalies in the speeds observed 
by the sensors over real time traffic in Minnesota

# Input data and Output Information

This project takes raw data related to speeds observed 
by the sensors over real time traffic in Minnesota,  
loads the data into data frames and then performs some anomaly detection 
on it to identify whether the sensors are on expressways or normal streets.

# Source of Raw/Processed Data

Raw data has been taken from : ### https://www.kaggle.com/boltzmannbrain/nab

# Data Processing and Feature Engineering

Minimal or No Data processing as there are just 2 columns and no null values.

Bucket Hour and Day from timestamp to different columns, Hour and Day respectively.

# Data Exploration and reports
1. Basic data exploration by plotting speeds over time.
    Observation : The graph clearly tells us that there are drops in speeds from 15th to 18th of September 2015
2. Identifying the mean speeds by days.
    Observation : This graph proves that our assumption of speeds going down after 14th is true

# Anomaly Detection and Conclusions

I used Anomaly detection algorithms, OneClassSVM and Isolation Forest
By above analysis, We clearly see that, anomalies are identified at lower speeds.
For Isolation forest mean speeds for which anomalies are identified are 38 and 
for One class SvM it is 54 Both the algorithms identified around 65 to be normal speeds, 
Which can conclude that, 
the sensors are placed on expressways, where the minimum speed is around 60.