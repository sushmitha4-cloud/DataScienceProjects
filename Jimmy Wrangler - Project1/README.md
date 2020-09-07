# Jimmy Wrangler - Project 1 - Data Explorer

My first DataScience project that compares the Air Quality between the cities of Taiwan and Madrid in year 2014 and 2015.

# Input data and Output Information

This project takes raw Air Quality data from four csv files from Taiwan and Madrid in years 2014 and 2015, 
The Information it provides is a comparison of Air Quality between the two cities and the comparison of Air Quality in same city in different years.

# Source of Raw Data

Madrid pollution data : https://www.kaggle.com/decide-soluciones/air-quality-madrid

Taiwan pollution data : https://www.kaggle.com/phlinhng/air-quality

Please note that : We used only 2014 and 2015 datasets, as the data for those years was our scope bound to.

# Raw Data Processing

Step 1: Selecting only required columns from raw data and renaming columns to keep the dataframes in sync. 
        Columns used: 'SO2','CO','O3','NO','NO2','PM10','PM25'
Step 2: Dropping rows that have NAN values and truncating all data sets to least possible size among all the datasets to keep the datasets of same size.
Step 3: Truncating all dataframes to same size to get accurate results.
Step 4: Adding required columns to the dataframes.

# Data Concatenation and exploration

Step 1: Concatenate all the data frames into 1 dataframe.
Step 2: Comparing air qualities in Taiwan in 2015 against 2014 using City filter as Taiwan and grouping by Year and processing the means.
Step 3: Comparing air qualities in Madrid in 2015 against 2014 using City filter as Madrid and grouping by Year and processing the means.
Step 4: Comparing air qualities in Taiwan and Madrid in year 2014 using Year filter 2014 and grouping by City and processing the means
Step 5: Comparing air qualities in Taiwan and Madrid in year 2015 using Year filter 2015 and grouping by City and processing the means

# Plotting and exploration results

All the results are plotted and available in reports folder.
