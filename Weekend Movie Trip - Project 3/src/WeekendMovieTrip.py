#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required libraries
from pandas import DataFrame, read_csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# ### Read the data into dataframes for further manipulation

# In[2]:


movies = pd.read_csv('/Weekend Movie Trip - Project 3/data/processed/movies.csv')
tags = pd.read_csv('/Weekend Movie Trip - Project 3/data/processed/tags.csv')
ratings = pd.read_csv('/Weekend Movie Trip - Project 3/data/processed/ratings.csv')


# ### Feature Engineering

# #### As part of the below step, we are processing the genres of the movies and identifying a unique set of genres.

# In[3]:


genresSet= set() # need unique values in the set hence using the set datastructure.
for genre in movies['genres'].str.split('|').values:
    genresSet = genresSet.union(genre)
print(genresSet)    


# #### We are processing the unique genres identified and giving them unique scores in a way where clustering 
# #### or grouping the movies based on genre similarity can be achieved.

# In[4]:


# The genre scores are identified in such a way that similar genres are together.

data = [['(no genres listed)', 0], ['Children', 1], ['Animation', 2], ['Musical', 4], 
        ['Western', 8], ['Documentary', 16], ['Comedy', 32],['Drama',64],
        ['Romance',128],['Fantasy', 256], ['Mystery', 512], ['Sci-Fi', 1024],
        ['Action', 2048],['War',4096],['Film-Noir',8192],['Adventure', 16384],
        ['Crime',32768],['Thriller', 65536],['Horror',131072],['IMAX',262144]];
        
  
genreScores = pd.DataFrame(data, columns = ['genre', 'score']) 


# In[5]:


genreScores


# #### In the below step, we add up the genrescore for each movie using the genreScores data frame we build in the previous step.

# In[6]:


moviesWithCalcGenreScore = movies

temp = []
for i in movies.index:
    genres = movies.at[i, 'genres'].split('|')
    genreScore = 0
    for eachGenre in genres:
        genreScore = genreScore + (genreScores[genreScores.genre==eachGenre].score.values[0])
    temp = np.append(temp, genreScore)    
moviesWithCalcGenreScore['genreScore'] = temp     


# In[7]:


moviesWithCalcGenreScore


# ### Merge data into consolidated data frame for further analysis

# #### Below we are merging data from ratings and tags, on movieId and userId, we are using a left join because, we don't want to loose the rating records if the tags records are unavailable.

# In[8]:


ratingsandtagspermovieperuser = pd.merge(ratings, tags, how='left', on=['movieId','userId'])


# In[9]:


# dropping unnecessary columns.
ratingsandtagspermovieperuser = ratingsandtagspermovieperuser.drop(['timestamp_x', 'timestamp_y'], axis = 1) 
ratingsandtagspermovieperuser


# In[10]:


#### We are now merging data from ratingsAndTags data with moviesAndGenreScores, on movieId column
ConsolidatedMovieData = pd.merge(moviesWithCalcGenreScore,ratingsandtagspermovieperuser,on=['movieId'])


# In[11]:


ConsolidatedMovieData


# ### Exploratory analysis..

# ### Lets get the average rating per movie.

# In[12]:


moviesAvgrating = ConsolidatedMovieData.groupby('movieId')['rating'].mean()
MoviesAverageRating=moviesAvgrating.to_frame()


# In[13]:


moviesAvgrating


# ### Lets consolidate a dataframe with averageRating and genreScores

# In[14]:


MoviesAverageRatingandCalculatedGenreScore = pd.merge(moviesWithCalcGenreScore,moviesAvgrating,on=['movieId'])


# In[15]:


### data processing - remove unnecessary columns and re-index the data frame by movieId column.
MoviesAverageRatingandCalculatedGenreScore=MoviesAverageRatingandCalculatedGenreScore.drop(['title','genres'],axis=1)
MoviesAverageRatingandCalculatedGenreScore=MoviesAverageRatingandCalculatedGenreScore.set_index('movieId')


# In[16]:


MoviesAverageRatingandCalculatedGenreScore


# ### Clustering.. Using the genrescores and mean rating to cluster the data.

# #### Using Kmeans, with 4 clusters (derived 4 by running it multiple times .. not sure how to provide it with out running it)

# In[17]:


from sklearn.cluster import KMeans

# Initializing KMeans
kmeanGrp = KMeans(n_clusters=20)

# Fitting with inputs
kmeanGrp = kmeanGrp.fit(MoviesAverageRatingandCalculatedGenreScore)

clustersGrp=kmeanGrp.fit_predict(MoviesAverageRatingandCalculatedGenreScore)

# Getting the cluster centers
centersGrp = kmeanGrp.cluster_centers_

# plotting
plt.title("Clustering - Movies Genre Score & Average Ratings \n Using KMeans" )
plt.scatter(centersGrp[:, 0], centersGrp[:, 1], c='green', s=2000);


# #### Below we are just checking how the data is clustered and we can observe that movies with similar genre scores are put together.

# In[18]:


print((MoviesAverageRatingandCalculatedGenreScore.iloc[list(np.where(clustersGrp==0))[0],:]))
print((MoviesAverageRatingandCalculatedGenreScore.iloc[list(np.where(clustersGrp==1))[0],:]))
print((MoviesAverageRatingandCalculatedGenreScore.iloc[list(np.where(clustersGrp==2))[0],:]))
print((MoviesAverageRatingandCalculatedGenreScore.iloc[list(np.where(clustersGrp==3))[0],:]))


# #### Plotting KMEANS Scatter clusters

# In[19]:



# Set a 2 KMeans clustering
kmeans = KMeans(n_clusters=20)
# Compute cluster centers and predict cluster indices

clusteredData1 = kmeans.fit_predict(MoviesAverageRatingandCalculatedGenreScore)
plt.plot(figsize = (600,600))
# Plot the scatter digram

plt.scatter(MoviesAverageRatingandCalculatedGenreScore.index,clusteredData1, c= clusteredData1, alpha=0.8) 

plt.title(" Clustering based on Movies Genre score and mean ratings.\n K Means")
plt.show()


# #### DBSCAN Clustering

# In[20]:


from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=3, min_samples=1000)
clustering.fit(MoviesAverageRatingandCalculatedGenreScore)

clusteredData2 = clustering.fit_predict(MoviesAverageRatingandCalculatedGenreScore)
plt.scatter(clusteredData2,MoviesAverageRatingandCalculatedGenreScore.index, c= clusteredData2)

plt.title("Clustering based on Movies Genre score and mean ratings \n DBSCAN")


# ### Finding the similar movie from the clusters.

# In[21]:


# Set each cluster on the data.
MoviesAverageRatingandCalculatedGenreScore['Cluster1']=clusteredData1 #Kmeans cluster
MoviesAverageRatingandCalculatedGenreScore['Cluster2']=clusteredData2 #DBSCAN cluster
MoviesAverageRatingandCalculatedGenreScore


# In[22]:


MoviesAverageRatingandCalculatedGenreScore.groupby(['Cluster1'])['genreScore'].count()


# In[23]:


MoviesAverageRatingandCalculatedGenreScore.groupby(['Cluster2'])['genreScore'].count()


# #### Some data processing to get dataframes with similar movies in an order.
# #### We sort the dataframes by Cluster,genreScore and rating and that is the order in which we retrieve data for similar movies.

# In[24]:


SimilarMoviesByKMeans = MoviesAverageRatingandCalculatedGenreScore.sort_values(by=['Cluster1','genreScore','rating'],ascending=False)
SimilarMoviesByDBSCAN = MoviesAverageRatingandCalculatedGenreScore.sort_values(by=['Cluster2','genreScore','rating'],ascending=False)
SimilarMoviesByDBSCAN = SimilarMoviesByDBSCAN.reset_index()
SimilarMoviesByKMeans = SimilarMoviesByKMeans.reset_index()
SimilarMoviesByDBSCAN = SimilarMoviesByDBSCAN.drop(['Cluster1'],axis=1)
SimilarMoviesByKMeans= SimilarMoviesByKMeans.drop(['Cluster2'],axis=1)
SimilarMoviesByDBSCAN = pd.merge(SimilarMoviesByDBSCAN,movies[['movieId','title']],on='movieId')
SimilarMoviesByKMeans = pd.merge(SimilarMoviesByKMeans,movies[['movieId','title']],on='movieId')
SimilarMoviesByKMeans=SimilarMoviesByKMeans.rename(columns={"Cluster1": "ClusterId"})
SimilarMoviesByDBSCAN=SimilarMoviesByDBSCAN.rename(columns={"Cluster2": "ClusterId"})


# In[25]:


SimilarMoviesByKMeans.head(5)


# In[26]:


SimilarMoviesByDBSCAN.head(5)


# ### Method that returns top N similar movies by movieTitle and clustering mechanism (either KMeans or DBSCAN)
# #### This function gets the genreScore by movieTitle and the clusterID by movie title and then in the same cluster, pulls the movie records that have genrescore less than or equal to the current movies genreScore. It returns n records.
# 

# In[27]:


def returnTopNSimilarMovies(movieTitle,clusterType,n):
    if(clusterType == "kmeans"): moviesdata = SimilarMoviesByKMeans
    else : moviesdata = SimilarMoviesByDBSCAN
    print(moviesdata)
    genreScore = moviesdata.loc[moviesdata['title'] == movieTitle]['genreScore'].values[0]
    clusterId = moviesdata.loc[moviesdata['title'] == movieTitle]['ClusterId'].values[0]
    moviesUnderSameCluster = moviesdata.loc[(moviesdata['ClusterId'] == clusterId) & (moviesdata['genreScore'] <= genreScore)]
    return pd.DataFrame(moviesUnderSameCluster['title'].values).head(n)


# In[28]:


returnTopNSimilarMovies("Shrek the Halls (2007)","kmeans",10)


# In[29]:


returnTopNSimilarMovies("Shrek the Halls (2007)","dbscan",10)


# ### More Data Exploration

# In[30]:


# Plotting movies by its ratings.

import matplotlib.pyplot as plt

plt.figure(figsize=(1500,1500))
settings = MoviesAverageRating.plot(kind='barh')
settings.set(xlabel='Rating', ylabel='Movie ID')
plt.title("Movies and its ratings")


# ### Data processing and Feature engineering for another way of clustering.

# In[31]:


# Removing tags that are null and converting all the tags to lower case
ConsolidatedMovieDataWithNonNullTags=ConsolidatedMovieData[ConsolidatedMovieData['tag'].notnull()]
ConsolidatedMovieDataWithNonNullTags['tag'] = ConsolidatedMovieDataWithNonNullTags['tag'].str.lower()


# In[32]:


# Identifying a unique id for each tag
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(ConsolidatedMovieDataWithNonNullTags["tag"])
ConsolidatedMovieDataWithNonNullTags["tag"]=encoder.transform(ConsolidatedMovieDataWithNonNullTags["tag"])
ConsolidatedMovieDataWithNonNullTags


# In[33]:


## get mean of ratings per tag
AverageMovieRatingsByTags=ConsolidatedMovieDataWithNonNullTags.groupby(['tag'])['rating'].mean().to_frame()

## get unique list of movies by tag
UniqueMoviesByTags=ConsolidatedMovieDataWithNonNullTags.groupby(['tag'])['movieId'].unique()


# In[34]:


# merging ratings and unique list of movies together by tag
uniqueMoviesAndMeanRatinsByTags = pd.merge(UniqueMoviesByTags,AverageMovieRatingsByTags,on=['tag'])


# ### Simple data view of list of unique movies and mean rating by tag id

# In[35]:


uniqueMoviesAndMeanRatinsByTags


# ### data exploration for Average movies Ratings by tags..

# In[36]:


import matplotlib.pyplot as plt

plt.figure(figsize=(600,600))
settings = AverageMovieRatingsByTags.plot(kind='barh')
settings.set(xlabel='User Ratings', ylabel='User Tags')
plt.title("User tags vs User Ratings")


# In[37]:


AverageMovieRatingsByTags

