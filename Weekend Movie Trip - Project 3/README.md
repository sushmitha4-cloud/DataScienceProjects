# Weekend Movie Trip - Project 3 - Clustering

This data science project works on data related to movies to group/cluster similar movies together.

# Input data and Output Information

This project takes raw data related to movies, its ratings and tags from different files,
loads the data into data frames and then clusters the data based on K Means algorithms to group
similar movies together. 

# Source of Raw/Processed Data

Raw data has been taken from : http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# Data Exploration, Processing and Feature Engineering

1. Feature engineer - Genre score
Movies data consists of a column called "Genres", which is '|' separated field.
We are iterating over all the movies and identifying the unique genres.
Once we have the unique genres list, I give the genres a score in such a way that, 
similar genres have scores similar.
My scoring mechanism is placed in such a way that even having multiple genres generates a score
that is unique.
(I used 2^ for each of the genres.)
Example : 1,2,4,8,16.... for each of the genres.
['(no genres listed)', 0], ['Children', 1], ['Animation', 2], ['Musical', 4], 
['Western', 8], ['Documentary', 16], ['Comedy', 32],['Drama',64],
['Romance',128],['Fantasy', 256], ['Mystery', 512], ['Sci-Fi', 1024],
['Action', 2048],['War',4096],['Film-Noir',8192],['Adventure', 16384],
['Crime',32768],['Thriller', 65536],['Horror',131072],['IMAX',262144]
We then generate a genreScore for each movies, so 
my assumption is that.. 
Any movie that is Animated, Children will never be classified as a similar movie to
one with Horror and Thriller etc.
Because the Genre score for a movie that has genres like 'Children' and 'Animations'
will have the GenreScore = 3, where as the movie with genres 'Horror and Thriller' will have
a score of 2048+16384 = 196608 

2. Movies and their ratings:
Merged the movies and ratings dataframes and grouped the resultant data frame by movieId
and achieved the mean of the ratings.

Plotted this and result is available in reports folder.

3. I also tried using the tags. There are multiple tags with same content, so we converted the 
content to lower and also encoded the tags to integers.
I merged the tags data with movies data and the ratings data, grouped by tags and calculated the 
average of ratings.

4. I also identified the list of unique movies and mean rating per distinct tags.

Plotted this and result is available in reports folder.
 
# Data Clustering

I used clustering algorithms like K Means and DBSCAN to perform clustering on the data.

Data I used to cluster is as follows: 
movieId - genreScores and Average Rating.

Expectations from results of this Clustering:
Similar movies (by genres and rating) are grouped together.


# Function to return Similar movies

I have written a function that takes input, movie title, 
clusteringMechanism and integer n
and returns, n similar movies to the movie title that 
are classified based on provided clustering type mechanism


# Plotting and exploration results

All the results are plotted and available in reports folder.

