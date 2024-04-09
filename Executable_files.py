#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression Single Variable
# Using the Home prices in Monroe Township, Newjersey(USA)
# 
# sample data
# 
# |area | price |
# |-----|-------|
# |2600 | 550000|
# |-----|-------|
# |3000 | 565000|
# |-----|-------|
# |3200 | 610000|
# 
# Given these home prices find out prices of homes whose area is,
# - __3300 square feet__
# - __5000 square feet__

# In[1]:


import pandas as pd # pandas for Data manipulation
import numpy as np # Numerical computation
import matplotlib.pyplot as plt # Plotting charts
from sklearn import linear_model # machine learning model


# In[2]:


df = pd.read_csv("houseprices.csv")
df


# The regression line ( There is always a slope and intercept represented by m and b)
# - price = m * area + b
# - y = m *x1 + x2 + X3 + b

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area (sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area, df.price, color="red", marker="+")


# In[4]:


reg = linear_model.LinearRegression()
reg.fit(df[["area"]], df.price) # The first argument as to like 2d array, the second argument can be a one dimensional array


# In[5]:


np.array(3300)


# In[6]:


reg.predict([[3300]])


# In[7]:


reg.coef_


# In[8]:


reg.intercept_


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area", fontsize= 20)
plt.ylabel("price", fontsize= 20)
plt.scatter(df.area, df.price, color="red", marker="+")
plt.plot(df.area, reg.predict(df[["area"]]), color="blue")


# In[ ]:





# # logistic regression

# ### Predicting if a person would buy life insurance based on his age using logistic regression
# 
# Logistic regression is one of the techniques used for solving classification problems.
# 
# Examples include:
# - Email is spam or not
# - Will customer buy life insurance? YES/NO (Binary Classification)
# - Which party will a person vote for? (Multiclass Classification)
#   -Democratic
#   -Republican
#   -Independent

# In[10]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[12]:


plt.scatter(df.age, df.bought_insurance, marker="+", color= "red")


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(df[["age"]], df.bought_insurance, test_size=0.2)


# In[15]:


X_test


# In[16]:


y_train


# In[17]:


X_train


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model = LogisticRegression()


# In[20]:


model.fit(X_train, y_train)


# In[21]:


X_test


# In[22]:


model.predict(X_test)


# In[23]:


model.score(X_test, y_test)


# In[24]:


model.predict_proba(X_test)


# In[25]:


model.predict([[70]])


# In[ ]:





# # Hierarchical clustering

# ## Objective
# To segment the clients of a wholesale distributor on their annual spending on diverse product categories, like milk, grocery, region etc.

# ## Dataset
# Wholesale Customer Data: The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units(m.u) on diverse product categories

# ## Importing Standard ML Libraries and data

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


# Pandas can read files from an online source
data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Wholesale_customers_data.csv")


# In[28]:


# Check the first five rows
data.head()


# In[29]:


data.describe()


# There are muilple product categories-- Fresh, milk, Grocery, etc. The values represent the number of units purchased by each client for each product. Our aim is make cluster from this data that can segment similar clients together. We will use Hierarchical clustering for this problem.

# ## Scaling
# Before applying Hierarchical Clustering, we have to normalize the data so that the scale of each variable is the same. Why is this important? Well. if the scle of the variable is not thesame, the model might become biased towards the variables with a higher magnitude like Fresh or milk(refer to the above table).
# 
# So, let's normalize the data and bring all the variables to the same scale:

# In[30]:


from sklearn.preprocessing import normalize


# In[31]:


data_scaled = normalize(data)


# In[32]:


data_scaled


# In[33]:


data_scaled = pd.DataFrame(data_scaled, columns=data.columns)


# In[34]:


data_scaled.head()


# In[35]:


data_scaled.describe()


# ## Creating a Dendrogram and Identifying the numbers of clusters
# 
# Here, we can see that the scale of all the variables is almost similar. Now, are good to go. Let's first draw the dendrogram to help us decide the number of cluster for this paticular problem:

# In[36]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
plt.xticks(rotation = 45)
dend = shc.dendrogram(shc.linkage(data_scaled, method="ward"))
plt.show()


# The X-axis contains the samples and y-axis represents the distance between these samples. The vertical line with maximum distance is the blue line and hence we can decide a threshold of 6 and cut the dendrogram.

# In[37]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method="ward"))
plt.axhline(y=6, color= "r", linestyle= "--")


# We have two clusters as this line cuts the denndrogram at two points. Let's now apply hierarchical clustering for 2 clusters.

# #### Applying Hierarchical Agglomerative Clustering

# In[38]:


from sklearn.cluster import AgglomerativeClustering


# In[39]:


cluster = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
cluster.fit_predict(data_scaled)


# We can see the values of 0s and 1s in the outtput since we defined 2 clusters. 0 represents the points that belong to the first cluster and 1 represents the points in the second cluster. Let's now visualize the two clusters:

# In[40]:


cluster.labels_


# In[41]:


data_scaled.head()


# In[42]:


plt.figure(figsize=(10, 7))
plt.scatter(data_scaled["Milk"], data_scaled["Grocery"], c=cluster.labels_)


# In[43]:


plt.figure(figsize=(10, 7))
plt.scatter(data_scaled["Fresh"], data_scaled["Frozen"], c=cluster.labels_)


# Hierarchical clustering is a super useful way of segmenting observations. the advantage of not having to pre-define the number of clusters gives it quite an edge over k-Means.

# In[ ]:





# ## Movie Recommendation System
# A movie recommendation system is a way to describe a process that tries to predict movies preferred based on similar movies from an historic movie dataset.

# In[44]:


# Steps
# import data
# Data pre-procssing
  # filling missing data
  # Data transformation(converting text to vector)
# Feature extraction -> User input -> Cosine Similarities(Used to find the similarities between vectors)

# List of movies


# In[45]:


# Import libraries
import numpy as np
import pandas as pd
import difflib # For comparing movies (we will find a close match between the user input and available movies)
from sklearn.feature_extraction.text import TfidfVectorizer # Use to convert textual data into numbers
from sklearn.metrics.pairwise import cosine_similarity # To find the similarity or cosine score between the movies, 
# High similarity scocre means those movies are similar, so we can recommend them.


# In[46]:


### Data collection and processing

# loading the data from the csv file to pandas dataframe
movies_data = pd.read_csv("movies.csv")


# In[47]:


# printing the first five rows of the dataframe
movies_data.head()


# In[48]:


movies_data.tagline[0]


# In[49]:


# number of rows and columns in the data frame
movies_data.shape


# In[50]:


# selecting the relevant feature for recommendation
selected_features = ["genres", "keywords", "tagline", "cast", "director"] 
print(selected_features)


# In[51]:


movies_data[selected_features]


# In[52]:


# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")


# In[53]:


# combining all the 5 selected features
combined_features = movies_data["genres"] + " "+ movies_data["keywords"]+ " "+ movies_data["tagline"]+ " "+movies_data["cast"]+ " "+movies_data["director"]


# In[54]:


combined_features


# In[55]:


# converting the text data to feature vectors (The cosine similarity works best with numeric data)
vectorizer = TfidfVectorizer()


# In[56]:


features_vectors = vectorizer.fit_transform(combined_features)


# In[57]:


print(features_vectors)


# ### Cosine Similarity
# Finding the similaririty score(confidence value), numerical score value for all the different movies

# In[58]:


# getting the similarity scores using cosine similarity
similarity = cosine_similarity(features_vectors)


# In[59]:


similarity


# In[60]:


print(similarity.shape)


# In[61]:


# Getting the movie name from the user
movie_name = input("Enter your favorite movie name : ")


# In[62]:


# creating a list with all the movie names given in the dataset
# This will contain all the movies name in the dataset for comparism with the input given
list_of_all_titles = movies_data["title"].tolist()
list_of_all_titles


# In[63]:


# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
find_close_match


# In[64]:


# Picking the first value in the list as my close match
close_match = find_close_match[0]
print(close_match)


# In[65]:


# Finding the index of the movie with title
# Checking which particular row the close match(iron man), for its similarities
index_of_the_movie = movies_data[movies_data.title == close_match]["index"].values[0]


# In[66]:


print(index_of_the_movie)


# In[67]:


list_1 = ["Victor", 2, 3, 4, 5, 6, 7, 8, 9]
for i in enumerate(list_1):
    print(i)


# In[68]:


print(similarity[index_of_the_movie])


# In[69]:


# Getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))


# In[70]:


# The first number is the index of the movies and the second number is the similarity score between the input movie and other movies
print(similarity_score)


# In[71]:


# len of the list
len(similarity_score)


# In[72]:


# We want only the movies with highest similarity values
# We need to sort the list from the highest to the lowest similarity score

sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1], reverse = True)


# In[73]:


movies_data.title[153]


# In[74]:


sorted_similar_movies


# In[75]:


# print the names of similar movies based on the index

print("Movies suggested for you : \n")

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]["title"].values[0]
    if (i < 5):
        print(i, ".", title_from_index)
        i += 1


# ### Movie Recommandation System

# In[76]:


movie_name = input("Enter your favorite movie name : ")

list_of_all_titles = movies_data["title"].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]["index"].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1], reverse = True)

print("Movies suggested for you : \n")

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]["title"].values[0]
    if (i < 30):
        print(i, ".", title_from_index)
        i += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




