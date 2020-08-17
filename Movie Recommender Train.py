# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:16:14 2020

@author: derph
"""

import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
import pickle
sns.set_style("darkgrid")

# Using read_csv then yeeting index column
df = pd.read_csv('Movie Dataset.txt', header = None, names = ['Index', 'Cust_Id', 'Rating', 'Movie_Id'], usecols = [0,1,2,3])
df = df.drop('Index', 1)
print('-Dataset examples-')
print(df.iloc[::4000000, :])

# Reading dropped movie list
drop_movie_list = pd.read_csv('drop_movie_list.txt', header = None, names = ['Index', 'Cust_Id', 'Rating', 'Movie_Id'], usecols = [0,1,2,3])
drop_movie_list = drop_movie_list.drop('Index', 1)
print(drop_movie_list.head(10))
"""
# Pivoting dataset to create a single giant matrix
df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
print(df_p.shape)
print('-Data Examples-')
print(df_p.iloc[::10000, :])
"""
# Importing the Movie Id to title converter
df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print (df_title.head(10))

# Using SVD for collaborative filtering
reader = Reader()

# get just top 100K rows for faster run time
#data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)
svd = SVD()
    
# Loading the info of one user and the movies he/she really liked
# original id is 785314
df_183928 = df[(df['Cust_Id'] == 183928) & (df['Rating'] == 5)]
df_183928 = df_183928.set_index('Movie_Id')
df_183928 = df_183928.join(df_title)['Name']
print(df_183928)

# Creating a user
user_183928 = df_title.copy()
user_183928 = user_183928.reset_index()
user_183928 = user_183928[~user_183928['Movie_Id'].isin(drop_movie_list)]

# Training on full dataset
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

trainset = data.build_full_trainset()
svd.fit(trainset)

user_183928['Estimate_Score'] = user_183928['Movie_Id'].apply(lambda x: svd.predict(183928, x).est)

user_183928 = user_183928.drop('Movie_Id', axis = 1)

user_183928 = user_183928.sort_values('Estimate_Score', ascending=False)
print(user_183928.head(10))

# Saving model
Pkl_SVD = "PickleSVD.pkl"
with open(Pkl_SVD, 'wb') as file:
    pickle.dump(svd, file)



