# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:06:44 2020

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
import joblib
sns.set_style("darkgrid")


# Loading SVD on pickle
with open('PickleSVD.pkl', 'rb') as file:
    svd = pickle.load(file)
print(type(svd))

# Using read_csv then yeeting index column
df = pd.read_csv('Movie Dataset.txt', header = None, names = ['Index', 'Cust_Id', 'Rating', 'Movie_Id'], usecols = [0,1,2,3])
df = df.drop('Index', 1)
print('-Dataset examples-')
print(df.iloc[::4000000, :])

# Reading dropped movie list
drop_movie_list = pd.read_csv('drop_movie_list.txt', header = None, names = ['Index', 'Cust_Id', 'Rating', 'Movie_Id'], usecols = [0,1,2,3])
drop_movie_list = drop_movie_list.drop('Index', 1)
print(drop_movie_list.head(10))

# Importing the Movie Id to title converter
df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print (df_title.head(10))

# Creating users and predicting their scores and displaying their 10 favorite movies
df_183928 = df[(df['Cust_Id'] == 183928) & (df['Rating'] == 5)]
df_183928 = df_183928.set_index('Movie_Id')
df_183928 = df_183928.join(df_title)['Name']

user_183928 = df_title.copy()
user_183928 = user_183928.reset_index()
user_183928 = user_183928[~user_183928['Movie_Id'].isin(drop_movie_list)]

user_183928['Estimate_Score'] = user_183928['Movie_Id'].apply(lambda x: svd.predict(183928, x).est)
user_183928 = user_183928.drop('Movie_Id', axis = 1)
user_183928 = user_183928.sort_values('Estimate_Score', ascending=False)

print("\nuser_183928's favorite movies")
print(df_183928)
print("Top 10 Predictions")
print(user_183928.head(10), "\n")

# 2nd user
df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]
df_785314 = df_785314.set_index('Movie_Id')
df_785314 = df_785314.join(df_title)['Name']

user_785314 = df_title.copy()
user_785314 = user_785314.reset_index()
user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]

user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)
user_785314 = user_785314.drop('Movie_Id', axis = 1)
user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)

print("user_785314's favorite movies")
print(df_785314)
print("Top 10 Predictions")
print(user_785314.head(10), "\n")



