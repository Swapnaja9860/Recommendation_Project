#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv(".data/Google_form_ratings.csv")
N = df.UserID.max()
M = df.PlaceID.max()
df = shuffle(df)

cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]
user2movie = {}
movie2user = {}
usermovie2rating = {}
print("Calling: update_user2movie_and_movie2user")
count = 0
def update_user2movie_and_movie2user(row):
    global count
    count += 1
    #print("\n")
    i = int(row.UserID)
    #print(i)
    j = int(row.PlaceID)
    #print(j)
    if i not in user2movie:
        user2movie[i] = [j]
    else :
        user2movie[i].append(j)
        
    if j not in movie2user:
        movie2user[j] = [i]
    else :
        movie2user[j].append(i)
        
    usermovie2rating[(i,j)] = row.Rating
df_train.apply(update_user2movie_and_movie2user, axis=1)

usermovie2rating_test = {}
print("Calling: update_user2movie_and_movie2user")
count = 0
def update_usermovie2rating_test(row):
    global count
    count += 1
    
    i = int(row.UserID)
    j = int(row.PlaceID)
    usermovie2rating_test[(i,j)] = row.Rating
df_test.apply(update_usermovie2rating_test, axis=1)


with open('user2movie.json' ,'wb') as f:
    pickle.dump(user2movie, f)
    
with open('movie2user.json' ,'wb') as f:
    pickle.dump(movie2user, f)

with open('usermovie2rating.json' ,'wb') as f:
    pickle.dump(usermovie2rating, f)
    
with open('usermovie2rating_test.json' ,'wb') as f:
    pickle.dump(usermovie2rating_test, f)
