import csv
from flask import Blueprint,Flask, redirect, url_for, request, render_template,session
item = Blueprint("item",__name__, static_folder="static",template_folder="templates")

@item.route('/plc/<userid>')
def plc(userid):
    data = []
    with open(".data/place_dataset.csv") as csv_file:
      reader = csv.reader(csv_file)
      for row in reader :
            data.append(row)
    
    activeUser = int(userid)
    rec = abcd(activeUser)
    rec = str(rec)

    col0 = [x[0] for x in data]
    col1 = [x[1] for x in data]
    col2 = [x[2] for x in data]
    col3 = [x[3] for x in data]
    places = []
    temp = 0   
    for it in rec :
        if it in col0 :
            for k in range(0,len(col0)):
                if it == col0[k]:
                    temp = temp+1
                    it = it
                    places.append({
                        "PlaceID" : col0[k],
                        "category" : col1[k],
                        "Place" : col2[k],
                        "image" : col3[k]
                    })
                    break;
    session['places'] = places
    return render_template("reco_places.html", places= places)
    
        

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

import os
if not os.path.exists('user2movie.json') or   not os.path.exists('movie2user.json') or   not os.path.exists('usermovie2rating.json') or   not os.path.exists('usermovie2rating_test.json'):
    import preprocess

with open('user2movie.json','rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json','rb')as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json','rb') as f:
    usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json','rb')as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys()))
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m) ,r in usermovie2rating_test.items()])
M = max(m1,m2)+1
print("N:",N,"M:",M)


# In[2]:


K=10 #no of neighbors
limit=2 #no of common movies
neighbors=[]
averages=[]
deviations=[]
for i in range(M):
    try:
        users_i = movie2user[i]
        users_i_set=set(users_i)
        ratings_i={user:usermovie2rating[(user,i)] for user in users_i}
        avg_i=np.mean(list(ratings_i.values()))
        dev_i={user:(rating - avg_i) for user,rating in ratings_i.items()}
        dev_i_values=np.array(list(dev_i.values()))
        sigma_i=np.sqrt(dev_i_values.dot(dev_i_values))
        averages.append(avg_i)
        deviations.append(dev_i)
    
        sl=SortedList()
        for j in range(M):
            if j!=i:
                try:
                    #print("hello")
                    #print(i)
                    #print(j)
                    users_j = movie2user[j]
                    users_j_set = set(users_j)
                    common_users = (users_i_set & users_j_set)
                    if len(common_users)>limit:
                        ratings_j = {user:usermovie2rating[(user,j)] for user in users_j}
                        avg_j = np.mean(list(ratings_j.values()))
                        dev_j = {user:(rating - avg_j) for user,rating in ratings_j.items()}
                        dev_j_values = np.array(list(dev_j.values()))
                        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                        numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
                        w_ij= numerator/(sigma_i*sigma_j)
                
                        sl.add((-w_ij,j))
                        if(len(sl)>K):
                            del sl[-1]
                        
                except KeyError:
                    #print("Key error:", j)
                    pass
       
        
        neighbors.append(sl);
    
        #if i%1==0:
            #print(i)
    except KeyError:
        neighbors.append(None)
        averages.append(None)
        deviations.append(None)
        pass
    
def predict(i,u):
    numerator=0
    denominator=0
    if neighbors[i] is not None:
        #print("\n In Predict")
        for neg_w,j in neighbors[i]:
            #if  not (u,i) in usermovie2rating.items() and not (u,i) in usermovie2rating_test.items():
                try:
                    #print("\n In for")
                    numerator+= -neg_w*deviations[j][u]
                    #print("\nNumerator:",numerator)
                    denominator+= abs(neg_w)
                    #print("\nDenominator:",denominator)
                except KeyError:
                    pass
        if denominator==0:
            prediction= averages[i]
        else:
            prediction= numerator/denominator + averages[i]
        prediction=min(5,prediction)
        prediction=max(0.5,prediction)
        return prediction
    else:
        return -1
#print("Length is ",len(neighbors))    
train_predictions = []
train_targets = []
for(u,m),target in usermovie2rating.items():
    if(m<M-1 and u<N-1):
        #print(m," ",u)
        prediction=predict(m,u)
        if(prediction!=-1):
            train_predictions.append(prediction)
            train_targets.append(target)
    
test_predictions = []
test_targets = []
for(u,m),target in usermovie2rating_test.items():
    if(m<M-1 and u<N-1):
        
        prediction=predict(m,u)
        if(prediction!=-1):
            test_predictions.append(prediction)
            test_targets.append(target)
        
def mse(p,t):
    #print("\nIn mse")
    p = np.array(p)
    t = np.array(t)
    #print(p)
    #print(t)
    #print(np.mean((p-t)**2))
    return np.mean((p-t)**2)
    
    
print('train_mse:',mse(train_predictions,train_targets))
print('test_mse:', mse(test_predictions,test_targets))
#print(train_predictions)
#print(train_targets)

2020
# In[4]:

def abcd(activePlace) :
    i=0
    nb=[]
    popularPlaces = [5,16,31,75,32]
    #print(len(neighbors[activePlace]))
    if (neighbors[activePlace]) is not None and len(neighbors[activePlace]) is not 0:
       # print("In 2")
        while i < len(neighbors[activePlace]):
            #print("IN 3")
            #print(neighbors[activePlace][i][1])
            nb.append(neighbors[activePlace][i][1])
            i=i+1
    else:
        #print("IN ELSE")
        nb = popularPlaces
       # print(popularPlaces)
        #print(nb)
    return list(nb);
#activePlace=int(input("Enter PlaceID: "))
#print(abcd(activePlace))



# In[ ]:



