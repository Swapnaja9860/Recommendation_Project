import csv
from flask import Blueprint,Flask, redirect, url_for, request, render_template,session

user = Blueprint("user",__name__, static_folder="static",template_folder="templates")

@user.route('/place1/<userid>')
def place1(userid):
        data = []
        with open(".data/place_dataset.csv") as csv_file:
          reader = csv.reader(csv_file)
          for row in reader:
            data.append(row)
        i = int(userid)
        X = recommendations(i)
        col0 = [x[0] for x in data]
        col1 = [x[1] for x in data]
        col2 = [x[2] for x in data]
        col3 = [x[3] for x in data]
        places = []
        
        for item in X:
            if item in col2 :
                for k in range(0,len(col2)):
                    if item == col2[k]:
                        places.append({
                            "PlaceID" : col0[k],
                            "category" : col1[k],
                            "Place" : col2[k],
                            "image" : col3[k] 
                        })
                        break;
        session['places'] = places
        return render_template("reco_places.html", places= places);
              #return "login successful"
    
        




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


# In[18]:


K=10 #no of neighbors
limit=2 #no of common movies
neighbors=[]
averages=[]
deviations=[]
for i in range(N):
    try:
        movies_i = user2movie[i]
        movies_i_set=set(movies_i)
        ratings_i={movie:usermovie2rating[i,movie] for movie in movies_i}
        avg_i=np.mean(list(ratings_i.values()))
        dev_i={movie:(rating - avg_i) for movie,rating in ratings_i.items()}
        dev_i_values=np.array(list(dev_i.values()))
        sigma_i=np.sqrt(dev_i_values.dot(dev_i_values))
        averages.append(avg_i)
        deviations.append(dev_i)
    
        sl=SortedList()
        for j in range(N):
            if j!=i:
                try:
                    #print("hello")
                    #print(i)
                    #print(j)
                    movies_j = user2movie[j]
                    movies_j_set = set(movies_j)
                    common_movies = (movies_i_set & movies_j_set)
                    if len(common_movies)>limit:
                        #print("Limit sufficient",i,j)
                        ratings_j = {movie:usermovie2rating[j,movie] for movie in movies_j}
                        avg_j = np.mean(list(ratings_j.values()))
                        dev_j = {movie:(rating - avg_j) for movie,rating in ratings_j.items()}
                        dev_j_values = np.array(list(dev_j.values()))
                        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                        w_ij= numerator/(sigma_i*sigma_j)
                
                        sl.add((-w_ij,j))
                        if(len(sl)>K):
                            del sl[-1]
                        
                except KeyError:
                    pass
       
        
        neighbors.append(sl);
    
    except KeyError:
        neighbors.append(None)
        averages.append(None)
        deviations.append(None)
        pass
    
def predict(i,m):
    numerator=0
    denominator=0
    #print("\nInside Predict-")
    print(neighbors[i])
    if neighbors[i] is None:
        prediction = averages[i]
    for neg_w,j in neighbors[i]:
        try:
            #print("\nInside for")
            numerator+= -neg_w*deviations[j][m]
            denominator+= abs(neg_w)
            print(numerator)
            print(denominator)
            
        except KeyError:
            pass
    if denominator==0:
        prediction= averages[i]
    else:
        prediction= numerator/denominator + averages[i]
    prediction=min(5,prediction)
    prediction=max(0.5,prediction)
    return prediction
    
train_predictions = []
train_targets = []
for(i,m),target in usermovie2rating.items():
    if(i<N-1):
        prediction=predict(i,m)
        train_predictions.append(prediction)
        train_targets.append(target)
    
test_predictions = []
test_targets = []
for(i,m),target in usermovie2rating_test.items():
    if(i<N-1):
        prediction=predict(i,m)
        test_predictions.append(prediction)
        test_targets.append(target)
        
def mse(p,t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p-t)**2)
    
print('train_mse:',mse(train_predictions,train_targets))
print('test_mse:', mse(test_predictions,test_targets))
#print(train_predictions)
#print(train_targets)
print("Length is",len(neighbors))

   


# In[27]:


placeInfo=pd.read_csv("C:/Users/Swapnaja/Downloads/Place_dataset.csv")
predictions=([])          
def recommendations(activeuser):
    #activeuser = 68
    for m in range(M-1):
        prediction = predict(activeuser,m)
        print(activeuser)
        print(m)
        print(prediction)
        predictions.append([m,prediction])
    predictions.sort(key = lambda x: x[1], reverse=True)
    #print(predictions)
    recommended_ids=np.array([a[0] for a in predictions])[:5]
    #print(recommended_ids)
    placeInfo2 = placeInfo[placeInfo['itemId'].isin(recommended_ids)].title
    #print(activeuser)
    #print(placeInfo2)
    return(placeInfo2)
 
    


# In[28]:


#activeUser=int(input("Enter userid: "))
#print("The recommended places for you are: ")
#recommendations(activeUser)


# In[23]:


print(neighbors)


# In[ ]:
