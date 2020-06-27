import csv
from flask import Blueprint,Flask, redirect, url_for, request, render_template,session

second = Blueprint("second",__name__, static_folder="static",template_folder="templates")

@second.route('/plce/<userid>')
def plce(userid):
        data = []
        with open(".data/place_dataset.csv") as csv_file:
          reader = csv.reader(csv_file)
          for row in reader:
            data.append(row)
        i = int(userid)
        X = reco(i)
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

df1 = pd.read_csv("C:\\Users\\Swapnaja\\Desktop\\BE_project\\Place_dataset.csv")


import os
if not os.path.exists('user2movie.json') or\
   not os.path.exists('movie2user.json') or\
   not os.path.exists('usermovie2rating.json') or\
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict


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


K = 13
W = np.random.randn(N,K)
b = np.zeros(N)
U = np.random.randn(M,K)
A = np.random.randn(N,M)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))
S = np.zeros(M)

def get_loss(d):
    N = float(len(d))
    sse = 0
    for k,r in d.items():
        i,j = k
        if i<72:
            p = W[i].dot(U[j]) + b[i]+c[j] + mu
            sse += (p-r)*(p-r)
    return sse / N
epochs = 25
reg = 20. 
train_losses = []
test_losses = []
for epoch in range(epochs):
    print("epoch:",epoch)
    epoch_start = datetime.now()
    
    t0 = datetime.now()
    for i in range(N) :
        
        matrix = np.eye(K)*reg
        vector = np.zeros(K)
        
        bi=0
        try:
            for j in user2movie[i]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(U[j],U[j])
                vector += (r-b[i]-c[j]-mu)*U[j]
                bi += (r-W[i].dot(U[j])-c[j]-mu)
            
            W[i] = np.linalg.solve(matrix,vector)
            b[i] = bi/(len(user2movie[i])+reg)
        
            if i % (N//10) == 0 :
               '''print("i:",i,"N:",N)'''
        except KeyError:
            pass
    print("updated w and b:", datetime.now() - t0)
    
    t0 = datetime.now()
    for j in range(M):
        
        matrix += np.eye(K)*reg
        vector += np.zeros(K)
        
        cj = 0
        try:
            for i in movie2user[j]:
                if i<72 :
                    r = usermovie2rating[(i,j)]
                    matrix += np.outer(W[i],W[i])
                    vector += (r-b[i]-c[j]-mu)*W[i]
                    cj += (r-W[i].dot(U[j])-b[i] - mu)
                
                 
            U[j] = np.linalg.solve(matrix,vector)
            c[j] = cj/(len(movie2user[i])+reg)
            
            if  j % (M//10) == 0:
                '''print("j:",j,"M:",M)'''

        except KeyError:
            
            pass
    #print(W)
    #print("updated w and b:", datetime.now() - t0)  
    #print("epoch Duration:", datetime.now() - epoch_start)
    
    
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))
    
    
    test_losses.append(get_loss(usermovie2rating_test))
    #print("calculate cost:" ,datetime.now() - t0)
    #print("train loss:", train_losses[-1])
    #print("test loss:",test_losses[-1])
    
#print("train losses:",train_losses)
#print("test losses:",test_losses)

Ut = U.transpose();


for i in range(len(W)):  
    
    for j in range(len(Ut[0])):
    
       for k in range(len(Ut)):
            A[i][j] += W[i][k] * Ut[k][j]
            

def reco(y):
    for j in range(len(A[0])):
        S[j] = A[y][j]

    S.sort()
    s1 = S[::-1]
    print("Recommended places are")
    print(s1[0:5])

    temp =0
    #ac=bc=0
    
#import distance
    X = []
    for s in range(len(s1[0:5])):
        for i,j in enumerate(A):
            for k,l in enumerate(j):
                if l==s1[s]:
                    X.append(df1.at[k,'Place'])
                    temp = temp+1
                #print (s1[s])
                #print (i,k)
                #ac,bc=distance.abcd(df1.at[k,'Place'])
                #print(ac,bc)
                    
                    #print(df1.at[k,'Place'])
    return list(X);

plt.plot(train_losses,label="train loss")
plt.plot(test_losses,label="test loss")
plt.legend()
plt.show()