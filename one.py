from flask import Blueprint,Flask, redirect, url_for, request, render_template , session
import csv
one = Blueprint("one",__name__, static_folder="static",template_folder="templates")

@one.route('/places/<userid>')
def places(userid):
    data = []
    with open(".data/place_dataset.csv") as csv_file:
      reader = csv.reader(csv_file)
      for row in reader :
            data.append(row)
    
    activeUser = int(userid)
    fav = favoritePlace(activeUser, 5)
    rec = topNRecommendations(activeUser, 5)
    
    col0 = [x[0] for x in data]
    col1 = [x[1] for x in data]
    col2 = [x[2] for x in data]
    col3 = [x[3] for x in data]
    places = []
        
    for item in rec :
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
    return render_template("reco_places.html", places= places)

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation
data=pd.read_csv("C:/Users/Swapnaja/Downloads/Main_dataset.csv")
placeInfo=pd.read_csv("C:/Users/Swapnaja/Downloads/Place_dataset.csv")

app = Flask(__name__)

data=pd.merge(data,placeInfo,left_on='itemId',right_on="itemId")
userIds=data.userId
userIds2=data[['userId']]

data.loc[0:10,['userId']]
data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])
data

def favoritePlace(activeUser,N):
    topPlace=pd.DataFrame.sort_values(
        data[data.userId==activeUser],['Rating'],ascending=[0])[:N]
    return list(topPlace.title)

userItemRatingMatrix=pd.pivot_table(data, values='Rating',
                                    index=['userId'], columns=['itemId'])


def similarity(user1,user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")


def nearestNeighbourRatings(activeUser,K):
    try:
        similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
        for i in userItemRatingMatrix.index:
            similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],userItemRatingMatrix.loc[i])
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        nearestNeighbours=similarityMatrix[:K]
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
        predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return predictItemRating

def topNRecommendations(activeUser,N):
    try:
        predictItemRating=nearestNeighbourRatings(activeUser,10)
        placeAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
        predictItemRating=predictItemRating.drop(placeAlreadyWatched)
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
        topRecommendationTitles=(placeInfo.loc[placeInfo.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    print(topRecommendationTitles.title)
    return list(topRecommendationTitles.title)






