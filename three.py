from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import numpy as np


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from sklearn.utils import shuffle

# In[4]:




import tensorflow as tf

from tensorflow import keras


# In[ ]:





# In[5]:



from keras.models import Model
from keras.layers import Input,Embedding,Dot,Add,Flatten
from keras.regularizers import l2
from keras.optimizers import SGD,Adam


# In[6]:


df=pd.read_csv('Main_dataset.csv')


# In[7]:


N=df.UserID.max()+1
M=df.PlaceID.max()+1


# In[8]:


df = shuffle(df)

cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


# In[9]:


K=10
mu=df_train.Rating.mean()
epochs=25
reg=0


# In[10]:


u=Input(shape=(1,))
m=Input(shape=(1,))


# In[11]:


u_embedding= Embedding(N,K, embeddings_regularizer=l2(reg))(u)
m_embedding= Embedding(M,K, embeddings_regularizer=l2(reg))(m)


# In[12]:


u_bias= Embedding(N,1, embeddings_regularizer=l2(reg))(u)
m_bias= Embedding(M,1, embeddings_regularizer=l2(reg))(m)
x=Dot(axes=2)([u_embedding,m_embedding])


# In[13]:


x=Add()([x,u_bias,m_bias])
x=Flatten()(x)


# In[14]:


model = Model(inputs=(u,m), outputs=x )
model.compile(loss='mse',
             optimizer=SGD(lr=0.01, momentum= 0.9),
             metrics=['mse'],
             )


# In[15]:


r=model.fit(
x=[df_train.UserID.values,df_train.PlaceID.values],
    y=df_train.Rating.values-mu,
    epochs=epochs,
    batch_size=128,
    validation_data=([df_test.UserID.values,df_test.PlaceID.values],
    df_test.Rating.values-mu,
    )
)


# In[16]:


print([df_test.UserID.values,df_test.PlaceID.values],
    df_test.Rating.values)


# In[17]:


df_train.sort_values('UserID')


# In[18]:


df_test.sort_values('UserID')


# In[19]:


df_test[df_test.UserID==1]


# In[20]:


placeInfo=pd.read_csv("C:/Users/DELL/Downloads/Place_dataset.csv")


# In[21]:


def favoritePlace(activeUser,N):
    topPlace=pd.DataFrame.sort_values(
        df[df.UserID==activeUser],['Rating'],ascending=[0])[:N]

    rslt_df = placeInfo[placeInfo['itemId'].isin(topPlace.PlaceID)].title
    return rslt_df

    


# In[22]:


def topNRecommendations(activeUser,N):
    topPlace=pd.DataFrame.sort_values(
        df_test[df_test.UserID==activeUser],['Rating'],ascending=[0])[:N]

    rslt_df = placeInfo[placeInfo['itemId'].isin(topPlace.PlaceID)].title
    return rslt_df


# In[23]:





# In[44]:


def topRecommendations(activeuser):
    place_data = np.array(list(set(placeInfo.itemId)))
    #print(place_data)
    user = np.array([activeuser for i in range(len(place_data))])
    #print(user)
    predictions = model.predict([user, place_data])
    predictions = np.array([a[0] for a in predictions])
    #print(predictions)
    recommended_ids = (-predictions).argsort()[:15]
    print(recommended_ids)
    print(predictions[recommended_ids])
    placeInfo2 = placeInfo[placeInfo['itemId'].isin(recommended_ids)].title
    print(placeInfo2)


# In[ ]:


#Creating dataset for making recommendations for the first user
book_data = np.array(list(set(dataset.book_id)))
user = np.array([1 for i in range(len(book_data))])
predictions = model.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]
print(recommended_book_ids)
print(predictions[recommended_book_ids])


# In[45]:


activeUser=int(input("Enter userid: "))
print("The user's favorite places are: ")
print(favoritePlace(activeUser,5))
print("The recommended places for you are: ")
print(topRecommendations(activeUser))


# In[ ]:




