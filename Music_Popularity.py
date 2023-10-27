#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd 
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Hot-100-Audio-Features.csv")


# In[108]:


df


# # Cleaning

# In[109]:


null = pd.isnull(df).sum()
null


# In[110]:


#Here is a matrix of null values in our data. Useful for identifying and glaring patterns that may need to be addressed.
msno.matrix(df)
plt.show()


# In[111]:


df = df.dropna()
df = df.drop_duplicates()


# In[112]:


print(df.duplicated().sum(),"\n", df.isnull().sum())


# # Preprocessing

# In[113]:


#Changing boolean column to int (0s and 1s) data type so it can be used in model.
df['spotify_track_explicit'] = df['spotify_track_explicit'].astype('int')


# In[114]:


#Creates correlation coefficients for every variable. Mask created to just consider half of the values.
corr_mat = df.corr().abs()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
tf = corr_mat.mask(mask)
tf


# #### We can see that there isn't a feature present that has a significant correlation to the target variable (popularity).
# #### Also, we must drop some of the string columns because if we were to encode them there would be over 10k new columns added.

# In[115]:


#Preparing the genre column to be included in the model. Remvoving special characters first.

df['spotify_genre'] = df['spotify_genre'].str.replace(r'\[', '').str.replace(r'\'', '').str.replace(r'\]', '')


# In[116]:


#Splitting columns with multi-genres. Considering only first 4.
split = df['spotify_genre'].str.split(",", expand=True)
max_genres = 4

for i in range(max_genres):
    genre = f'genre{i + 1}'
    df[genre] = split[i]
    
df.head()


# In[117]:


#Turning genre columns into numerical ones in order to be included in model.
df = pd.get_dummies(df, columns=['genre1','genre2','genre3','genre4'])
print(df.shape)


# # Predictive Model on Popularity

# In[119]:


#Splitting Data into a 75/25 train/test split.
X = df.drop(['spotify_track_popularity','Song','spotify_genre','spotify_track_album','Performer'], axis=1)
y = df.spotify_track_popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)


# In[ ]:


#Using 100 decision trees. This demands more computational power. If running out of curiosity, consider using less trees.
model = RandomForestRegressor(n_estimators=100, random_state=40)
model.fit(X_train, y_train)


# # Results and Intepretation

# In[146]:


#R-squared score of model
print("Model score: ",model.score(X_test,y_test))

#Root Mean Squared Error
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("RMSE:",rmse)


# #### The R-squared score indicates the fit of the model to the data. 1 being a perfect fit. This model explains 48% of the variance between the predicted and actual values.
# 
# #### The RMSE also tells us that the average difference between predicted and actual popularity levels is about 16 units. It's not clear as to how significant these units are because we are not sure what the scale of popularity is. As in, what exactly is 1 unit of popularity?

# In[143]:


plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, label='Test vs. Predicted', c='b', edgecolors='k')
plt.scatter(y_test, y_test, alpha=0.5, label='Actual', c='g', edgecolors='k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

