import pandas as pd 
import numpy
import csv

df = pd.read_csv(r"C:\Users\padma\Documents\Hot-100-Audio-Features.csv")
#print(df)
temp =  df['tempo']

temp_null = pd.isnull(temp)
#print(temp_null)

df2 = df[['spotify_genre','tempo']]
#print(df2)

temp1 = df2['tempo'].mode()[0] 
print(temp1)