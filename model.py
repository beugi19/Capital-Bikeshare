from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

import pickle

## read the file

df=pd.read_csv('hour.csv')

## preprocess

df.rename(columns={'instant':'rec_id',
'dteday':'datetime',
'holiday':'is_holiday',
'workingday':'is_workingday',
'weathersit':'weather_condition',
'hum':'humidity',
'mnth':'month',
'cnt':'total_count',
'hr':'hour',
'yr':'year'},inplace=True)

df = df[['weekday','weather_condition','season','hour','is_holiday','temp','humidity','windspeed', 'total_count']]


onehotencoder = OneHotEncoder()
bike_newdata= onehotencoder.fit_transform(np.c_[(df['weekday'], df['weather_condition'], df['season'], df['hour'])]).toarray()
bike_newdata=np.c_[(bike_newdata[:,1:10], bike_newdata[:,12:-1])]

df['IsMonday']=bike_newdata[:,0]
df['IsTuesday']=bike_newdata[:,1]
df['IsWednesday']=bike_newdata[:,2]
df['IsThursday']=bike_newdata[:,3]
df['IsFriday']=bike_newdata[:,4]
df['IsSaturday']=bike_newdata[:,5]
df['IsWeatherNice']=bike_newdata[:,6]
df['IsMisty']=bike_newdata[:,7]
df['IsRainy']=bike_newdata[:,8]
df['IsSpring']=bike_newdata[:,9]
df['IsSummer']=bike_newdata[:,10]
df['IsFall']=bike_newdata[:,11]
df['HourIs0']=bike_newdata[:,12]
df['HourIs1']=bike_newdata[:,13]
df['HourIs2']=bike_newdata[:,14]
df['HourIs3']=bike_newdata[:,15]
df['HourIs4']=bike_newdata[:,16]
df['HourIs5']=bike_newdata[:,17]
df['HourIs6']=bike_newdata[:,18]
df['HourIs7']=bike_newdata[:,19]
df['HourIs8']=bike_newdata[:,20]
df['HourIs9']=bike_newdata[:,21]
df['HourIs10']=bike_newdata[:,22]
df['HourIs11']=bike_newdata[:,23]
df['HourIs12']=bike_newdata[:,24]
df['HourIs13']=bike_newdata[:,25]
df['HourIs14']=bike_newdata[:,26]
df['HourIs15']=bike_newdata[:,27]
df['HourIs16']=bike_newdata[:,28]
df['HourIs17']=bike_newdata[:,29]
df['HourIs18']=bike_newdata[:,30]
df['HourIs19']=bike_newdata[:,31]
df['HourIs20']=bike_newdata[:,32]
df['HourIs21']=bike_newdata[:,33]
df['HourIs22']=bike_newdata[:,34]

#print(df.iloc[0,:])
df = pd.get_dummies(df)

## modelling

x = df.drop(columns = ['total_count', 'weekday', 'weather_condition', 'season', 'hour'])
y = df['total_count']

## modelling

regressor=RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x,y)
pickle.dump(regressor, open('model.pkl','wb'))