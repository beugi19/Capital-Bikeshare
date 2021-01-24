# Bike sharing project: Capital Bikeshare
## Introduction
The aim of this project is to do analysis on the data by Capital Bikeshare, a company operating a bike sharing service in Washington DC, using several different machine learning methods.
The following questions are in the focus:

a) How can we make sure that registered users always find a bike to ride when they need it?

b) What about casual users? Do their habits differ significantly from the registered ones?

c) How does weather affect the number of rides?

d) What about air pollution and rush hour traffic?

## Technical details

The programming language used was Python 3.7.7 64-bit. The following packages were used: pandas, numpy, seaborn, matplotlib.pyplot, scikit-learn, statsmodels, plotly, datetime, holidays

## First part: data from 2011-12 (bike1.iypnb)

In the first part, I analyzed data from 2011-12, already available in database form from https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset, which contains also the main renormalized hour-by-hour meteo values.

Using pairplots, barplots, the correlation matrix and scatterplots, with Python 3.7.7.

The achieved results were the following:
1) Registered users use the service mainly during the summer and fall. 
Their demand is remarkably lower on holidays, and in addition to that they use the service the most between Tuesday and Thursday, the least on Sundays. Moreover, the usage of the bikeshare is clearly enhanced by high temperatures: all else being equal, every degree attracts an extra 180 registered users per hour. 
Finally, registered users ride Capital Bikeshare bikes between 8 and 9 am and between 5 and 7 pm, which is consistent with the hypothesis about them being mostly salaried employees cycling to/from work. In contrast, rides between 10 and 12 am and in the late evening are remarkably few. 
Therefore, I would suggest Capital Bikeshare to reinforce the bike service on workdays and in the fall. An additional reason for lack-of-bicycle complaints might be that less registered users use the service in the morning with respect to the evening. This might cause an accumulation of bikes in the suburbs and a lack of them in the city center; one could consider raising fees for customers riding away from the city and reducing them for rides towards the center.

2) Casual users show a completely different pattern. Their demand is also higher on warm and sunny days, but that is the only similarity. Causal user demand is highest on Saturdays and lowest between Monday and Thursday; their peak hours are 1-6 pm; and in the spring, casual users ride almost as much as in the summer.

3) Rain lessens the demand for Capital Bikeshare's bikes. Compared to a sunny day, mist reduced the number of customers by 15%, rain by 45%, and hail or snow by 85%. 

4) According to several reports, the rush hour in Washington DC, which can cause significant congestion, is between 3 and 7 pm. This might be a cause why more users cycle in the late afternoon and evening than in the morning. Moreover, air quality in Washington is worst between May and August. This reflects in the fact that, all other things being equal, your bike service is less used in July and August than in September and October.

## Second part: data from 2019

The second part of the project involved taking data directly from the company website (https://www.capitalbikeshare.com/system-data) and trying to set up a database with all relevant info, along with the main meteorological data (average temperature, wind speed, precipitation). 

## Third part: data from 2021

The third part focused on the usage of real-time data from https://gbfs.capitalbikeshare.com/gbfs/gbfs.json, which is in .json format.
