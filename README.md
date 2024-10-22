# Bike sharing project: Capital Bikeshare

## Introduction
The aim of this project is to do analysis on the data by Capital Bikeshare, a company operating a bike sharing service in Washington DC, using several different machine learning methods, in order to be able to locate KPI's and to give suggestions.
The following questions are in the focus:

a) How can we make sure that registered users always find a bike to ride when they need it?

b) What about casual users? Do their habits differ significantly from the registered ones?

c) How does weather affect the number of rides?

d) What about air pollution and rush hour traffic?

e) What are the effects of the Covid-19 pandemic?

## Brief Domain Analysis

The bike sharing domain is in fast growth during the past couple years. The growing needs of urban transportation have led to giant traffic jams and high pollution. To alleviate this issue, bikesharing has become increasingly popular in North America and Europe. Growing technology, such as faster payments and better GPS technology, have induced less locking and tracking costs for bikesharing company and therefore eased the market. Dockless bike sharing has become increasingly common. The major players in the bikesharing business are Uber, Ofo, and Lyft. 

The sector is growing at approximately 6,5% a year. 

## File Description

bike1.ipynb: Insights on bike share usage between 2011 and 2012

hour.csv: corresponding .csv file

bike2.ipynb: Insights on 2019-20 bike usage

bike3.ipynb: Insights on 2021 bike usage

app.py: App giving a prediction on shared bikes

model.py: The model our app works on

index.html: Website that is called up when the app runs

temp.html, hum.html: interactive plots

requirements.txt, run.txt: text files for the app

## Technical details

The programming language used was Python 3.7.7 64-bit. The following packages were used: pandas, numpy, seaborn, matplotlib.pyplot, scikit-learn, statsmodels, plotly, datetime, holidays, flask, prophet.

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

Another very important part of the project is to do several types of regression in order to guess the number of bikes rented on a particular day. Linear Regression, K-nearest neighbors, Naive Bayes, and Random Forest were all taken into consideration.

The best predictors of high request of bikes were high temperature and low humidity, where the plots showing the correlation were striking.

## Second part: data from 2019-20

The second part of the project involved taking data directly from the company website (https://www.capitalbikeshare.com/system-data) and trying to set up a database with all relevant info, along with the main meteorological data (average temperature, wind speed, precipitation). The data were then analyzed hour-by-hour to see whether there is an daily/weekly trend which influences the whole time series (there is!). Moreover, time series analysis allowed us to remove the seasonality component, and the generic trend then appears more clearly: rides are more frequent during the central months of the year. It was furthermore possible to subdivide the stations into three groups (low, medium and high bike turnover).
The next goal is to use these data in order to find the KPI's of the business.
One very important part of the data analysis is to measure business growth. By analyzing the 2011, 2012 and 2019 data, I saw that the 2011-12 growth was 64% and the 2012-19 growth was 66% - the service stabilizing at high numbers. The effects of the Covid-19 pandemic were also taken into consideration by comparing data from May 2019 and May 2020. The 2020 data were 50-60% smaller.

Two further insights were extremely interesting:
1) The total traffic from registered users dropped significantly, however the traffic from casual users only slightly;
2) On weekends, the traffics were approximately the same, while on working days the 2020 data are clearly lower.

This is probably due to the fact that, while it is still possible to run errands and go for a Sunday bike ride (no hard lockdown), many more people worked from home in 2020 than in 2019, and since salaried employees make up the bulk of registered customers, their numbers dried up during the pandemic. 

Moreover, the full 2019-20 data were used to give a long-time prognosis using Facebook Prophet. The prognosis is rather bleak.

## Third part: data from 2021

The third part focused on the usage of real-time data from https://gbfs.capitalbikeshare.com/gbfs/gbfs.json, which is in .json format.
This database has extensive information according to which it was possible to look at, for instance, the latitudes and longitudes of the bike-sharing stations. Then, we could compare the data with the 2019 ones, making clear where a bike drain is in progress (that is, which stations continuously bikes being transported to). 
The data show that there are some stations, mainly in the city center, where bikes are taken more frequently than they are returned. This will cost the bikesharing company hefty fees in bike transportation and thus should be avoided, for instance, by making rides away from the bike drain hotspots cost extra.

## Fourth part: building a prediction app

Now, it is the time do something more complicated than simple data analysis and build an app giving predictions. We want the app to take all possible data (hour, weather, season etc.) and give an accurate prediction of rented bikes per hour. The prediction is done using random forest regression, while the interface needs to be setup in .html format. 
The app was made with Flask. It takes an input of weekday, hour, weather condition, wind speed and temperature, plus saying whether it is a holiday or not, and gives a prediction of bike usage within the hour.
The app can be found on https://bike-prevision.herokuapp.com
