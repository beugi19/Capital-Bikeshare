from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

day_dict = {'Fri':[0,0,0,0,1,0], 'Mon':[0,1,0,0,0,0],
            'Sat': [0,0,0,0,0,1], 'Sun':[0,0,0,0,0,0],
            'Thu':[0,0,0,0,1,0], 'Tue':[0,0,1,0,0,0],
            'Wed': [0,0,0,1,0,0],
            '5':[0,0,0,0,1,0], '1':[0,1,0,0,0,0],
            '6': [0,0,0,0,0,1], '0':[0,0,0,0,0,0],
            '4':[0,0,0,0,1,0], '2':[0,0,1,0,0,0],
            '3': [0,0,0,1,0,0]
            }

weather_dict={'Nice':[1,0,0], 'Misty':[0,1,0],
            'Rainy': [0,0,1], 'Snowy':[0,0,0],
            '1':[1,0,0], '2':[0,1,0],
            '3': [0,0,1], '4':[0,0,0]
            }
season_dict={'Spring':[1,0,0], 'Summer':[0,1,0],
            'Fall': [0,0,1], 'Winter':[0,0,0],
            '2':[1,0,0], '3':[0,1,0],
            '4': [0,0,1], '1':[0,0,0]
            }
hour_dict={ '0':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            '1':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '2':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            '3':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '4':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '5':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '6':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '7':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '8':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '9':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            '10':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            '11':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            '12':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            '13':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            '14':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            '15':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            '16':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            '17':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            '18':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            '19':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
            '20':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            '21':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            '22':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            '23':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            }

# cols = ['hour', 'is_holiday', 'day_of_week']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    item = [x for x in request.form.values()]
    print(item)

    ## postman begin

    data = []

    # As the The training data was dummified one, so we have to pass the 
    # test data in the same format ('hour','is_holiday','day_of_week')
    # is holiday
    if item[0] == 'Yes':
        data.append(1)
    else:
        data.append(0)
    #temperature
    data.append((int(item[1])+8)/47)
    #humidity
    data.append(int(item[2])/100)
    #windspeed
    data.append(int(item[3])/67)
    #day of the week
    data.extend(day_dict[item[4]])
    #weather
    data.extend(weather_dict[item[5]])
    #season
    data.extend(season_dict[item[6]])
    #hour
    data.extend(hour_dict[item[7]])
    
    ### postman end
    #data = []

    # As the The training data was dummified one, so we have to pass the 
    # test data in the same format ('hour','is_holiday','day_of_week')
    
   # hour = request.args.get('hr')
    #is_holiday = request.args.get('holiday')
    #day_of_week = request.args.get('weekday')
    #temp = request.args.get('temp')
    #hum = request.args.get('hum')
    #windspeed = request.args.get('windspeed')
    #weathersit = request.args.get('weathersit')
    #season = request.args.get('season')

    #data.append(hour)
    #data.append(is_holiday)
    #data.append(temp)
    #data.append(hum)
    #data.append(windspeed)

    #data.extend(day_dict[day_of_week])
    #data.extend(weather_dict[weathersit])
    #data.extend(season_dict[season])
    #data.extend(hour_dict[hour])
    

    
    # is holiday
    #if item[0] == 'Yes':
    #    data.append(1)
    #else:
    #    data.append(0)

    #data.append((int(item[1])+8)/47)
    #data.append(int(item[2])/100)
    #data.append(int(item[3])/67)
    #data.extend(day_dict[item[4]])
    #data.extend(weather_dict[item[5]])
    #data.extend(season_dict[item[6]])
    #data.extend(hour_dict[item[7]])
    
    print(data)
   
    prediction = int(model.predict([data]))
    
    # postman begin

    # return 'the predicted total bike count :' + str(prediction) 

    # postman end
   


    return render_template('index.html',pred='Total Bike ride counts on {} at {}:00 Hrs will be {}'.format(item[7], item[4],prediction))



#if __name__ == '__main__':
#    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)


if __name__ == "__main__":
    app.run(debug=True)
