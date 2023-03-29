import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from dateutil.parser import parse
df = pd.read_csv('train.csv')

def stringToint(s):
    s = s.replace(' km','')
    s = int(s)
    return s 

oglist = df['Mileage']
result = list(map(stringToint,oglist))
df['Mileage'] = result

date_time = datetime.datetime.now()
df['Age']=date_time.year - df['Prod. year']

df['Wheel'] = df['Wheel'].map({'Left wheel':0,'Right-hand drive':1})

df['Fuel type'] = df['Fuel type'].map({'Petrol':0,'Diesel':1,'CNG':2,'Hybrid':3, 'LPG':4,'Hydrogen':5,'Plug-in Hybrid':6})

df['Category'] = df['Category'].map({'Jeep':0,'Hatchback':1,'Sedan':2,'Microbus':3, 'Goods wagon':4,'Universal':5,'Coupe':6,'Minivan':7,'Cabriolet':8,'Limousine':9,'Pickup':10})

df['Leather interior'] = df['Leather interior'].map({'Yes':1,'No':0})

df['Gear box type'] = df['Gear box type'].map({'Automatic': 0,'Tiptronic':1,'Variator':2,'Manual':3}) 

X = df.drop(['ID','Price','Levy','Manufacturer','Drive wheels','Engine volume','Color','Doors','Model','Prod. year'],axis=1)
Y = df['Price']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10,random_state=42)

# lr = LinearRegression()
# lr.fit(X_train,Y_train)

# rf = RandomForestRegressor(max_depth=2, random_state=42)
# rf.fit(X_train,Y_train)

# xgb = GradientBoostingRegressor(max_depth=2,n_estimators=100,random_state=85)
# xgb.fit(X_train,Y_train)

xg = XGBRegressor()
xg_final = xg.fit(X,Y)
# xg.fit(X_train,Y_train)


# Y_pred1 = lr.predict(X_test)
# Y_pred2 = rf.predict(X_test)
# Y_pred3 = xgb.predict(X_test)
# Y_pred4 = xg.predict(X_test)

# score1 = metrics.r2_score(Y_test,Y_pred1)
# score2 = metrics.r2_score(Y_test,Y_pred2)
# score3 = metrics.r2_score(Y_test,Y_pred3)
# score4 = metrics.r2_score(Y_test,Y_pred4)



joblib.dump(xg_final,'car_price_predictor')

model = joblib.load('car_price_predictor')



def predict_price(**Input_values):
    data_new = pd.DataFrame({
       'Category' : [int(Input_values['Category'])],
       'Leather interior': [int(Input_values['Leather_interior'])],
       'Fuel type' : [int(Input_values['Fuel_type'])],
       'Mileage' : [int(Input_values['Mileage'])],
       'Cylinders': [int(Input_values['Cylinders'])],
       'Gear box type' : [int(Input_values['Gear_box_type'])],
       'Wheel' : [int(Input_values['Wheel'])],
       'Airbags' : [int(Input_values['Airbags'])],
       'Age' : [date_time.year - parse(Input_values['Production_year'], fuzzy=True).year] 
    })

    result = model.predict(data_new)
    return int(result)

