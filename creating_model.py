import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

cars = pd.read_csv('Cleaned_Car_data.csv')

X = cars[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = cars['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
print(pipe.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array(
    ['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5))))
