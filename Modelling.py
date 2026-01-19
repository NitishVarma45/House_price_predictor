from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

data = pd.read_csv('cleandata.csv')
X = data.drop(columns = 'price')
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print(X_train.shape)

#linear regression
column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ['location']),
    remainder='passthrough'
)

scaler = StandardScaler()

lr = LinearRegression()
pipe = make_pipeline(column_trans, scaler, lr)
pipe.fit(X_train, Y_train)
Y_pred_lr = pipe.predict(X_test)
r2_score(Y_test, Y_pred_lr)

#lasso
lasso = Lasso()
pipe = make_pipeline(column_trans, scaler, lasso)
pipe.fit(X_train, Y_train)
Y_pred_lasso = pipe.predict(X_test)
r2_score(Y_test, Y_pred_lasso)

#ridge
ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(X_train, Y_train)
Y_pred_ridge = pipe.predict(X_test)
r2_score(Y_test, Y_pred_ridge)

print("No Regularization:", r2_score(Y_test, Y_pred_lr))
print("Lasso:", r2_score(Y_test, Y_pred_lasso))
print("Ridge:", r2_score(Y_test, Y_pred_ridge))


import joblib

# after training Ridge
best_model = pipe

joblib.dump(best_model, "house_price_model.pkl")
print("Model saved successfully")