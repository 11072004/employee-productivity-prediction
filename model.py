import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import pickle
data = pd.read_csv("garments_worker_productivity.csv")

print(data.head())

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Correlation matrix
corrMatrix = numeric_data.corr()

# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
sns.heatmap(corrMatrix, annot=True, linewidths=0.5)
plt.show()

data = pd.read_csv("garments_worker_productivity.csv")

print(data.head())

print(data.isnull().sum())

print(data.describe())

data = pd.read_csv("garments_worker_productivity.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())
data.drop(['wip'], axis=1, inplace=True)

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract month
data['month'] = data['date'].dt.month

# Drop original date
data.drop(['date'], axis=1, inplace=True)
print(data['month'])

print(data['department'].value_counts())

data['department'] = data['department'].apply(
    lambda x: 'finishing' if x.replace(" ", "") == 'finishing' else 'sweing'
)

print(data['department'].value_counts())

import MultiColumnLabelEncoder

mcle = MultiColumnLabelEncoder.MultiColumnLabelEncoder()
data = mcle.fit_transform(data)

print(data.head())

x = data.drop(['actual_productivity'], axis=1)
y = data['actual_productivity']

x = x.to_numpy()

print(x[:5])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=0
)

from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

pred_test = model_lr.predict(x_test)

print("test_MSE:", mean_squared_error(y_test, pred_test))
print("test_MAE:", mean_absolute_error(y_test, pred_test))
print("R2_score:{}".format(r2_score(y_test, pred_test)))

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=200, max_depth=5)
model_rf.fit(x_train, y_train)

pred = model_rf.predict(x_test)

print("test_MSE:", mean_squared_error(y_test, pred))
print("test_MAE:", mean_absolute_error(y_test, pred))
print("R2_score:{}".format(r2_score(y_test, pred)))

import xgboost as xgb

model_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
model_xgb.fit(x_train, y_train)

pred3 = model_xgb.predict(x_test)

print("test_MSE:", mean_squared_error(y_test, pred3))
print("test_MAE:", mean_absolute_error(y_test, pred3))
print("R2_score:{}".format(r2_score(y_test, pred3)))

print("\n----- Model Comparison -----")
print("Linear Regression R2:", r2_score(y_test, pred_test))
print("Random Forest R2:", r2_score(y_test, pred))
print("XGBoost R2:", r2_score(y_test, pred3))

import pickle

with open("gwp.pkl", "wb") as f:
    pickle.dump(model_xgb, f)