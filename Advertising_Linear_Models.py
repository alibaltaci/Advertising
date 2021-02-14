# Advertising - Simple Linear Models

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()

# Load the datasets
df = pd.read_csv(r"C:\Users\TOSHIBA\Desktop\Advertising\Advertising.csv", index_col=0)
df.head()


def load_advertising():
    dataframe = pd.read_csv(r"C:\Users\TOSHIBA\Desktop\Advertising\Advertising.csv", index_col=0)
    return dataframe

df = load_advertising()
df.head()

# General Overview
df.describe().T
df.info()

sns.jointplot(x="TV", y="sales", data=df, kind="reg")
plt.show()

X = df[["TV"]]
X.head()
y = df[["sales"]]

# MODEL
reg_model = LinearRegression()
reg_model.fit(X, y)
dir(reg_model)
# y_hat = b0 + b1X

reg_model.intercept_
reg_model.coef_
reg_model.score(X, y)

# Predictive
# y_hat = 7.032 + 0.047*TV
7.032 + 0.047*160

g = sns.regplot(x="TV", y="sales", data=df, scatter_kws={'color': 'b', 's': 9})
g.set_title("Model equation: Sales = 7.03 + TV*0.05")
g.set_ylabel("Number of sales")
g.set_xlabel("TV Expenses")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

reg_model.intercept_ + reg_model.coef_*500
reg_model.predict([[165]])
new_data = [[5], [15], [30]]
reg_model.predict(new_data)

# MULTIPLE LINEAR REGRESSION
df = load_advertising()
df.head()

X = df.drop('sales', axis=1)
y = df[["sales"]]

# MODEL
reg_model = LinearRegression()
reg_model.fit(X, y)
reg_model.intercept_
reg_model.coef_

# Sales = 2.94 + TV * 0.04 + radio * 0.19 - newspaper * 0.001

# 30 unit TV, 10 unit radio, 40 unit newspaper

2.94 + 30 * 0.04 + 10 * 0.19 - 40 * 0.001

new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T
reg_model.predict(new_data)
y.head()

# Evaluating Prediction Success
reg_model.predict(X)
y_pred = reg_model.predict(X)
y_pred[0:5]
y.head()

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
df.describe()


# HOLDOUT METHOD
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=42)

X_train.head()
y_train.head()
X_test.head()
y_test.head()

X_train.shape
X_test.shape

# Train Error
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# K-FOLD CV
reg_model = LinearRegression()

# WAY 1: Using all data to calculate the error.
-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")

np.mean(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))
np.std(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))
# rmse
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))


# WAY 2: Separating the data as Train-Test, applying CV to the Train set and testing with the test set.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Validation Error
np.mean(np.sqrt(-cross_val_score(reg_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))

y_pred = reg_model.predict(X_test)

# Test Error
np.sqrt(mean_squared_error(y_test, y_pred))

# RIDGE REGRESSION
df = load_advertising()
df.head()

X = df.drop('sales', axis=1)
y = df[["sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# MODEL
ridge_model = Ridge().fit(X_train, y_train)
ridge_model.coef_
ridge_model.intercept_
ridge_model.alpha

# Predictive

# Train Error
y_pred = ridge_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test Error
y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# MODEL TUNING
alphas = 10 ** np.linspace(10, -2, 100) * 0.5
ridge_model = Ridge()
coefs = []

for i in alphas:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

ridge_params = {"alpha": 10 ** np.linspace(10, -2, 100) * 0.5}
ridge_model = Ridge()
gs_cv_ridge = GridSearchCV(ridge_model, ridge_params, cv=10).fit(X_train, y_train)
gs_cv_ridge.best_params_

# FINAL MODEL
ridge_tuned = Ridge(**gs_cv_ridge.best_params_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
pd.Series(ridge_tuned.coef_, index=X_train.columns)


# LASSO REGRESSION

# MODEL
df = load_advertising()
X = df.drop('sales', axis=1)
y = df[["sales"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.head()

lasso_model = Lasso().fit(X_train, y_train)
lasso_model.intercept_
lasso_model.coef_

# Predictive

# Train Error
lasso_model.predict(X_train)
y_pred = lasso_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test Error
lasso_model.predict(X_test)
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# MODEL TUNING
lasso_param = {"alpha": 10 ** np.linspace(10, -2, 100) * 0.5}
lasso_model = Lasso()
gs_cv_lasso = GridSearchCV(lasso_model, lasso_param, cv=10).fit(X_train, y_train)
gs_cv_lasso.best_params_
lasso_tuned = Lasso(**gs_cv_lasso.best_params_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
pd.Series(lasso_tuned.coef_, index=X_train.columns)

