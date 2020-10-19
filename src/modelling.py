from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler


data_dir = Path("data/")

columns_to_use = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
]

all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")

# replace NA's with the mean of the feature
all_data = all_data.fillna(all_data.mean())

target_column = "SalePrice"


X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train = X_train[columns_to_use]
X_test = X_test[columns_to_use]




imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)
# determine mean absolute error (MAE)
def mae_cal(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# estimate MAE for linear regression model
linear_model = LinearRegression()
mae = mae_cal(linear_model)
print(f"MAE for linear regression is {mae}")




# L1 Regularization with lasso
# determine MAE for lasso regression model with alpha = 0.1
lasso= Lasso(alpha=0.1)
mae = mae_cal(lasso)
print(f"MAE estimate for lasso alpha with 0.1: {mae}")

# calculate RMSE over several alphas
lom = []
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_lasso = [mae_cal(Lasso(alpha = alpha)).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)

optimalLassoAlpha = cv_lasso[cv_lasso == cv_lasso.min()].index.values[0]
print("Optimal ridge alpha: {}".format(optimalLassoAlpha))

lasso_model = Lasso(optimalLassoAlpha)
mae = mae_cal(lasso_model)
print(f"MAE for lasso regression with optimal alpha is {mae}")


# L2 regularization with ridge

# determine MAE for ridge regression model with alpha = 0.1
ridge = Ridge(alpha=0.1)
mae = mae_cal(ridge)
print(f"MAE estimate for ridge alpha with 0.1: {mae}")


# calculate MAE over several alphas
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 10.6, 15, 15.2, 30, 50, 75]
cv_ridge = [mae_cal(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
optimalRidgeAlpha = cv_ridge[cv_ridge == cv_ridge.min()].index.values[0]
print("Optimal ridge alpha: {}".format(optimalRidgeAlpha))

# determine MAE for ridge regression model with optimal alpha
ridge_model = Ridge(optimalRidgeAlpha)
mae = mae_cal(ridge_model)
print(f"MAE for ridge regression with optimal alpha is {mae}")


# L1&2 regularization with ElasticNet
# determine MAE for ridge regression model with alpha = 0.1
elastic = ElasticNet(alpha=0.1)
mae = mae_cal(elastic)
print(f"MAE estimate for ElasticNet alpha with 0.1: {mae}")

# calculate MAE over several alphas
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 10.6, 15, 15.2, 30, 50, 75]
cv_elastic = [mae_cal(ElasticNet(alpha=alpha)).mean() for alpha in alphas]
cv_elastic = pd.Series(cv_elastic, index=alphas)

optimalelaAlpha = cv_elastic[cv_elastic == cv_elastic.min()].index.values[0]
print("Optimal ElasticNet alpha: {}".format(optimalelaAlpha))

# determine MAE for ElasticNet model with optimal alpha
elastic_model = ElasticNet(optimalelaAlpha)
mae = mae_cal(elastic_model)
print(f"MAE for ElasticNet with optimal alpha is {mae}")

# chosen_model = Pipeline([("imputer", imputer), ("model", linear_model)])
# chosen_model.fit(X_train, y_train)
# y_pred = chosen_model.predict(X_test)
