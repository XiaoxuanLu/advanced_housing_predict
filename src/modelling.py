from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer

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

linear_model = Pipeline([("imputer", imputer), ("model", LinearRegression())])
# fit train and test data.
linear_model.fit(X_train, y_train)
# predict test data
y_pred = linear_model.predict(X_test)
print(f"MAE for linear regression is {mean_absolute_error(y_test, y_pred)}")




# L1 Regularization with lasso
# Lasso tuning for lambda
lasso_model = Lasso()
# y_pred = lasso_model.predict(X_test)
# print(f"MAE for lasso regression is {mean_absolute_error(y_test, y_pred)}")
parameters= {'alpha':[x for x in [0.0005,0.001,0.01,0.1,1]]}
lasso_model = GridSearchCV(lasso_model, param_grid=parameters)
lasso_model.fit(X_train, y_train)
print("The best value of Alpha for Lasso is: ", lasso_model.best_params_)

lasso_model = Pipeline([("imputer", imputer), ("model", Lasso(0.0005))])
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
print(f"MAE for lasso regression is {mean_absolute_error(y_test, y_pred)}")


# L2 regularization with ridge
ridge_model = Ridge()
parameters= {'alpha':[x for x in [0.1,0.2,0.4,0.5,0.7,0.8,1]]}
ridge_model=GridSearchCV(ridge_model, param_grid=parameters)
ridge_model.fit(X_train,y_train)
print("The best value of Alpha for Ridge is: ",ridge_model.best_params_)

ridge_model = Pipeline([("imputer", imputer), ("model", Ridge(1))])
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
print(f"MAE for ridge regression is {mean_absolute_error(y_test, y_pred)}")

# Test different alphas for ridge
#alphas = np.logspace(start=-2, stop=2, base=10, num=30)
#cv_ridge = [cross_validation_evaluation(Ridge(alpha=alpha)) for alpha in alphas]
#optimised_alpha_r = alphas[cv_ridge.index(min(cv_ridge))]
#print('Optimised alpha for ridge is: ' + str(optimised_alpha_r))

# Instantiate the improved ridge regression model
#ridge_regression_improved_model = Pipeline([("imputer", imputer), ("model", Ridge(alpha=optimised_alpha_r))])

#ridge_regression_improved_model.fit(X_train, y_train)
#y_pred = ridge_regression_improved_model.predict(X_test)
#print(f"MAE for improved ridge regression is {mean_absolute_error(y_test, y_pred)}")

elastic_model = Pipeline([("imputer", imputer), ("model", ElasticNet())])
elastic_model.fit(X_train, y_train)
y_pred = elastic_model.predict(X_test)
print(f"MAE for elasticnet model is {mean_absolute_error(y_test, y_pred)}")
# chosen_model = Pipeline([("imputer", imputer), ("model", linear_model)])
# chosen_model.fit(X_train, y_train)
# y_pred = chosen_model.predict(X_test)
