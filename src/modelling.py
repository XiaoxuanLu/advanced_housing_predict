from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_dir = Path("data/")
img_dir = Path("../img")
columns_to_use = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
]


def evaluate_model(
    estimator, Xtest, ytest, train=False, Xtrain=None, ytrain=None
):
    """Evaluate an estimator object using Xtest and ytest as evaluation data. If
    train==True, use Xtrain and ytrain to train the model first. Evaluates model
    based on MAE, MSE, R^2.

    :param estimator: an estimator object with a 'fit' and 'predict' method
    :param Xtest: data to be predicted for
    :param ytest: true target values to be used in evaluation
    :param train: Default == False. If true, fit estimator using Xtrain, ytrain
    :param Xtrain: Default == None. Array or DataFrame used when training the
    model
    :param ytrain: Default == None. Array or DataFrame used when training the
    model
    :returns: evaluation metrics
    :rtype: string
    """
    if train == True:
        estimator.fit(Xtrain, ytrain)
    ypred = estimator.predict(Xtest)
    r2 = r2_score(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    return {"r2": r2, "mae": mae, "mse": mse}


all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
target_column = "SalePrice"

X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train = X_train[columns_to_use]
X_test = X_test[columns_to_use]

imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)

linear_model = LinearRegression()

pipeline = Pipeline([("imputer", imputer), ("model", linear_model)])


evaluation_dict = evaluate_model(
    pipeline, X_test, y_test, train=True, Xtrain=X_train, ytrain=y_train
)
print(f"R^2: {evaluation_dict['r2']}")
print(f"Mean Absolute Error: {evaluation_dict['mae']}")
print(f"Mean Squared Error: {evaluation_dict['mse']}")
