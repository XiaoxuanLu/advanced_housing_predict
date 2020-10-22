from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import skew, norm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")


data_dir = Path("data/")


all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")


target_column = "SalePrice"

# log(1+x) transform to make the saleprice normally distributed
# all_data["SalePrice"] = np.log1p(all_data["SalePrice"])

## plot the distribution of the saleprice

# sns.set_style("white")
# sns.set_color_codes(palette='deep')
# f, ax = plt.subplots(figsize=(8, 7))
# sns.distplot(all_data['SalePrice'], color="b");
# ax.xaxis.grid(False)
# ax.set(ylabel="Frequency")
# ax.set(xlabel="SalePrice")
# ax.set(title="SalePrice distribution")
# sns.despine(trim=True, left=True)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)


imputer_num = SimpleImputer(missing_values=np.nan,strategy="constant", fill_value=0)
imputer_cat = SimpleImputer(missing_values=np.nan,strategy="constant", fill_value='missing')
cat_list = list(all_data.select_dtypes(exclude=["float64", "int64"]))
num_list = list(all_data.drop(columns = target_column).select_dtypes(include=["float64", "int64"]))
numerical_transformer = Pipeline(steps=[
    ('imputer', imputer_num),
    ('scaler',StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', imputer_cat),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_list),
        ('cat', categorical_transformer, cat_list)
])


# determine mean absolute error (MAE)
def mae_cal(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae
def plot_learning_curves(model, X_train, X_test, y_train, y_test):
    train_errors, val_errors = [], []
    for m in range(1, 100):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_test, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


linear_model = LinearRegression()
linear_model = Pipeline([("preprocess", preprocessor), ("model", linear_model)])
linear_model.fit(X_train, y_train)
mae = mae_cal(linear_model)
print(f"MAE for linear model is {mae}")


# L1 Regularization with lasso
# determine MAE for lasso regression model with alpha = 0.1
lasso = Pipeline([("preprocess", preprocessor), ("model", Lasso(alpha=0.1))])
lasso.fit(X_train,y_train)
mae = mae_cal(lasso)
print(f"MAE estimate for lasso alpha with 0.1: {mae}")

# calculate MAE over several alphas
alphas = [0.0004,0.05, 0.1, 0.3, 1, 3, 5, 10,10.6, 15, 30, 50, 75]
cv_lasso = [mae_cal(Pipeline([("preprocess", preprocessor), ("model", Lasso(alpha=alpha))])).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
optimalLassoAlpha = cv_lasso[cv_lasso == cv_lasso.min()].index.values[0]
print("Optimal lasso alpha: {}".format(optimalLassoAlpha))

lasso_model = Pipeline([("preprocess", preprocessor), ("model",Lasso(optimalLassoAlpha))])
lasso_model.fit(X_train,y_train)
mae = mae_cal(lasso_model)
print(f"MAE for lasso regression with optimal alpha is {mae}")

# plot learnign curve for best lasso model as ll.png
# plot_learning_curves(lasso_model,X_train,X_test,y_train,y_test)
# plt.savefig('ll.png')

# L2 regularization with ridge

# determine MAE for ridge regression model with alpha = 0.1
ridge = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=0.1))])
ridge.fit(X_train,y_train)
mae = mae_cal(ridge)
print(f"MAE estimate for ridge alpha with 0.1: {mae}")

# calculate MAE over several alphas
alphas = [0.0004,0.05, 0.1, 0.3, 1, 3, 5, 10,10.6, 15, 30, 50, 75]
cv_ridge = [mae_cal(Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=alpha))])).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
optimalRidgeAlpha = cv_ridge[cv_ridge == cv_ridge.min()].index.values[0]
print("Optimal ridge alpha: {}".format(optimalRidgeAlpha))

ridge_model = Pipeline([("preprocess", preprocessor), ("model",Ridge(optimalRidgeAlpha))])
ridge_model.fit(X_train,y_train)
mae = mae_cal(ridge_model)
print(f"MAE for ridge regression with optimal alpha is {mae}")

# plot learning curve for best ridge model as rl.png
# plot_learning_curves(ridge_model,X_train,X_test,y_train,y_test)
# plt.savefig('rl.png')

# L1&2 regularization with ElasticNet
# determine MAE for ElasticNet regression model with alpha = 0.1
elastic = Pipeline([("preprocess", preprocessor), ("model", ElasticNet(alpha=0.1))])
elastic.fit(X_train,y_train)
mae = mae_cal(elastic)
print(f"MAE estimate for ElasticNet alpha with 0.1: {mae}")

# calculate MAE over several alphas
alphas = [0.0004,0.05, 0.1, 0.3, 1, 3, 5, 10,10.6, 15, 30, 50, 75]
cv_elastic = [mae_cal(Pipeline([("preprocess", preprocessor), ("model", ElasticNet(alpha=alpha))])).mean() for alpha in alphas]
cv_elastic= pd.Series(cv_elastic, index = alphas)
optimalElasticAlpha = cv_elastic[cv_elastic == cv_elastic.min()].index.values[0]
print("Optimal ElasticNet alpha: {}".format(optimalElasticAlpha))

elastic_model = Pipeline([("preprocess", preprocessor), ("model",ElasticNet(optimalElasticAlpha))])
elastic_model.fit(X_train,y_train)
mae = mae_cal(elastic_model)
print(f"MAE for ElasticNet regression with optimal alpha is {mae}")

# plot learning curve for best ElasticNet model as el.png
# plot_learning_curves(elstic_model,X_train,X_test,y_train,y_test)
# plt.savefig('el.png')

chosen_model = lasso_model
# y_pred = chosen_model.predict(X_test)

