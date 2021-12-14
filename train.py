#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# parameters

model_params = {
    "l2_leaf_reg": 1,
    "depth": 5,
    # 'l2_leaf_reg': 2, 'depth': 2, 'learning_rate': 0.03,
}
n_splits = 5
output_file = f"model.bin"
rs = 43

# data preparation

df = pd.read_csv("./data/train.csv", keep_default_na=False)

numerical = [
    # "1stFlrSF",
    # "2ndFlrSF",
    "3SsnPorch",
    "BedroomAbvGr",
    # "BsmtFinSF1",
    # "BsmtFinSF2",
    "BsmtFullBath",
    "BsmtHalfBath",
    "BsmtUnfSF",
    "EnclosedPorch",
    "Fireplaces",
    "FullBath",
    "GarageArea",
    "GarageCars",
    "GarageYrBlt",
    "GrLivArea",
    "HalfBath",
    "KitchenAbvGr",
    "LotArea",
    "LotFrontage",
    "MasVnrArea",
    # "MoSold",
    "OpenPorchSF",
    "ScreenPorch",
    "TotRmsAbvGrd",
    "TotalBsmtSF",
    "WoodDeckSF",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
]

categorical = [
    "BldgType",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "BsmtQual",
    "CentralAir",
    "Condition1",
    "Condition2",
    "Electrical",
    "ExterCond",
    "ExterQual",
    "Exterior1st",
    "Exterior2nd",
    "FireplaceQu",
    "Foundation",
    "Functional",
    "GarageCond",
    "GarageFinish",
    "GarageQual",
    "GarageType",
    "Heating",
    "HeatingQC",
    "HouseStyle",
    "KitchenQual",
    "LandContour",
    "LandSlope",
    "LotConfig",
    "LotShape",
    "MSSubClass",
    "MSZoning",
    "MasVnrType",
    "Neighborhood",
    "OverallCond",
    "OverallQual",
    "PavedDrive",
    "RoofMatl",
    "RoofStyle",
    "SaleCondition",
    "SaleType",
]


df["LotFrontage"].fillna(70, inplace=True)
df["GarageYrBlt"].fillna(1978, inplace=True)

df["MasVnrType"].fillna("None", inplace=True)
df["Electrical"].fillna("SBrkr", inplace=True)
df["Functional"].fillna("Typ", inplace=True)
df["SaleType"].fillna("WD", inplace=True)
df["MSZoning"].fillna("RL", inplace=True)
df["Exterior1st"].fillna("VinylSd", inplace=True)
df["KitchenQual"].fillna("TA", inplace=True)
df["Exterior2nd"].fillna("VinylSd", inplace=True)

df[categorical] = df[categorical].fillna("NA")
df = df.fillna(0)

# df["Age"] = df["YrSold"] - df["YearBuilt"]
# df["TotalBath"] = (
#     df["FullBath"]
#     + df["HalfBath"] * 0.5
#     + df["BsmtFullBath"]
#     + df["BsmtHalfBath"] * 0.5
# )
# df["TotalPorch"] = (
#     df["WoodDeckSF"]
#     + df["OpenPorchSF"]
#     + df["EnclosedPorch"]
#     + df["3SsnPorch"]
#     + df["ScreenPorch"]
# )
# numerical = numerical + ["Age", "TotalBath", "TotalPorch"]
# numerical = list(
#     set(numerical)
#     | set(
#         [
#             "YrSold",
#             "YearBuilt",
#             "FullBath",
#             "HalfBath",
#             "BsmtFullBath",
#             "BsmtHalfBath",
#             "WoodDeckSF",
#             "OpenPorchSF",
#             "EnclosedPorch",
#             "3SsnPorch",
#             "ScreenPorch",
#             "MoSold",
#         ]
#     )
# )

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=rs)


# training


def train(df_train, y_train, model_params=model_params):

    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = CatBoostRegressor(random_state=rs, logging_level="Silent", **model_params)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)

    y_pred = model.predict(X)

    return y_pred


# validation

print(f"doing validation")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=rs)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log1p(df_train["SalePrice"].values)
    y_val = np.log1p(df_val["SalePrice"].values)

    dv, model = train(df_train, y_train, model_params)
    y_pred = predict(df_val, dv, model)

    mse = mean_squared_error(y_val, y_pred)
    scores.append(mse)

    print(f"mse on fold {fold} is {mse:.5f}")
    fold = fold + 1


print("validation results:")
print("%.4f +- %.4f" % (np.mean(scores), np.std(scores)))


# training the final model

print("training the final model")

dv, model = train(
    df_full_train, np.log1p(df_full_train["SalePrice"].values), model_params
)
y_pred = predict(df_test, dv, model)

y_test = np.log1p(df_test["SalePrice"].values)
mse = mean_squared_error(y_test, y_pred)

print(f"mse={mse:.5f}")


# Save the model

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")
