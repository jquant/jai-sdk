import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

from jai import LinearModel

np.random.seed(42)
MAX_SIZE = 50


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("test_linear_class", "classification"),
        ("test_linear_sgdclass", "sgd_classification"),
    ],
)
def test_linear_classification(name, dtype):
    iris = load_iris()
    target = "species"
    data = pd.DataFrame(iris.data)
    data.columns = iris.feature_names
    target = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )
    X_test.index.name = "id"
    X_test = X_test.reset_index()

    model = LinearModel(name, dtype, safe_mode=True)

    if model.is_valid():
        model._delete_database(model.name)

    print(model.model_parameters)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))

    model.learn(X_test, y_test)

    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))

    y_pred = model.predict(X_test, predict_proba=True)
    print(y_pred)

    y_pred = model.predict(X_test.iloc[[0]])
    print(y_pred, y_test[0])

    y_pred = model.predict(X_test.iloc[[0]], predict_proba=True)
    print(y_pred, y_test[0])

    model._delete_database(model.name)
    assert not model.is_valid(), "valid name after delete failed"


@pytest.mark.parametrize(
    "name, dtype",
    [("test_linear_reg", "regression"), ("test_linear_sgdreg", "sgd_regression")],
)
def test_linear_regression(name, dtype):

    data, labels = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    X_test.index.name = "id"
    X_test = X_test.reset_index()

    model = LinearModel(name, dtype, safe_mode=True)

    if model.is_valid():
        model._delete_database(model.name)

    print(model.model_parameters)
    print(model.fit(X_train, y_train))

    y_pred = model.predict(X_test)
    print(mean_squared_error(y_pred["predict"], y_test))

    print(model.learn(X_test, y_test))

    y_pred = model.predict(X_test)
    print(mean_squared_error(y_pred["predict"], y_test))

    y_pred = model.predict(X_test.iloc[[0]])
    print(y_pred, y_test.iloc[0])
    print(mean_squared_error([y_pred["predict"]], [y_test.iloc[0]]))

    y_pred = model.predict(X_test.iloc[[0]], predict_proba=True)
    print(y_pred, y_test.iloc[0])
    print(mean_squared_error([y_pred["predict"]], [y_test.iloc[0]]))

    model._delete_database(model.name)
    assert not model.is_valid(), "valid name after delete failed"
