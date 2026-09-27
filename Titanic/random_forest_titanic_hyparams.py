import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

"""
Hyperparameter Tuning for Random Forest on the Titanic Dataset
"""

# Preprocessing and feature selection
train = pd.read_csv("titanic/data/train.csv")
test = pd.read_csv("titanic/data/test.csv")

features = [
    "Pclass", # passenger class (1st, 2nd, 3rd)
    "Sex",
    "Age",
    "SibSp", # number of siblings/spouses aboard
    "Parch", # number of parents/children aboard
    "Fare", # price of the ticket
    "Embarked" # boarding location (C, Q, S)
]
X = train[features].copy()
y = train["Survived"]
X_test = test[features].copy()


# missing values imputation

# median because it is less sensitive to outliers than the mean
X["Age"] = X["Age"].fillna(X["Age"].median())
X_test["Age"] = X_test["Age"].fillna(X["Age"].median())

X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X_test["Fare"] = X_test["Fare"].fillna(X["Fare"].median())

# mode because embarked is a categorical variable and we want to fill missing values with the most common category
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
X_test["Embarked"] = X_test["Embarked"].fillna(X["Embarked"].mode()[0])


# one-hot encoding
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"])

# Align columns of test set with training set
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# 10-fold cross-validation for hyperparameter tuning
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

# Random Forest
model = RandomForestClassifier(
    random_state=1
)

# Hyperparameter combinations to test
param_grid1 = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
"""
Best paramters from param_grid1:
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
CV accuracy: 0.8406367041198501
"""
# following up after param_grid1 to refine the search space
param_grid2 = {
    "n_estimators": [500, 750, 1000],
    "max_depth": [None, 8, 10, 12],
    "min_samples_split": [4,5,6],
    "min_samples_leaf": [1,2],
    "max_features": ["sqrt"]
}
"""
Best paramters from param_grid2:
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 750}
CV accuracy: 0.8417602996254681
"""

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid2,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest CV accuracy:")
print(grid_search.best_score_)