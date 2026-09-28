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


# one-hot encoding
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"])


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
Best parameters:
{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200}
CV accuracy: 0.8350187265917602
"""
# following up after param_grid1 to refine the search space
param_grid2 = {
    "n_estimators": [150, 200, 300, 400],
    "max_depth": [None],
    "min_samples_split": [10, 15, 20],
    "min_samples_leaf": [2, 3, 4],
    "max_features": ["sqrt"]
}
"""
Best paramters from param_grid2:
{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 150}
CV accuracy: 0.8372784019975029
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