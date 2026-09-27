import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

# mode for imputing missing values with the most common category
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
X_test["Embarked"] = X_test["Embarked"].fillna(X["Embarked"].mode()[0])

# one-hot encoding
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"])

# Align columns of test set with training set
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# ---------------------------------------
# Random Forest with tuned hyperparameters (random_forest_titanic_hyparams.py)

model = RandomForestClassifier(
    n_estimators=750,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=1
)

# Validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
validation_accuracy = accuracy_score(y_val, val_predictions)
print("Validation accuracy:", validation_accuracy)


# Train the model on the entire training set
model.fit(X, y)


# Predict on test set
test_predictions = model.predict(X_test)


# Create Kaggle submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv(
    "titanic/submission_rf_titanic.csv",
    index=False
)

print("Submission file created.")