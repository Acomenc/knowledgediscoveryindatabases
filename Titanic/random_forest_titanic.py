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

# one-hot encoding
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"])

# ---------------------------------------
# Random Forest with tuned hyperparameters (random_forest_titanic_hyparams.py)

model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=None, 
    min_samples_split=15, 
    min_samples_leaf=2, 
    max_features="sqrt", 
    random_state=1 
    )
# Note: Tuning without imputation produced the same Kaggle accuracy despite different hyperparameters.

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
    "titanic/submission_rf_titanic_no_imputation.csv",
    index=False
)

print("Submission file created.")