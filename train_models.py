# train_models.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("heart.csv")  # Make sure this file exists in the same folder

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)  # converts strings to 0/1

# Split features and target
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
with open("LogisticRegression.pkl", "wb") as f:
    pickle.dump(lr, f)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
with open("RandomForest.pkl", "wb") as f:
    pickle.dump(rf, f)

# Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train, y_train)
with open("svm.pkl", "wb") as f:
    pickle.dump(svm, f)

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
with open("tree.pkl", "wb") as f:
    pickle.dump(tree, f)

# Grid Search Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6],
    'min_samples_split': [2, 4],
    'random_state': [42]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_rf.fit(X_train, y_train)
with open("gridrf.pkl", "wb") as f:
    pickle.dump(grid_rf, f)

print("✅ All models trained and saved successfully.")