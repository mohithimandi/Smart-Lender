# ============================================================
# SmartLender - Loan Prediction using Machine Learning
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import imblearn
from imblearn.over_sampling import SMOTE

# ── EPIC 2: READ DATASET ─────────────────────────────────────
data = pd.read_csv('../Dataset/loan_prediction.csv')
print("Dataset Shape:", data.shape)
data.head()

# ── EPIC 2 STORY 2: UNIVARIATE ANALYSIS ──────────────────────
# Distribution plots for continuous variables
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.histplot(data['ApplicantIncome'], color='r', kde=True)
plt.subplot(122)
sns.histplot(data['Credit_History'], kde=True)
plt.show()

# Count plots for categorical variables
plt.figure(figsize=(18, 4))
plt.subplot(1, 4, 1)
sns.countplot(data['Gender'])
plt.subplot(1, 4, 2)
sns.countplot(data['Education'])
plt.show()

# ── EPIC 2 STORY 3: BIVARIATE ANALYSIS ───────────────────────
plt.figure(figsize=(20, 5))
plt.subplot(131)
sns.countplot(data['Married'], hue=data['Gender'])
plt.subplot(132)
sns.countplot(data['Self_Employed'], hue=data['Education'])
plt.subplot(133)
sns.countplot(data['Property_Area'], hue=data['Loan_Amount_Term'])
plt.show()

# ── EPIC 2 STORY 4: MULTIVARIATE ANALYSIS ────────────────────
# Visualised based on gender and income vs application status
sns.swarmplot(data['Gender'], data['ApplicantIncome'], hue=data['Loan_Status'])
plt.show()

# ── EPIC 3 STORY 1: HANDLING MISSING VALUES ───────────────────
print("Missing values per column:\n", data.isnull().sum())

data['Gender']        = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married']       = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents']    = data['Dependents'].str.replace('+', '')
data['Dependents']    = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount']    = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])
data['Loan_Amount_Term']  = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History']    = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

print("Missing values after treatment:\n", data.isnull().sum())

# ── EPIC 3 STORY 1b: HANDLING CATEGORICAL VALUES ──────────────
data['Gender']        = data['Gender'].map({'Female': 1, 'Male': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Married']       = data['Married'].map({'Yes': 1, 'No': 0})
data['Education']     = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Loan_Status']   = data['Loan_Status'].map({'Y': 1, 'N': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})

# Convert Dependents to numeric
data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce').fillna(0)

# Change float columns to int
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed',
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    data[col] = data[col].astype('int64')

data.info()

# Drop Loan_ID (not a feature)
data = data.drop('Loan_ID', axis=1)

# ── EPIC 3 STORY 2: BALANCING WITH SMOTE ─────────────────────
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

print("Before SMOTE:\n", y.value_counts())
smote = SMOTE(random_state=42)
x_bal, y_bal = smote.fit_resample(X, y)
print("After SMOTE:\n", y_bal.value_counts())

names = x_bal.columns

# ── EPIC 3 STORY 3: SCALING ───────────────────────────────────
scaler = StandardScaler()
x_bal = scaler.fit_transform(x_bal)
x_bal = pd.DataFrame(x_bal, columns=names)

# Save the scaler for Flask use
pickle.dump(scaler, open('../Flask/scale1.pkl', 'wb'))

# ── EPIC 3 STORY 4: TRAIN-TEST SPLIT ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    x_bal, y_bal, test_size=0.33, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ── EPIC 4: MODEL BUILDING ────────────────────────────────────

# Story 1 — Decision Tree
def DecisionTree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_tr   = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("=== Decision Tree ===")
    print("Training Accuracy :", accuracy_score(y_tr, y_train))
    print("Testing  Accuracy :", accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return model

dt_model = DecisionTree(X_train, X_test, y_train, y_test)

# Story 2 — Random Forest
def RandomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_tr   = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("=== Random Forest ===")
    print("Training Accuracy :", accuracy_score(y_tr, y_train))
    print("Testing  Accuracy :", accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return model

rf_model = RandomForest(X_train, X_test, y_train, y_test)

# Story 3 — KNN
def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_tr   = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("=== KNN ===")
    print("Training Accuracy :", accuracy_score(y_tr, y_train))
    print("Testing  Accuracy :", accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return model

knn_model = KNN(X_train, X_test, y_train, y_test)

# Story 4 — XGBoost (GradientBoostingClassifier)
def XGB(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_tr   = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("=== XGBoost (Gradient Boosting) ===")
    print("Training Accuracy :", accuracy_score(y_tr, y_train))
    print("Testing  Accuracy :", accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return model

xgb_model = XGB(X_train, X_test, y_train, y_test)

# Story 5 — Cross-validation & save best model
cv_scores = cross_val_score(xgb_model, x_bal, y_bal, cv=5)
print("XGBoost Cross-Val Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Save the best model (XGBoost)
model = xgb_model
pickle.dump(model, open('../Flask/rdf.pkl', 'wb'))
print("Model saved as rdf.pkl")
