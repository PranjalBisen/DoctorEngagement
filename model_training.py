import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from datetime import datetime, time, timedelta

df = pd.read_excel('dummy_npi_data.xlsx', sheet_name='Dataset', engine='openpyxl')

df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])

df['Login_Hour'] = df['Login Time'].dt.hour + df['Login Time'].dt.minute / 60
df['Login_Day'] = df['Login Time'].dt.day
df['Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60
df['Target'] = (df['Count of Survey Attempts'] > 0).astype(int)

doctor_features = df.groupby('NPI').agg({
    'Login_Hour': ['mean', 'std'],
    'Duration': ['mean', 'max'],
    'Speciality': 'first',
    'Region': 'first'
}).reset_index()

doctor_features.columns = [
    'NPI', 
    'Doctor_Avg_Login',
    'Doctor_Std_Login',
    'Doctor_Avg_Duration',
    'Doctor_Max_Duration',
    'Doctor_Speciality',
    'Doctor_Region'
]

doctor_features['Doctor_Std_Login'] = doctor_features['Doctor_Std_Login'].fillna(0)

merged = pd.merge(df, doctor_features, on='NPI', how='left')

features = [
    'Login_Hour',
    'Login_Day',
    'Duration',
    'Doctor_Avg_Login',
    'Doctor_Std_Login',
    'Doctor_Avg_Duration',
    'Doctor_Max_Duration',
    'Doctor_Speciality',
    'Doctor_Region'
]

X = pd.get_dummies(merged[features], columns=['Doctor_Speciality', 'Doctor_Region'])
X = X.fillna(0)
y = merged['Target']

merged = merged.sort_values('Login_Day')
X = X.reindex(merged.index)
y = y.reindex(merged.index)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

param_grid = {
    'C': [0.01, 0.1, 1, 10]  
}

tscv = TimeSeriesSplit(n_splits=3)
model = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

best_model = model.best_estimator_
print(f"Best Parameters: {model.best_params_}")

y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, best_model.predict(X_test)))


joblib.dump(best_model, 'survey_model.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')

doctor_profiles = merged[['NPI'] + features].drop_duplicates()
doctor_profiles.to_csv('doctor_profiles.csv', index=False)
