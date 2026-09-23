import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('custom_placement_data.csv')

df_placed = df[df['Placed'] == 1].copy()

le = LabelEncoder()
df_placed['Part_Time_Job'] = le.fit_transform(df_placed['Part_Time_Job'])
df_placed['Degree'] = le.fit_transform(df_placed['Degree'])

df_placed = pd.get_dummies(df_placed, columns=['Department'], drop_first=True)

X = df_placed.drop(['Placed', 'Salary_LPA'], axis=1)
y_regression = df_placed['Salary_LPA']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_regression, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = rf_regressor.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Mean Absolute Error:" , mae)
print("R-squared Score:" , r2)

joblib.dump(rf_regressor, 'regression_model.pkl')
joblib.dump(scaler, 'regression_scaler.pkl')