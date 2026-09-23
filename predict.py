import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('custom_placement_data.csv')

le = LabelEncoder()
df['Part_Time_Job'] = le.fit_transform(df['Part_Time_Job'])
df['Degree'] = le.fit_transform(df['Degree'])

df = pd.get_dummies(df, columns=['Department'], drop_first=True)

X = df.drop(['Placed', 'Salary_LPA'], axis=1)
y_classification = df['Placed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_classification, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_class, y_train_class)

y_pred = rf_model.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred)
conf_matrix = confusion_matrix(y_test_class, y_pred)

print("Accuracy Score:" , accuracy)
print("\nConfusion Matrix:\n" , conf_matrix)
print("\nClassification Report:\n" , classification_report(y_test_class, y_pred))

joblib.dump(rf_model, 'predict_model.pkl')
joblib.dump(scaler, 'predict_scaler.pkl')