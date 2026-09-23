import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

df = pd.read_csv('custom_placement_data.csv')

le = LabelEncoder()
df['Part_Time_Job'] = le.fit_transform(df['Part_Time_Job'])
df['Degree'] = le.fit_transform(df['Degree'])

df = pd.get_dummies(df, columns=['Department'], drop_first=True)

X = df.drop(['Placed', 'Salary_LPA'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Cluster Centers:\n" , kmeans.cluster_centers_)
print("\nStudents per Cluster:\n" , df['Cluster'].value_counts())

import joblib

joblib.dump(kmeans, 'clustering_model.pkl')
joblib.dump(scaler, 'clustering_scaler.pkl')