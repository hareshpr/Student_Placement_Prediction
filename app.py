import streamlit as st
import pandas as pd
import joblib

clf = joblib.load('predict_model.pkl')
clf_scaler = joblib.load('predict_scaler.pkl')

reg = joblib.load('regression_model.pkl')
reg_scaler = joblib.load('regression_scaler.pkl')

kmeans = joblib.load('clustering_model.pkl')
kmeans_scaler = joblib.load('clustering_scaler.pkl')

st.title("Intelligent Student Placement Predictor")
st.write("Our AI project")

ssc = st.number_input("SSC Marks", min_value=35.0, max_value=100.0, value=75.0)
hsc = st.number_input("HSC Marks", min_value=35.0, max_value=100.0, value=75.0)
cgpa = st.number_input("CGPA", min_value=4.0, max_value=10.0, value=7.5)
degree = st.selectbox("Degree", ["Bachelors", "Masters"])
dept = st.selectbox("Department", ["CS", "IT", "DS"])
internships = st.number_input("No of Internships", min_value=0, max_value=5, value=1)
if internships == 0:
    duration = st.number_input("Internship Duration (Months)", value=0, disabled=True)
else:
    duration = st.number_input("Internship Duration (Months)", min_value=1, max_value=24, value=3)

part_time = st.selectbox("Part Time Job", ["Yes", "No"])
projects = st.number_input("No of Projects", min_value=0, max_value=10, value=2)
aptitude = st.number_input("Aptitude Score", min_value=0, max_value=100, value=70)

if st.button("Predict Placement Profile"):
    deg_val = 0
    if degree == "Masters":
        deg_val = 1
        
    pt_val = 0
    if part_time == "Yes":
        pt_val = 1
        
    ds_val = 0
    it_val = 0
    if dept == "DS":
        ds_val = 1
    elif dept == "IT":
        it_val = 1

    input_data = pd.DataFrame([[ssc, hsc, deg_val, cgpa, internships, duration,
                                pt_val, projects, aptitude, ds_val, it_val]], 
                              columns=['SSC_Marks', 'HSC_Marks', 'Degree', 'CGPA',
                                       'No_of_Internships', 'Internship_Duration_Months', 
                                       'Part_Time_Job', 'No_of_Projects', 'Aptitude_Score',
                                       'Department_DS', 'Department_IT'])

    input_scaled_clf = clf_scaler.transform(input_data)
    placed_pred = clf.predict(input_scaled_clf)[0]

    input_scaled_reg = reg_scaler.transform(input_data)
    
    input_scaled_clus = kmeans_scaler.transform(input_data)
    cluster_pred = kmeans.predict(input_scaled_clus)[0]

    st.subheader("Member 1: Classification Results")
    if placed_pred == 1:
        st.success("Prediction: Placed")
        
        st.subheader("Member 2: Regression Results")
        salary_pred = reg.predict(input_scaled_reg)[0]
        st.info("Expected Salary (LPA): " + str(round(salary_pred, 2)))
    else:
        st.error("Prediction: Not Placed")
        
        st.subheader("Member 2: Regression Results")
        st.info("Salary Prediction: Not Applicable")

    st.subheader("Member 3: Clustering Results")
    st.warning("Student Profile Cluster: " + str(cluster_pred))