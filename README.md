# Intelligent Student Placement Predictor 🎓

## 📌 Project Overview
The Intelligent Student Placement Predictor is an end-to-end Machine Learning web application. Instead of relying on a single algorithm, this project features a tripartite AI architecture. It analyzes a student's academic performance, technical skills, and experience to provide a comprehensive forecast of their campus placement outcomes. 

The application evaluates inputs such as CGPA, SSC/HSC marks, Internship duration, and Aptitude scores without retraining the models on the fly, ensuring a highly optimized, instant-response web dashboard.

---

## 🧠 The AI Architecture (Team Breakdown)
This project is divided into three distinct Machine Learning pipelines, allowing each team member to own a unique AI domain:

1. **Member 1: Classification Engine (Placement Status)**
   * **Algorithm:** Random Forest Classifier
   * **Role:** Analyzes the student's profile to predict a binary outcome: *Will this student get placed? (Yes/No)*.
2. **Member 2: Regression Engine (Salary Forecasting)**
   * **Algorithm:** Random Forest Regressor
   * **Role:** If Member 1's engine predicts "Placed", this model activates to predict the continuous *Salary Package (in LPA)* based on factors like internship duration and degree level.
3. **Member 3: Clustering Engine (Student Profiling)**
   * **Algorithm:** K-Means Clustering (Unsupervised Learning)
   * **Role:** Evaluates the student's background to assign them to a mathematical cluster (e.g., "High Academics / Low Experience", "At-Risk"). This operates independently of placement status to help generate targeted improvement plans.

---

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means)
* **Data Processing:** Pandas, NumPy, StandardScaler, LabelEncoder
* **Model Serialization:** Joblib
* **Frontend Web Framework:** Streamlit

---

## 📂 Project Structure
```text
Project_Folder/
│
├── custom_placement_data.csv   # The synthesized dataset
├── predict.py                  # Member 1's training script (Classification)
├── regression.py               # Member 2's training script (Regression)
├── clustering.py               # Member 3's training script (Clustering)
├── app.py                      # The unified Streamlit web dashboard
└── README.md                   # Project documentation
```

---

## 🚀 How to Run the Application

Follow these step-by-step instructions to navigate to your project directory, set up your environment, train the models, and launch the dashboard.

### 1. Open Command Prompt and Navigate to Project Folder
Open your terminal or Command Prompt (cmd), and use the `cd` command followed by the path to your project folder to navigate inside it:
```bash
cd path/to/your/Project_Folder
```
*(Replace `path/to/your/Project_Folder` with the actual folder path on your computer, for example: `cd C:\Users\Username\Documents\Project_Folder`)*

### 2. Verify Prerequisites
Ensure you have Python 3.x installed on your system by running:
```bash
python --version
```

### 3. Install Required Dependencies
Run the following command in your terminal to install the necessary Python libraries:
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 4. Train the Models (Generate Serialized Files)
Before launching the frontend dashboard, you must train the machine learning models so `app.py` can load them. Execute the three training scripts in order:
```bash
python predict.py
python regression.py
python clustering.py
```
*(Note: These scripts will generate your saved `.joblib` or `.pkl` model files in the project folder).*

### 5. Launch the Streamlit Web App
Once the training is complete and your model files are saved, boot up the frontend server:
```bash
streamlit run app.py
```

### 6. Access the Dashboard
Streamlit will automatically launch the app in your default web browser. If it does not open automatically, copy and paste this URL into your browser:
```text
http://localhost:8501
```
