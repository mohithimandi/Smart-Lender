# 🏦 SmartLender — Loan Approval Prediction System

A machine learning-powered web application that predicts loan approval using
XGBoost (Gradient Boosting) with **90.8% training accuracy** and **79.6% testing accuracy**.

---

## 📁 Project Structure

```
SmartLender/
│
├── Dataset/
│   ├── loan_prediction.csv          ← Main training dataset
│   ├── loan_prediction.xlsx
│   └── loan_prediction_1.csv
│
├── Training/
│   └── Loan_Prediction_using_ML.py  ← Full ML pipeline (run in Jupyter or as .py)
│
└── Flask/
    ├── app.py                       ← Flask web application
    ├── rdf.pkl                      ← Saved XGBoost model  ✅ (pre-generated)
    ├── scale1.pkl                   ← Saved StandardScaler ✅ (pre-generated)
    ├── requirements.txt
    ├── static/
    │   └── css/
    │       └── input.css
    └── templates/
        ├── home.html                ← Landing page
        ├── input.html               ← Prediction form
        └── output.html              ← Result page
```

---

## 🚀 How to Run the Application

### Step 1 — Install Python & Anaconda
Download Anaconda: https://www.anaconda.com/download

### Step 2 — Install Dependencies
Open **Anaconda Prompt** (or any terminal) and run:
```bash
pip install flask numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### Step 3 — Navigate to the Flask folder
```bash
cd path/to/SmartLender/Flask
```
Example:
```bash
cd C:\Users\YourName\SmartLender\Flask
```

### Step 4 — Run the Flask App
```bash
python app.py
```

### Step 5 — Open in Browser
Go to: **http://localhost:5000**

---

## 🔁 Workflow

```
Home Page  →  Click "Start Prediction"
           →  Fill in applicant details (input.html)
           →  Click "Predict Loan Approval"
           →  View result: Approved ✅ or Rejected ❌ (output.html)
```

---

## 🤖 ML Models Trained

| Model               | Training Accuracy | Testing Accuracy |
|---------------------|:-----------------:|:----------------:|
| Decision Tree       | ~100%             | ~72%             |
| Random Forest       | ~100%             | ~76%             |
| KNN                 | ~82%              | ~74%             |
| **XGBoost (Best)**  | **~90.8%**        | **~79.6%**       |

> **Best model: XGBoost** — saved as `rdf.pkl` and integrated into Flask.

---

## 📊 Features Used

| Feature             | Type        | Description                          |
|---------------------|-------------|--------------------------------------|
| Gender              | Categorical | 0=Male, 1=Female                     |
| Married             | Categorical | 1=Yes, 0=No                          |
| Dependents          | Numerical   | Number of dependents                 |
| Education           | Categorical | 1=Graduate, 0=Not Graduate           |
| Self_Employed       | Categorical | 1=Yes, 0=No                          |
| ApplicantIncome     | Numerical   | Monthly income of applicant          |
| CoapplicantIncome   | Numerical   | Monthly income of co-applicant       |
| LoanAmount          | Numerical   | Loan amount (in thousands)           |
| Loan_Amount_Term    | Numerical   | Loan term in days                    |
| Credit_History      | Binary      | 1=Good, 0=Poor/None                  |
| Property_Area       | Categorical | 2=Urban, 1=Semiurban, 0=Rural        |

---

## 🔧 Preprocessing Steps

1. **Missing Values** — Filled with mode (categorical) or median (numerical)
2. **Encoding** — Categorical columns mapped to numeric values
3. **Balancing** — SMOTE applied to balance loan approval classes
4. **Scaling** — StandardScaler applied to all features
5. **Split** — 67% training / 33% testing (`random_state=42`)

---

## 🌐 IBM Cloud Deployment (Optional)

1. Install IBM Cloud CLI: https://cloud.ibm.com/docs/cli
2. Login: `ibmcloud login`
3. Push app: `ibmcloud cf push SmartLender`

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Flask** — Web framework
- **Scikit-learn** — ML models & preprocessing
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **imbalanced-learn** — SMOTE balancing
- **Pickle** — Model serialization

---

## ✅ Quick Test

Try these values for a likely **APPROVED** prediction:
- Gender: Male | Married: Yes | Dependents: 0
- Education: Graduate | Self Employed: No
- Applicant Income: 6000 | Co-Applicant: 2000
- Loan Amount: 150 | Term: 360 | Credit History: 1 | Area: Urban

Try these for a likely **REJECTED** prediction:
- Gender: Male | Married: No | Dependents: 3
- Education: Not Graduate | Self Employed: Yes
- Applicant Income: 2000 | Co-Applicant: 0
- Loan Amount: 200 | Term: 360 | Credit History: 0 | Area: Rural
