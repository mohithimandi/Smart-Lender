# 🏦 Smart-Lender: Applicant Credibility Prediction for Loan Approval

An AI-powered web application that predicts whether a loan applicant is eligible for approval based on their personal, financial, and demographic details using Machine Learning.

## 📌 Overview

Smart-Lender is a Machine Learning-based loan eligibility prediction system designed to assist banks and financial institutions in making faster and more accurate loan approval decisions.

The application analyzes applicant information such as income, education, employment status, marital status, loan amount, credit history, and property area to predict whether the loan should be approved.

This project combines Machine Learning with a Flask web application to provide an easy-to-use interface for real-time predictions. :contentReference[oaicite:0]{index=0}

---

# ✨ Features

- 🤖 Machine Learning-based loan approval prediction
- 📊 Real-time applicant eligibility prediction
- 🌐 Responsive web interface
- ⚡ Fast prediction using trained ML model
- 📁 Easy model deployment using Flask
- 💾 Pre-trained model saved using Pickle
- 📈 Clean and user-friendly interface

---

# 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Tools
- Jupyter Notebook
- VS Code
- Git & GitHub

---

# 📂 Project Structure

```
Smart-Lender/
│
├── Dataset/
│   └── loan_prediction.csv
│
├── Training/
│   ├── Model_Training.ipynb
│   └── Loan_Model.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The model is trained using a Loan Prediction dataset containing applicant information such as:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

---

# ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Saving
8. Flask Deployment
9. Real-Time Prediction

---

# 🧠 Machine Learning Algorithms

The project can be trained using multiple classification algorithms including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

The best-performing model is saved and integrated into the Flask application.

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/mohithimandi/Smart-Lender.git
```

```bash
cd Smart-Lender
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# 📈 Input Parameters

The application accepts the following inputs:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

---

# 🎯 Output

The application predicts:

- ✅ Loan Approved
- ❌ Loan Rejected

based on the applicant's information.

---

# 📷 Application Workflow

```
Applicant Details
        │
        ▼
 Data Preprocessing
        │
        ▼
 Machine Learning Model
        │
        ▼
 Loan Eligibility Prediction
        │
        ▼
 Display Result
```

---

# 📦 Python Libraries Used

- Flask
- NumPy
- Pandas
- Scikit-learn
- Pickle

---

# 📌 Future Enhancements

- User Authentication
- Loan Risk Score
- Explainable AI (Prediction Reason)
- Email Notification
- Database Integration
- Cloud Deployment
- Dashboard Analytics
- PDF Report Generation

---

# 📊 Sample Prediction

| Applicant Income | Credit History | Loan Amount | Prediction |
|-----------------|---------------|------------|------------|
| 5000 | 1 | 120 | Approved |
| 2500 | 0 | 180 | Rejected |

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch

```
git checkout -b feature-name
```

3. Commit changes

```
git commit -m "Added new feature"
```

4. Push

```
git push origin feature-name
```

5. Create a Pull Request

---

# 📄 License

This project is intended for educational and learning purposes.

---

# 👨‍💻 Author

**Mohith Imandi**

- GitHub: https://github.com/mohithimandi

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps support future improvements!
