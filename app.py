import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Approval App", layout="centered")

st.title("🏦 Loan Approval Prediction App")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_1000.csv")
    return df

df = load_data()

# ---------------------------
# Preprocessing
# ---------------------------

# Fill missing values
df["Income"] = df["Income"].fillna(df["Income"].mean())
df["Employment_Type"] = df["Employment_Type"].fillna(df["Employment_Type"].mode()[0])

# Encode Employment_Type
le = LabelEncoder()
df["Employment_Type"] = le.fit_transform(df["Employment_Type"])

# Split data
X = df.drop("Loan_Approved", axis=1)
Y = df["Loan_Approved"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# ---------------------------
# User Input Section
# ---------------------------

st.header("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Income", min_value=1000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=1000, value=200000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

employment_type = st.selectbox(
    "Employment Type",
    le.classes_
)

# Convert employment to encoded value
employment_encoded = le.transform([employment_type])[0]

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Loan_Amount": [loan_amount],
    "Credit_Score": [credit_score],
    "Employment_Type": [employment_encoded]
})

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
