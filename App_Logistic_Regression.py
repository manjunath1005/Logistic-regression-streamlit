import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

st.set_page_config("Logistic Regression - Churn Prediction", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown("""
<div class="card">
    <h1>Customer Churn Prediction</h1>
    <p>Predict whether a customer will <b>Churn</b> using <b>Logistic Regression</b></p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())


# Data Cleaning
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)


# Encoding
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# Predictions & Metrics
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)


# Performance Section
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.3f}")

st.text("Classification Report")
st.code(classification_report(y_test, y_pred))


# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importance
st.subheader("Top Features Influencing Churn")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(feature_importance.head(10))


# Prediction 
st.subheader("Predict Customer Churn")

# User Inputs
monthly_charges = st.slider("Monthly Charges ($)", 
                            float(df["MonthlyCharges"].min()), 
                            float(df["MonthlyCharges"].max()), 70.0)

tenure = st.slider("Tenure (Months)", 
                   int(df["tenure"].min()), 
                   int(df["tenure"].max()), 12)

contract = st.selectbox("Contract Type", 
                        ["Month-to-month", "One year", "Two year"])

# Create an input array of zeros matching the training features
input_data = np.zeros(len(X.columns))

# Map inputs to the correct column indices
cols = list(X.columns)

# 1. Map Numeric Features
input_data[cols.index("MonthlyCharges")] = monthly_charges
input_data[cols.index("tenure")] = tenure

# 2. Map Categorical Features (Handling drop_first=True)
# If "Month-to-month" is chosen, we leave both 'One year' and 'Two year' as 0.
if contract == "One year":
    if "Contract_One year" in cols:
        input_data[cols.index("Contract_One year")] = 1
elif contract == "Two year":
    if "Contract_Two year" in cols:
        input_data[cols.index("Contract_Two year")] = 1

# 3. Scale and Predict
input_scaled = scaler.transform([input_data])
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display Results
if prediction == 1:
    st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2%})")
else:
    st.success(f"✅ Customer is NOT likely to churn (Probability: {1 - probability:.2%})")

st.markdown('</div>', unsafe_allow_html=True)
