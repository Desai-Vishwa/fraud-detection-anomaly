import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Page configuration
st.set_page_config(page_title="Fraud Alert Dashboard", layout="wide")

st.title("🚨 Real-Time Fraud Alert Dashboard")
st.write("Detecting high-risk anomalous transactions using Isolation Forest.")

# 1. Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

df = load_data()

# 2. Hardcode the contamination rate (Removes the broken slider)
contamination = 0.01

# 3. Model Training & Prediction
X = df.drop(columns=['Class'], errors='ignore')
model = IsolationForest(contamination=contamination, random_state=42)
df['Anomaly'] = model.fit_predict(X)

# 4. Summary Metrics
total_tx = len(df)
alerts_df = df[df['Anomaly'] == -1]
total_alerts = len(alerts_df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions Monitored", f"{total_tx:,}")
col2.metric("Suspicious Alerts Flagged", f"{total_alerts:,}")
col3.metric("Flagged Rate", f"{(total_alerts / total_tx) * 100:.2f}%")

# 5. Alert Table Display
st.subheader("⚠️ Flagged High-Risk Transactions")
st.write("The table below shows transactions flagged as outliers by the model:")
st.dataframe(alerts_df.head(100), use_container_width=True)
