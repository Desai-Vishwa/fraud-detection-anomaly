# 🚨 Financial Fraud Detection

## Overview
This project focuses on identifying fraudulent financial transactions using Machine Learning. Because real-world financial data contains massive class imbalance (normal transactions vastly outnumber fraudulent ones), traditional classification models often fail. Instead, this project uses **Anomaly Detection** to isolate suspicious behavior.

## Technology Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Isolation Forest)
* **Environment:** Google Colab / Jupyter Notebook

## The Methodology
1. **Handling Class Imbalance:** Acknowledged the 99.8% to 0.2% ratio of normal vs. fraudulent transactions.
2. **Unsupervised Learning:** Dropped the target labels to allow the model to hunt for outliers blindly.
3. **Isolation Forest Algorithm:** Deployed an Isolation Forest model to isolate anomalies based on feature distances.
4. **Alert Dashboard Logic:** Filtered the dataset to extract and display only the transactions flagged as `-1` (anomalies) for further investigation. 

## Results
The model successfully parsed hundreds of thousands of transactions and isolated a highly concentrated subset of suspicious activities, mimicking the backend logic of a real-world banking alert system.
