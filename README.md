# Telco Customer Churn Prediction with Keras

This project uses a deep learning model built with Keras (TensorFlow backend) to predict customer churn based on the Telco dataset. It includes data preprocessing, model training, evaluation, and visualization.

## 📂 Dataset

The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) available on Kaggle.

Make sure to place it in the following path:
```
./data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## ⚙️ Features

- Data cleaning and preprocessing using `pandas`
- One-hot encoding for categorical features
- Feature scaling with `StandardScaler`
- Binary classification model using a sequential neural network in `Keras`
- Dropout and EarlyStopping for regularization
- Evaluation with Accuracy, AUC, Confusion Matrix, and Classification Report
- Visualization of loss and accuracy over training epochs

## 🧪 Requirements

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib tensorflow
```

## 🚀 How to Run

```
python churn_prediction_keras.py
```

Make sure the dataset CSV file is in the `./data` directory.

---

## 🔍 Useful Pandas Commands

Here are some useful `pandas` commands used in the project:

```
import pandas as pd

# Load CSV
df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# View first rows
print(df.head())

# Overview of data types and missing values
print(df.info())

# Summary statistics
print(df.describe())

# List categorical columns
print(df.select_dtypes(include='object').columns)

# Count unique values in a column
print(df['Churn'].value_counts())

# Convert column to numeric, coercing errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop a column
df = df.drop("customerID", axis=1)

# Drop rows with missing values
df = df.dropna()
```

---

## 📈 Sample Output

After training, the script prints test set performance and generates a plot of model accuracy, AUC, and loss over training epochs.

Example:
```
Test Loss: 0.4231, Test Accuracy: 0.8031, Test AUC: 0.8607
```

It also prints a classification report and confusion matrix.

---

## 🧠 Model Summary

- Input size: Number of features after encoding
- Hidden layers: 
  - Dense(32, ReLU) + Dropout(0.3)
  - Dense(16, ReLU) + Dropout(0.3)
- Output layer: Dense(1, Sigmoid)
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy, AUC

---

Feel free to fork this project, experiment with the architecture, or try different preprocessing techniques!
