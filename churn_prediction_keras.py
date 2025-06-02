import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import numpy as np
from keras import Input, Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

##############################
## STEP 1: Load the dataset ##
##############################

df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#############################################
## STEP 2: Data Cleaning and Preprocessing ##
#############################################

# Remove custumerID column
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric, forcing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove rows with NaN values
df = df.dropna()
df = df.reset_index(drop=True)

# Transform Churn column to binary values
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate features and target variable BEFORE one-hot encoding for easier scaling
X_cat = df.select_dtypes(include='object').columns.tolist()
X_num = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Churn' in X_num:
    X_num.remove('Churn')

# Transform categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df, columns=X_cat, drop_first=True) # drop_first=True pode ajudar a reduzir colinearidade

# Separate features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

##########################################################################################
## STEP 3: Split the dataset into train and test sets and Create Model with Tensor Flow ##
##########################################################################################

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# FEATURE SCALING
# Identificar colunas numéricas que não são resultado do one-hot encoding (se houver mais além de TotalCharges)
# No seu caso, após o get_dummies, todas as features são 0/1 ou numéricas originais.
# Vamos escalar todas as features para simplificar, já que o StandardScaler não prejudica colunas 0/1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convertendo de volta para DataFrame (opcional, mas pode ser útil para inspeção)
# X_train = pd.DataFrame(X_train, columns=X.columns)
# X_val = pd.DataFrame(X_val, columns=X.columns)
# X_test = pd.DataFrame(X_test, columns=X.columns)


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Added AUC as metric and adjuted learning rate
from keras.metrics import AUC
model.compile(
    optimizer=Adam(learning_rate=0.001), # 0.001, 0.0005, 0.0001
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')] 
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5, 
    min_lr=0.00001,
    verbose=1
)

# Class weights (opcional, mas recomendado para churn)
class_weights_val = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_val))

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Evaluation of the model on the test set
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

######################################
## STEP 4: Plot accuracy and losses ##
######################################

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Accuracy
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy / AUC', color=color)
ax1.plot(history.history['accuracy'], label='Train Accuracy', color=color, linestyle='-')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color=color, linestyle='--')
if 'auc' in history.history:
    ax1.plot(history.history['auc'], label='Train AUC', color='tab:cyan', linestyle=':')
    ax1.plot(history.history['val_auc'], label='Val AUC', color='tab:cyan', linestyle='-.')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second y-axis for Loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(history.history['loss'], label='Train Loss', color=color, linestyle='-')
ax2.plot(history.history['val_loss'], label='Val Loss', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Model Performance during Training')
plt.show()


######################################
## STEP 5: Detailed Evaluation      ##
######################################
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int) # Threshold = 0.5

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# PTo better visualize the confusion matrix:
# import seaborn as sns
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

print(f"\nAUC-ROC Score on Test Set: {roc_auc_score(y_test, y_pred_proba):.4f}")