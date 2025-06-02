import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Input, Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

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

# Transform Churn column to binary values
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Transform categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df)

# Separate features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

##########################################################################################
## STEP 3: Split the dataset into train and test sets and Create Model with Tensor Flow ##
##########################################################################################

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluation of the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

######################################
## STEP 4: Plot accuracy and losses ##
######################################

plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy during Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()