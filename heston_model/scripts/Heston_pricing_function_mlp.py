import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from sklearn.preprocessing import MinMaxScaler

# Data set allocation
data = pd.read_csv("heston_model_random_parameters.csv")
y_data = data[["Option_Price"]]
data = data[["Expiry", "Strike", "Forward", "v0","eta","rho","theta","kappa"]]

# Model Architecture
pred_val_dict = []
model = Sequential([
    Dense(64, input_dim=8, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')#,
])

# Data set split
X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2, random_state=42)
model.summary()

# Normalizes the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

# Network Settings
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=50)
history = model.fit(X_train, y_train, epochs=2000, batch_size=128, verbose=1, callbacks = [callback])
loss, mae = model.evaluate(X_train, y_train, verbose=0)
print(f"Test Loss: {loss:.5f}, Test MAE: {mae:.5f}")

# Saves trained model
model.save("Heston_Function_Model.keras")

# de-normalises data
prediction = model.predict(X_test)
prediction_original_scale = scaler.inverse_transform(prediction)
y_test_original_scale = scaler.inverse_transform(y_test)
print(prediction_original_scale)
print(y_test_original_scale)

# Uploads test data to a CSV file
comparison = pd.DataFrame({
    'Actual': np.expm1(y_test_original_scale).flatten(),
    'Predicted': np.expm1(prediction_original_scale).flatten()
})
comparison.to_csv("Heston_Function_Test_Data.csv")
