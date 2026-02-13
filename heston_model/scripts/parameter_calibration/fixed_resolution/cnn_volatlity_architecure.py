import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Load dataset
file_path = "Generated_Data/Varying Resolution/Surfaces-Test_10000_30x16.csv"
df = pd.read_csv(file_path)

samples = df["Sample"].unique()
num_samples = len(samples)

first_sample_data = df[df["Sample"] == samples[0]]
num_strikes = len(first_sample_data["Strike"].unique())
num_expiries = len(first_sample_data["Expiry"].unique())

max_size = 480

X, y = [], []

for sample in samples:
    sample_data = df[df["Sample"] == sample]
    implied_vols = sample_data["Implied_Volatility"].values
    padded_vols = np.pad(implied_vols, (0, max_size - len(implied_vols)), mode='constant')
    vol_surface = padded_vols.reshape(30, 16)
    X.append(vol_surface)
    heston_params = sample_data.iloc[0][["v0", "eta", "rho", "theta", "kappa"]].values
    y.append(heston_params)

X = np.array(X).reshape(-1, 30, 16, 1).astype(np.float32)
y = np.array(y).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(8, kernel_size=3, padding='same', activation='relu', input_shape=(30, 16, 1)),
    Conv2D(16, kernel_size=3, padding='same', activation='relu'),
    Conv2D(32, kernel_size=3, padding='same', activation='relu'),# kernel_regularizer=l2(0.01)),
    #BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),# kernel_regularizer=l2(0.01)),
    Dense(128, activation='relu'),# kernel_regularizer=l2(0.01)),
    Dense(5)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
callback = EarlyStopping(monitor='loss', patience=25)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-4)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500,
                    batch_size=512, verbose=1, callbacks=[callback, reduce_lr])

# Save loss history
df_loss = pd.DataFrame({"Epoch": range(1, len(history.history['loss']) + 1),
                        "Training Loss": history.history['loss'], "Validation Loss": history.history['val_loss']})
df_loss.to_csv("CNN/Keras/Loss History/training_validation_loss_30x16.csv", index=False)


y_pred = model.predict(X_val)
df_results = pd.DataFrame({
    'v0_actual': y_val[:, 0],
    'eta_actual': y_val[:, 1],
    'rho_actual': y_val[:, 2],
    'theta_actual': y_val[:, 3],
    'kappa_actual': y_val[:, 4],
    'v0_predicted': y_pred[:, 0],
    'eta_predicted': y_pred[:, 1],
    'rho_predicted': y_pred[:, 2],
    'theta_predicted': y_pred[:, 3],
    'kappa_predicted': y_pred[:, 4]
})
df_results.to_csv("CNN/Keras/Parameter Comparisons/predicted_vs_actual_parameters_30x16.csv", index=False)

# Save model
model.save("CNN/Keras/Models/Heston_CNN_Model_30x16.keras")
