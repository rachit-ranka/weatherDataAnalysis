import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import datetime as dt
import joblib

# Load and preprocess the data
data = pd.read_csv("Dataset/bda_project_dataset.csv")
data = data[data['location_id'] == 0]
data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y')
data = data[['time', 'temperature_2m_mean']]

data.rename(columns={'temperature_2m_mean': 'temp', 'time': 'date'}, inplace=True)
data['SMA_50'] = data['temp'].rolling(window=50).mean()

plt.figure(figsize=(16, 9))
plt.plot(data['date'], data['temp'], label='Temperature')
plt.plot(data['date'], data['SMA_50'], color='red', label='SMA_50')
plt.legend()
plt.show()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['temp'].values.reshape(-1, 1))

# Prepare the dataset
prediction_days = 30
x = []
y = []

for i in range(prediction_days, len(scaled_data) - 10):
    x.append(scaled_data[i - prediction_days:i, 0])
    y.append(scaled_data[i + 10, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# Create the LSTM model
def create_lstm_model(units=50, dropout_rate=0.2, input_shape=(x_train.shape[1], 1)):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_lstm_model()

# Train the model
early_stop = EarlyStopping(monitor='loss', patience=5)
history = model.fit(
    x_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(x_test, y_test), 
    callbacks=[early_stop], 
    verbose=1
)

# Evaluate the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted temperatures
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_test, color='black', label='Actual Temperature')
plt.plot(y_pred, color='green', label='Predicted Temperature')
plt.legend()
plt.show()

# Save the model, scaler, and configuration
model.save('temperature_forecast_model.h5')
print("Model saved successfully!")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully!")

config = {'prediction_days': prediction_days}
joblib.dump(config, 'config.pkl')
print("Configuration saved successfully!")