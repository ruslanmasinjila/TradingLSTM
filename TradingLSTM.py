#!/usr/bin/env python
# coding: utf-8

# In[93]:


##########################################################################################
# TradingLSTM
# AUTHOR: RUSLAN MASINJILA
##########################################################################################
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# NUMBER OF COLUMNS TO BE DISPLAYED
pd.set_option('display.max_columns', 500)

# MAXIMUM TABLE WIDTH TO DISPLAY
pd.set_option('display.width', 1500)  

 
# ESTABLISH CONNECTION TO MT5 TERMINAL
if not mt5.initialize():
    print("initialize() FAILED, ERROR CODE =",mt5.last_error())
    quit()


# In[94]:


# MT5 TIMEFRAME
MN1  = mt5.TIMEFRAME_MN1
W1   = mt5.TIMEFRAME_W1
D1   = mt5.TIMEFRAME_D1
H12  = mt5.TIMEFRAME_H12
H8   = mt5.TIMEFRAME_H8
H6   = mt5.TIMEFRAME_H6
H4   = mt5.TIMEFRAME_H4
H3   = mt5.TIMEFRAME_H3
H2   = mt5.TIMEFRAME_H2
H1   = mt5.TIMEFRAME_H1
M30  = mt5.TIMEFRAME_M30
M20  = mt5.TIMEFRAME_M20
M15  = mt5.TIMEFRAME_M15
M12  = mt5.TIMEFRAME_M12
M10  = mt5.TIMEFRAME_M10
M6   = mt5.TIMEFRAME_M6
M5   = mt5.TIMEFRAME_M5
M4   = mt5.TIMEFRAME_M4
M3   = mt5.TIMEFRAME_M3
M2   = mt5.TIMEFRAME_M2
M1   = mt5.TIMEFRAME_M1

symbols = None
with open('symbols.txt') as f:
    symbols = [line.rstrip('\n') for line in f]

##########################################################################################


# In[95]:


def getRates(symbol, mt5Timeframe,offset, numCandles):
    rates_frame =  mt5.copy_rates_from_pos(symbol, mt5Timeframe, offset, numCandles)
    rates_frame = pd.DataFrame(rates_frame)
    return rates_frame

##########################################################################################


# In[96]:


numCandlesForTraining     = 10000
offset                    = 100
symbol                    = "EURUSD"
mt5Timeframe              = H1
rates_frame               =  getRates(symbol, mt5Timeframe, offset, numCandlesForTraining)


# In[98]:


# Load the dataframe
# Assuming you have already loaded the dataframe named 'rates_frame'

# Extract the 'close' column
close_data = rates_frame['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Prepare the data for LSTM
window_size      = 100  # Number of rows to use as input
nFuture          = 10   # Number of future predictions
nFirstLSTMNodes  = 50   # Number of Nodes in the first LSTM Layer
nSecondLSTMNodes = 50   # Number of Nodes in the first LSTM Layer

X, y = [], []
for i in range(len(scaled_data) - window_size - nFuture):
    window = scaled_data[i:(i + window_size), 0]
    X.append(window)
    y.append(scaled_data[(i + window_size):(i + window_size + nFuture), 0]) 
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=nFirstLSTMNodes, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(units=nSecondLSTMNodes))
model.add(Dense(units=nFuture))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

# Print the model summary and evaluation results
print(model.summary())
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)


# In[107]:


numCandlesForPrediction     = 100
offset                      = 1
symbol                      = "EURUSD"
mt5Timeframe                = H1
rates_frame                 =  getRates(symbol, mt5Timeframe, offset, numCandlesForPrediction)

# Extract the 'close' column from the new data
pricesForPrediction = rates_frame['close'].values.reshape(-1, 1)

# Normalize the new data using the same scaler used during training
scaled_new_data = scaler.transform(pricesForPrediction)

# Prepare the input sequence for prediction
new_input_sequence = scaled_new_data[-window_size:, 0]  # Take the last 100 rows as input

# Reshape the input sequence for prediction
new_input_sequence = new_input_sequence.reshape(1, window_size, 1)

# Make predictions
predicted_sequence = model.predict(new_input_sequence)

# Rescale the predictions back to the original scale
predicted_sequence = scaler.inverse_transform(predicted_sequence)

# Extract the next 10 closing price values from the predicted sequence
predicted_close_values = predicted_sequence[0]

# Print the predicted close values
print('Predicted Close Values:', predicted_close_values)


# In[101]:


predicted_close_values

