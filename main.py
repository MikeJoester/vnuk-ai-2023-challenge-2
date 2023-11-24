import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler


start = dt.datetime(2017, 1,1)
end = dt.datetime(2024, 1, 1)

st.title('Crypto Predicto by Dan The Nguyen')

user_input = st.selectbox("Select Cryptocurrency:", ('BTC', 'ETH', 'DOGE', 'SOL'))

# Fetch data using yfinance
crypto = yf.download(user_input + "-USD", start = start, end = end)

# Describing Data
st.subheader('Data from 2017 - 2023')
st.write(crypto.describe())
st.write(crypto.tail(20))


#Visualization
st.subheader('Close Price')
fig = plt.figure(figsize = (12,6))
plt.plot(crypto['Close'])
st.pyplot(fig)

#Creat a new dataframe with only Close Price
data = crypto.filter(['Close'])
#Convert the dataframe to numpy array
dataset = data.values
# Get the number of rows to train the model on. we need this number to create our train and test sets
# math.ceil will round up the number
training_data_len = math.ceil(len(dataset) * .9) # We are using 90% of the data for training

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Load model
model = load_model('model_' + user_input + '.h5')

#Testing Part
test_data = scaled_data[training_data_len - 60 : , :]

#Create the data sets X_test and y_test
X_test = []
y_test = dataset[training_data_len : , :]
for i in range(60, len(test_data)):
  X_test.append(test_data[i-60 : i, 0])

# Convert the data to a numpy array 
X_test = np.array(X_test)
# Reshape the test data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the last 60 day closing price values and convert the datadrame to an array
last_60_days = data[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.fit_transform(last_60_days)
# create an empty list
new_X_test = []
# Append the past 60 days
new_X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
new_X_test = np.array(new_X_test)
# Reshape the data
new_X_test = np.reshape(new_X_test, (new_X_test.shape[0], new_X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(new_X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)

# Final Graph
st.header('Prediction and Original')
fig2 = plt.figure(figsize=(6, 12))
plt.plot(y_test, 'b', label='Orignal Price')
# plt.plot(pred_price, 'r', label='Predicted Price')
st.subheader("Predicted price after 60 days: :blue[" + str(pred_price[0][0]) + "] USD")
plt.xlabel('Time (Hour)')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(fig2)

