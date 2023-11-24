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

st.title('Stock Trend Prediction')

user_input = st.selectbox("Select Cryptocurrency:", ('BTC', 'ETH', 'DOGE', 'SOL'))

# Check if user input is 'yahoo'
if user_input.lower() == 'yahoo':
    st.warning("Please enter a valid stock ticker other than 'yahoo'.")
else:
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

    # Spliting the datatset into Training and Testing
    # data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    # data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7): int(len(df))])

    # print(data_training.shape)
    # print(data_testing.shape)

    # # MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # training_data_len = scaler.fit_transform(data_training)

    #Creat a new dataframe with only Close Price
    data = crypto.filter(['Close'])
    #Convert the dataframe to numpy array
    dataset = data.values
    # Get the number of rows to train the model on. we need this number to create our train and test sets
    # math.ceil will round up the number
    training_data_len = math.ceil(len(dataset) * .8) # We are using %80 of the data for training

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
   


    #Load model
    model = load_model('model_bitcoin.h5')

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

# Making Predictions
# y_predicted = model.predict(X_test)
# scaler = scaler.scale_

# scale_factor = 1 / scaler[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor
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
st.subheader('Prediction and Original')
fig2 = plt.figure(figsize=(6, 12))
plt.plot(y_test, 'b', label='Orignal Price')
# plt.plot(pred_price, 'r', label='Predicted Price')
pred_price
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

