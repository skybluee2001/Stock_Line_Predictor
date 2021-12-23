import tensorflow as tf
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import pandas_datareader as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('wood_.jpg')


start="2010-01-01"
today=date.today().strftime("%Y-%m-%d")

st.title("Stock Line Predictor")

user_input=st.text_input("Enter the Stock Ticker of any company to view it's Stock Line Prediction!")
df= data.DataReader(user_input,'yahoo',start,today)


df=df.reset_index()
df.index = df['Date']

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values
train = dataset[0:int((dataset.shape[0])*0.8),:]
valid = dataset[int((dataset.shape[0])*0.8):,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])


model=load_model('keras_model_stock_prediction_lstm_model.h5')


# testing part
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)

closing_price = scaler.inverse_transform(closing_price)

st.subheader('Predictions vs Original Stock Line')
train = new_data[:int((dataset.shape[0])*0.8)]
valid = new_data[int((dataset.shape[0])*0.8):]
valid['Predictions'] = closing_price
figf=plt.figure(figsize=(12,6))
plt.plot(train['Close'])
plt.plot(valid[['Close']],label='Original',color='g')
plt.plot(valid[['Predictions']],label='Predicted',color='orange')
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(figf)


st.header("Other Details")

st.subheader("Data from 2010 till today")
st.write(df.tail())

st.subheader("Data Summary")
st.write(df.describe())

n_years=st.slider("Years of predictions",1,4)
period = n_years * 365

st.subheader('Closing price vs Year')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Years')
plt.ylabel('Stock Price')
st.pyplot(fig)

st.subheader('Closing price vs Year with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA100 Trendline')
plt.plot(df.Close)
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Year with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, label='MA100 Trendline')
plt.plot(ma200, label='MA200 Trendline')
plt.plot(df.Close)
plt.xlabel('Years')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig)

