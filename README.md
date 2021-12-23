# Stock_Line_Predictor

Stock Line Predictor is a Streamlit Web Application which can take the Stock Ticker (Stock Symbol) of any company as input and predict it's Stock Line. The data for the inputted Stock Ticker of any company will be automatically obtained by Web scraping the Yahoo Finance website. The Stock Line predictions are based on the LSTM model trained on the NSE-TATAGLOBAL dataset. 

This web App also provides other details of the Stock Ticker like the recent data of High, Low, Open, Close, Volume, Adj Close datewise and their statistics from 2010 to till date. Futhermore, it will display the charts of the Closing Prices vs Years with their 100 MA & 200 MA (Moving Average) trendlines. 

For instance, if we input the Stock Ticker of Google i.e. GOOG, we will obtain the following results-


<img width="960" alt="image" src="https://user-images.githubusercontent.com/62761795/147252017-d86131ce-b3dd-4d79-b908-06b8b7a0c3c1.png">


<img width="960" alt="image" src="https://user-images.githubusercontent.com/62761795/147252061-b5e995c4-f7fa-498d-a667-077a77d370b8.png">


<img width="960" alt="image" src="https://user-images.githubusercontent.com/62761795/147252656-d8df0928-c846-4818-be09-d0fb70f46793.png">
