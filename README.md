**Time series modelling using Seasonal ARIMA (Auto Regressive Integrated Moving Average) on Apple stocks data**

Performing prediction on stock data is always challenging as it contains various other factors. 
The AAPL.csv dataset contains only date and stock prices components. Using a time series model was obvious on such a data.
I used Seasonal ARIMA on the dataset as after decomposing the observational plot into trend, seasonality and noise,
it can be clearly seen that the seasonality does show a strong pattern.

![alt text](https://raw.githubusercontent.com/ShashankNardekar/ML_projects/master/Time_Series_Modelling/Decomposed_Observations.png)

But when predicted, the prediction goes correct with the trend, but it doesnt match the pattern that the actual observations show.
This is because of the random nature of stocks dependent on various other factors.

Below is the predicted graph:

![alt text](https://raw.githubusercontent.com/ShashankNardekar/ML_projects/master/Time_Series_Modelling/Prediction_Output.png)
