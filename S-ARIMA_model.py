
# =============================================================================
# IMPORT ALL PACKAGES
# =============================================================================
import pandas as pd
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
#SARIMA
from pyramid.arima import auto_arima

# =============================================================================
# IMPORT APPLE STOCKS' DATASET
# =============================================================================

#download apple stock dataset from https://www.kaggle.com/khoongweihao/aaplcsv/version/1
#your path to the input folder
path = r'D:\study\ARIMA\aaplcsv'

data = pd.read_csv(path + r'\AAPL.csv')
data = data[['Date','Adj Close']]
data = data.set_index('Date')

data.index = pd.to_datetime(data.index)
data.columns = ['Closing_Price']

# decomposing time series into trend, seasonality and noise 
# the time series seems to be one observation per week, so keeping the frequency as 7
result = seasonal_decompose(data, model='multiplicative',freq=7)
fig = result.plot()
plot_mpl(fig)



# =============================================================================
# SEASONAL ARIMA (SARIMA) ON THE SAME DATA
# =============================================================================

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())


data = data.reset_index()

data = data.sort_values('Date')

train = data.loc[0:int(len(data)*0.7)]

test = data.loc[int(len(data)*0.7):]

train = train.set_index('Date')
test = test.set_index('Date')
data = data.set_index('Date')


stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=len(test))

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])

pd.concat([test,future_forecast],axis=1).plot()


pd.concat([data,future_forecast],axis=1).plot()

'''this shows a not so good prediction. But for a time series data like that 
of Apple which is looks unpredictable and patternless, this model did a good job
if looked closely to see the prediction where it starts from, comparing it with the
previous window'''