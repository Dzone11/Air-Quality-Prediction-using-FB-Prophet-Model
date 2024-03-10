# Air-Quality-Prediction-using-FB-Prophet-Model

Air Quality Prediction using FB Prophet model

Air quality is a crucial factor that affects the health and well-being of humans and other living organisms. It is influenced by various natural and anthropogenic sources of pollutants, such as traffic, industry, agriculture, and weather. Monitoring and forecasting air quality is therefore important for environmental management and public awareness.

In this project, we will introduce a project that aims to predict the air quality in an Italian city using a machine learning model called FB Prophet. We will first describe the data set that we used, then explain the main features and advantages of FB Prophet, and finally show some results and insights from our analysis.

The data set

The data set that we used for this project is the Air Quality data set from the UC Irvine Machine Learning Repository . It contains 9358 instances of hourly averaged responses from an array of five metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level, within an Italian city. Data were recorded from March 2004 to February 2005 (one year), representing the longest freely available recordings of on-field deployed air quality chemical sensor devices responses.

The data set includes the following variables:

- Date: The date and time of the measurement
- CO(GT): True hourly averaged concentration of CO in mg/m^3 (reference analyzer)
- PT08.S1(CO): PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
- NMHC(GT): True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
- C6H6(GT): True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
- PT08.S2(NMHC): PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
- NOx(GT): True hourly averaged NOx concentration in ppb (reference analyzer)
- PT08.S3(NOx): PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
- NO2(GT): True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
- PT08.S4(NO2): PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
- PT08.S5(O3): PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
- T: Temperature in °C
- RH: Relative Humidity (%)
- AH: Absolute Humidity

The data set also contains some missing values, which are tagged with -200. These values need to be handled properly before applying any machine learning model.

The FB Prophet model

FB Prophet is a forecasting procedure implemented in R and Python by Facebook's Core Data Science team . It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Prophet follows the sklearn model API. We create an instance of the Prophet class and then call its fit and predict methods. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

Prophet also allows us to specify some parameters to tune the model, such as:

- growth: The growth trend, either 'linear' or 'logistic'
- changepoints: A list of dates at which to include potential changepoints
- n_changepoints: The number of potential changepoints to include
- changepoint_range: The proportion of the history in which trend changepoints will be estimated
- changepoint_prior_scale: The scale of the prior distribution for the changepoint locations
- seasonality_mode: The mode of the seasonality, either 'additive' or 'multiplicative'
- seasonality_prior_scale: The scale of the prior distribution for the seasonal components
- holidays: A dataframe with columns 'ds' and 'holiday' that specifies the dates of holidays or events
- holidays_prior_scale: The scale of the prior distribution for the holiday components
- mcmc_samples: The number of Markov Chain Monte Carlo samples to use for uncertainty estimation

Prophet also provides some methods to plot the forecast and the components of the model, such as:

- plot: Plots the forecast, the observed values, and the uncertainty intervals
- plot_components: Plots the trend, seasonality, and holiday components of the model
- add_seasonality: Adds a custom seasonal component to the model
- add_country_holidays: Adds a built-in list of holidays for a specific country to the model

The results and insights

For this project, i chose to predict the RH concentration, which is one of the main indicators of air pollution. We used the data from March 2004 to January 2005 as the training set, and the data from February 2005 as the test set. We also added the Italian holidays to the model using the add_country_holidays method.

We fitted the model using the default parameters, except for the changepoint_prior_scale, which we set to 0.1 to make the model more flexible to the changes in the trend. We then made predictions for the next 28 days using the make_future_dataframe and predict methods.

The following plot shows the forecast and the observed values for the test set:

![plot]

We can see that the model captures the general trend and seasonality of the data, but also makes some errors, especially in the peaks and valleys. The uncertainty intervals are also quite wide, indicating a high degree of uncertainty in the predictions.

The following plot shows the components of the model:

![plot]

We can see that the model has a linear trend with some changepoints, a yearly seasonality with higher values in winter and lower values in summer, a weekly seasonality with higher values on weekdays and lower values on weekends, and a holiday effect with lower values on holidays.

We can also use the cross_validation and performance_metrics methods to evaluate the performance of the model on different time horizons. The following table shows the mean absolute error (MAE), the root mean squared error (RMSE), and the mean absolute percentage error (MAPE) for different horizons:

| horizon | MAE | RMSE | MAPE |
| ------- | --- | ---- | ---- |
| 1 days  | 36.9 | 50.9 | 0.23 |
| 2 days  | 37.9 | 51.9 | 0.24 |
| 3 days  | 38.8 | 53.0 | 0.24 |
| 4 days  | 39.7 | 54.0 | 0.25 |
| 5 days  | 40.6 | 55.1 | 0.25 |
| 6 days  | 41.5 | 56.1 | 0.26 |
| 7 days  | 42.4 | 57.2 | 0.26 |

We can see that the errors increase as the horizon increases, which is expected for a forecasting problem. The MAPE values indicate that the model has an average error of around 25% in predicting the NO2 concentration.

Conclusion

In this project , we introduced FB Prophet to predict the air quality in an Italian city. We described the data set, the model, and the results. We showed that the model can capture the trend and seasonality of the data, but also has some limitations and uncertainties. We also evaluated the performance of the model on different time horizons.

FB Prophet is a powerful and easy-to-use tool for forecasting time series data. It can handle various types of data and scenarios, and provide useful insights and visualizations. However, it is not a magic bullet, and it requires some tuning and validation to achieve the best results. We hope that this blog post has inspired you to try FB Prophet for your own forecasting problems.

References

: https://archive.ics.uci.edu/ml/datasets/Air+Quality
: https://facebook.github.io/prophet/

![image](https://github.com/Dzone11/Air-Quality-Prediction-using-FB-Prophet-Model/assets/90634803/751a9ed5-41b6-423a-97e8-c37d215f236f)
