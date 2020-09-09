# web_traffic
Web traffic challenge on kaggle.com
(https://www.kaggle.com/c/web-traffic-time-series-forecasting)

Ranked 100 over 400 participant
(SMAPE 41.96493)

# Main
2 main files main_competition.py and main_competition_prophet.py

main_competition uses a DNN to predict
main_competition_prophet uses prophet to predict

# preprocess
The data preprocess has been done in the pre_process_competition.py script

# Classes
lib_model_regression.py contain the classes for all the models used

# sub sample
on the spanish_example_ARIMA.py and spanish_example_ML.py a subset of the data are used.
ARIMA (Autoregressive Integrated Moving Average) and standard DNN regression are tested.
