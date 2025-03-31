
# Microsoft Stock Price Prediction

A brief description of what this project does and who it's for

Overview

This project aims to predict Microsoft stock prices using historical stock market data and deep learning techniques. The model is built using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) suitable for time series forecasting.
## Frameworks Used
Python

Pandas

NumPy

Matplotlib & Seaborn (for data visualization)

TensorFlow & Keras (for deep learning)

Scikit-learn (for data preprocessing)
## Exploratory Data Analysis (EDA)
EDA is performed to understand trends in the stock prices and trading volume.Steps include:

1. Plotting stock opening and closing prices over time

2. Analyzing trading volume trends

3. Generating a heatmap to study correlations between features
## Data Preprocessing
1. The date column is converted to datetime format.

2. Closing price values are extracted for model training.

3. Data is normalized using StandardScaler.

4. The dataset is split into training (95%) and testing (5%) sets.

5. Time-series batches of 60 days are created as input features for the LSTM model.
## Model Architecture
A sequential LSTM model is used with the following layers:

1. LSTM Layer (64 units, return sequences = True)

2. LSTM Layer (64 units)

3. Dense Layer (128 units)

4. Dropout Layer (0.5)

5. Dense Layer (1 unit, output layer)
## Model Training
1. Optimizer: Adam

2. Loss function: Mean Absolute Error (MAE)

3. Metric: Root Mean Squared Error (RMSE)
## Model Evaluation
1. The trained model is used to predict stock prices for the test dataset.

2. The predictions are compared with actual closing prices.

3. A visualization is provided to compare training, actual test prices, and predicted values.
## Results
1. The model successfully captures stock price trends with reasonable accuracy.

2. The prediction results are visualized using Matplotlib.
## Future Improvements
1. Fine-tune hyperparameters for better accuracy.

2. Experiment with different sequence lengths for input features.

3. Use additional technical indicators to enhance predictions.

4. Implement real-time stock price prediction.