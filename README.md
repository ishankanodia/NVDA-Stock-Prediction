### Project Title: Stock Price Prediction Using LSTM for NVIDIA Stock (NVDA)

### Objective
To predict future stock prices of NVIDIA (NVDA) using historical stock price data with an LSTM (Long Short-Term Memory) model. LSTMs are suitable for time-series forecasting due to their ability to capture long-term dependencies.

### Project Workflow

1. **Data Collection**:
   - Utilize the `yfinance` library to download historical stock price data for NVDA, covering dates from `2019-01-01` to `2024-09-30`.
   - Columns include `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

2. **Data Preprocessing**:
   - Load the data into a Pandas DataFrame and save it as a CSV file (`nvidia.csv`) for future use.
   - Select the `Low` price column for training the model, as it provides a simplified representation of stock price fluctuations while retaining essential information.
   - Normalize the data using `MinMaxScaler` from `sklearn.preprocessing` to ensure all values are between 0 and 1, which helps improve the LSTM model’s training stability and performance.

3. **Train-Test Split**:
   - Separate the dataset into training and testing sets:
     - First 1000 entries are used for training.
     - The remaining entries (445 data points) are reserved for testing.

4. **Dataset Preparation for LSTM**:
   - Create sequences of stock prices to use as input data for the LSTM model. The function `create_dataset`:
     - Takes the dataset and a specified timestep (in this case, 100).
     - Generates pairs of input sequences (`X`) and targets (`Y`) where each sequence is of length 100, and the target is the next price.
   - Shape `x_train`, `x_test`, `y_train`, and `y_test` to match the LSTM requirements:
     - Reshape inputs to 3D format: `[samples, time steps, features]`.

5. **LSTM Model Design**:
   - Construct a Sequential model with Keras consisting of:
     - **Three LSTM layers**:
       - First layer with 50 units, `return_sequences=True` to allow the next LSTM layer to process all time steps.
       - Second layer with 50 units, `return_sequences=True` for further feature extraction.
       - Final LSTM layer with 50 units, `return_sequences=False` to output a single vector for the next layer.
     - **Dense layer**: A fully connected layer with 1 neuron to predict the closing stock price.
   - Compile the model using `mean_squared_error` as the loss function and `adam` as the optimizer for efficient training.

6. **Model Training**:
   - Fit the model using the following parameters:
     - `epochs=100`: Number of times the model will see the entire training dataset.
     - `batch_size=64`: Number of samples per gradient update.
     - `validation_data=(x_test, y_test)`: Monitor validation performance during training.
   - Set `verbose=1` to display a progress bar and the loss for each epoch.

7. **Evaluation and Prediction**:
   - Once the model is trained, use it to make predictions on the test dataset and analyze its performance.
   - Plot the predictions against actual values to visually inspect the accuracy of the model and calculate metrics like RMSE (Root Mean Square Error) for quantitative evaluation.

### Additional Details

- **LSTM Layers with `return_sequences=True`**:
   - `return_sequences=True` is applied to intermediate LSTM layers to pass the full sequence output to the next LSTM layer, capturing temporal dependencies.
- **Train-Test Evaluation**:
   - The model will be evaluated on the test set, and predictions will be rescaled back to original price values using the inverse of the `MinMaxScaler`.

### Tools and Libraries Used
- **Data Analysis**: `Pandas`, `NumPy`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Data Scaling**: `MinMaxScaler` from `sklearn.preprocessing`
- **LSTM Modeling**: `Keras` (with `TensorFlow` as the backend)
- **Data Source**: `yfinance`

### Project Results
- **Prediction Accuracy**: The model’s effectiveness is demonstrated through the validation loss and the difference between predicted and actual stock prices.
- **Potential Improvements**:
   - Experiment with more hyperparameter tuning (e.g., learning rate, batch size, or number of LSTM units).
   - Include additional features like trading volume or other technical indicators.
- **Conclusion**: The project highlights the feasibility of using LSTM for time-series forecasting, specifically for stock prices. While the model may not perfectly predict future prices, it demonstrates a general trend-following capability that could be fine-tuned with more data and additional parameters.
