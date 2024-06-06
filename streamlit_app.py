import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
st.balloons()
st.markdown("# Stock Price Prediction")

st.markdown("## Price Prediction")

stock_name = st.text_input(label="Stock Name", value="MSFT")

df = yf.download(stock_name, start="2000-01-01", end="2023-12-31")
# Create sequences
from copy import deepcopy as dc

def prepare_dataframe(df, n_steps):
    df = dc(df)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    columns = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
    df.drop(columns=columns, inplace=True)
    X, y = df.drop('Close', axis=1), df['Close']
    return X, y
n_steps = 15
X,y = prepare_dataframe(df, n_steps)



st.write(df)

st.title(stock_name)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y.index, y)
st.pyplot(fig)

st.write("Here we are at the end of getting started with streamlit! Happy Streamlit-ing! :balloon:")

data = yf.download(stock_name,start = "2000-01-01",end="2023-12-31")

data = data[['Close']]

scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
def predict_ensemble(models, test_loader):
    y_pred_ensemble = []
    for seq, _ in test_loader:
        seq = seq.to('cpu')
        preds = []
        for model in models:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to('cpu'),
                                 torch.zeros(1, 1, model.hidden_layer_size).to('cpu'))
            with torch.no_grad():
                preds.append(model(seq).item())
        y_pred_ensemble.append(np.mean(preds))
    return y_pred_ensemble

SEQUENCE_SIZE = 15

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)

    return np.array(x), np.array(y)

X, y = to_sequences(SEQUENCE_SIZE, data['Close'].values)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = None

    def forward(self, input_seq):
        self.hidden_cell
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

new_data = yf.download(stock_name, start='2000-01-01', end='2023-12-31')
new_data = new_data[['Close']]
new_data['Close'] = scaler.transform(new_data['Close'].values.reshape(-1, 1))

X_new, y_new = to_sequences(SEQUENCE_SIZE, new_data['Close'].values)
X_new = X_new.reshape(-1, SEQUENCE_SIZE, 1)

test_dataset_new = TimeSeriesDataset(X_new, y_new)
test_loader_new = DataLoader(test_dataset_new, batch_size=1, shuffle=False)

num_models = 5
models_loaded = []
for i in range(num_models):
    model = LSTM(input_size=1)
    model.load_state_dict(torch.load(f'/workspaces/ml18/Trained_models/RNN_LSTM/LSTM/lstm_model_{i}.pth',
                                     map_location=torch.device('cpu')))
    models_loaded.append(model)

y_pred_ensemble_new = predict_ensemble(models_loaded, test_loader_new)

y_test_new_inverse = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_new), 3)), y_new.reshape(-1, 1))))[:, 3]
y_pred_ensemble_new_inverse = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_pred_ensemble_new), 3)), np.array(y_pred_ensemble_new).reshape(-1, 1))))[:, 3]

rmse_new = np.sqrt(mean_squared_error(y_test_new_inverse, y_pred_ensemble_new_inverse))
r_squared_new = r2_score(y_test_new_inverse, y_pred_ensemble_new_inverse)
st.write(f'New RMSE: {rmse_new}')
st.write(f'New R-squared: {r_squared_new}')

def plot_predictions(y_test, y_pred_ensemble, stock_name):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(y_test, label='Actual')
    ax.plot(y_pred_ensemble, label='Predicted')
    ax.legend()
    ax.set_title(f'Ensemble LSTM Predictions vs Actual for {stock_name}')

    return fig

# Call the function to get the plot as a figure object
fig = plot_predictions(y_test_new_inverse, y_pred_ensemble_new_inverse, stock_name)

#$ Display the plot using st.pyplot()
st.pyplot(fig)