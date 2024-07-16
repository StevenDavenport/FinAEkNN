import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class DataPreparation:
    def __init__(self, scaler=MinMaxScaler(feature_range=(0, 1))):
        self.scaler = scaler

    def fetch_financial_data(self, ticker, start_date, end_date):
        """Fetches historical financial data for a given ticker from start_date to end_date."""
        data = yf.download(ticker, start=start_date, end=end_date)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]

    def create_sequences(self, df, sequence_length=10):
        sequences = []
        for i in range(len(df) - sequence_length + 1):
            seq = df.iloc[i:i + sequence_length].values
            sequences.append(seq)
        return np.array(sequences)

    def add_technical_indicators(self, data):
        """Adds technical indicators as new columns to the DataFrame."""
        data['feature_MA'] = self.moving_average(data)
        data['feature_RSI'] = self.rsi(data)
        macd, signal_line = self.macd(data)
        data['feature_MACD'] = macd
        data['feature_MACD_signal'] = signal_line
        upper_band, lower_band = self.bollinger_bands(data)
        data['feature_Upper_BB'] = upper_band
        data['feature_Lower_BB'] = lower_band
        return data

    def preprocess_data(self, data):
        """Applies necessary preprocessing steps to the data, including normalization."""
        data_with_indicators = self.add_technical_indicators(data)
        data_with_indicators.bfill(inplace=True)
        data_with_indicators.ffill(inplace=True)
        data_scaled = pd.DataFrame(self.scaler.fit_transform(data_with_indicators), columns=data_with_indicators.columns, index=data_with_indicators.index)
        return data_scaled

    def moving_average(self, data, period=10, column='Close'):
        try :
            return data[column].rolling(window=period).mean()
        except:
            pass
        finally:
            return data[column.lower()].rolling(window=period).mean()

    def rsi(self, data, period=14, column='Close'):
        try:
            delta = data[column].diff(1)
        except:
            pass
        finally:
            delta = data[column.lower()].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    def macd(self, data, column='Close', slow=26, fast=12, signal=9):
        try:
            exp1 = data[column].ewm(span=fast, adjust=False).mean()
            exp2 = data[column].ewm(span=slow, adjust=False).mean()
        except:
            pass
        finally:
            exp1 = data[column.lower()].ewm(span=fast, adjust=False).mean()
            exp2 = data[column.lower()].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def bollinger_bands(self, data, period=20, column='Close'):
        try:
            sma = data[column].rolling(window=period).mean()
            std = data[column].rolling(window=period).std()
        except:
            pass
        finally:
            sma = data[column.lower()].rolling(window=period).mean()
            std = data[column.lower()].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band






