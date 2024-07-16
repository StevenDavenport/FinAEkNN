import pandas as pd
import datetime
import numpy as np
import time

from gym_trading_env.environments import TradingEnv
from gym_trading_env.downloader import download
import gymnasium as gym

from data import DataPreparation

# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
download(exchange_names = ["binance"],
    symbols= ["BTC/USDT"],
    timeframe= "1h",
    dir = "data",
    since= datetime.datetime(year= 2020, month= 1, day=1),
)
# Import your fresh data
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
print(df.head())

# Create features
data_prep = DataPreparation()
df = data_prep.add_technical_indicators(df)
# drop rows with NaN values
df.dropna(inplace=True)
df = df.reset_index()
df['date_open'] = df['date_open'].astype(np.int64) // 10**9
print(df.head())
print('Shape of the data:', df.shape)
print('Columns:', df.columns)
# drop feature_data_close
#df.drop(columns=['feature_date_close'], inplace=True)
print(df.dtypes)

# Create your own reward function with the history object
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1

# Create a TradingEnv object
env = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df = df,
        windows= 5,
        positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 'random', #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration = 500,
    )

env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )

done, truncated = False, False
observation, info = env.reset()
print(info)
while not done and not truncated:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(observation)
# Save for render
# env.save_for_render()
