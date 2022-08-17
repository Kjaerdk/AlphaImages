"""
From minute data to a lower granularity
"""

import pandas as pd
import datetime as dt

FREQ_number = '15'
FREQ_type = 'm'
FREQ = FREQ_number + FREQ_type
# Load data and clean
path = '/path/to/data/'
data = pd.read_csv(path + '1m/data_file.csv', sep=',', header=0)

data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                     'volume': 'volume'}, inplace=True)
data['date'] = pd.to_datetime(data.Timestamp, unit='s')
data['year'] = data.date.dt.year
data = data[data.year.isin(list(range(2015, 2022)))]

# Get dates for new candle frame
dt_start = data.date.min().round(freq='D') - dt.timedelta(days=1)  # ensure to cover all dates
dt_stop = data.date.max().round(freq='D') + dt.timedelta(days=1)
dt_range = pd.date_range(dt_start, dt_stop, freq=FREQ)

# Initialize new df and make date columns
new_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
new_df['date'] = dt_range
new_df['year'] = new_df.date.dt.year
new_df = pd.merge(new_df, data[['Timestamp', 'date']], how='left', on='date')

# Fill remaining columns (prices, volume)
for t in new_df.date:
    print(t)
    if FREQ_type == 'h':
        tmp_range = pd.date_range(t, t + dt.timedelta(hours=int(FREQ_number)) - dt.timedelta(minutes=1), freq='1min')
    elif FREQ_type == 'd':
        tmp_range = pd.date_range(t, t + dt.timedelta(days=int(FREQ_number)) - dt.timedelta(minutes=1), freq='1min')
    elif FREQ_type == 'm':
        tmp_range = pd.date_range(t, t + dt.timedelta(minutes=int(FREQ_number)) - dt.timedelta(minutes=1), freq='1min')

    tmp_data = data[data.date.isin(tmp_range)]

    if tmp_data.empty or tmp_data[['open', 'high', 'low', 'close', 'volume']].isna().all().all():
        continue

    new_df.loc[new_df.date == t, 'open'] = tmp_data.loc[~(tmp_data.open.isna()), 'open'].iloc[0]
    new_df.loc[new_df.date == t, 'close'] = tmp_data.loc[~(tmp_data.close.isna()), 'close'].iloc[-1]
    new_df.loc[new_df.date == t, 'low'] = tmp_data[['open', 'high', 'low', 'close']].min().min()
    new_df.loc[new_df.date == t, 'high'] = tmp_data[['open', 'high', 'low', 'close']].max().max()

    new_df.loc[new_df.date == t, 'volume'] = tmp_data['volume'].sum()

new_df[['open', 'high', 'low', 'close', 'volume']] = new_df[['open', 'high', 'low', 'close', 'volume']].astype(float)  # convert from object to float
new_df.to_pickle(path + FREQ + '/source_' + FREQ + '_data_dates.pkl')
