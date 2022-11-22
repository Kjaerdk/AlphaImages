import numpy as np
import pandas as pd
import datetime as dt
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from binance.client import Client

import config


class ImageDataset(Dataset):
    def __init__(self, data, n_periods, height_pixels, target_p_ahead):
        self.data = data
        self.img_col_name = 'I' + str(n_periods) + 'H' + str(height_pixels) + \
                            'P' + str(target_p_ahead)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data.loc[idx, self.img_col_name]
        ret = self.data.loc[idx, 'return']

        return image, label, ret


class ImageBase:
    def __init__(self, n_periods: int, height_pixels: int, target_price: str, target_p_ahead: int,
                 volume_bars=False, moving_average=False, volume_height_ratio=1/5):

        self.n_periods = n_periods
        self.height_pixels = height_pixels

        self.target_price = target_price
        self.target_p_ahead = target_p_ahead
        self.target_price_pred = str(target_price) + str(target_p_ahead)  # close10 (for example)

        self.img_col_name = 'I' + str(n_periods) + 'H' + str(height_pixels) + \
                            'P' + str(target_p_ahead)

        self.vb = volume_bars
        self.ma = moving_average

        if volume_bars:
            self.volume_height_pixels = int(height_pixels * volume_height_ratio)
            self.candle_height_pixels = int(height_pixels * (1 - volume_height_ratio))
        else:
            self.candle_height_pixels = height_pixels

        if moving_average:
            self.moving_average_length = n_periods

    def candle_maker(self, raw_candles):
        """
        Take the inputted OHLC prices, rescale them to fit dimensions and get which for the candle.
        Args:
            raw_candles: OHLC prices for a period and the (n_periods-1) last periods.

        Returns: Pixels for candles
        """
        min_tmp = np.min(raw_candles)
        max_tmp = np.max(raw_candles)

        if min_tmp == max_tmp:
            return [], True

        normalized_candles = (raw_candles - min_tmp) / (max_tmp - min_tmp)

        scaled_candles = np.round(normalized_candles * (self.candle_height_pixels - 1)).astype(int)  # subtract one since normalize_candles \in [0, 1] which would give one to many pixels
        # Highest value is (candle_height_pixels - 1) but highest index in image is 0 (start from top).
        # Thus, flip pixels to run from 0 (highest position) to (candle_height_pixels - 1) (lowest position)
        scaled_candles = (self.candle_height_pixels - 1) - scaled_candles

        return scaled_candles, False

    def place_candles(self, candles):
        """
        Take candle pixels and place them in candle image.
        Args:
            candles:

        Returns: array with dim (n_periods x 3) x candle_height_pixels
        """
        # Initialize the image for candles to be zeros
        candle_image = np.zeros((1, self.candle_height_pixels, self.n_periods * 3))  # 1 for number of channels

        for period in range(self.n_periods):
            # Set open pixel
            open_col = period * 3
            candle_image[:, candles[period, 0], open_col] = 1
            # Set high-low pixels
            high_low_bar_col = (period * 3) + 1
            candle_image[:, candles[period, 1]:(candles[period, 2] + 1), high_low_bar_col] = 1
            # Set close pixel
            close_col = (period * 3) + 2
            candle_image[:, candles[period, 3], close_col] = 1

        candle_image = candle_image.astype(np.double)

        return candle_image

    def place_moving_averages(self, candle_image, prices):
        """
        Take candle image and ensure pixels with moving average is 1
        - prices are high, low and moving averages
        """
        min_tmp = np.min(prices[['high', 'low']].values)
        max_tmp = np.max(prices[['high', 'low']].values)

        normalized_ma = (prices['ma' + str(self.n_periods)].values - min_tmp) / (max_tmp - min_tmp)
        normalized_ma = np.clip(normalized_ma, 0, 1)  # ensure in [0, 1]: could be above if prices before higher/lower

        scaled_ma = np.round(normalized_ma * (self.candle_height_pixels - 1)).astype(int)  # subtract one since normalize_candles \in [0, 1] which would give one to many pixels
        scaled_ma = (self.candle_height_pixels - 1) - scaled_ma

        for period in range(self.n_periods):
            # First
            if period == 0:
                candle_image[:, scaled_ma[period], period] = 1  # should ideally take into account previous close not shown in image
            else:
                pixel_height = int(np.round(scaled_ma[period - 1] * (1/3) + scaled_ma[period] * (2/3), 0))
                candle_image[:, pixel_height, period * 3] = 1
            # Second
            candle_image[:, scaled_ma[period], period * 3 + 1] = 1
            # Third
            if (period + 1) == self.n_periods:  # the last columns
                candle_image[:, scaled_ma[period], period * 3 + 2] = 1
            else:
                pixel_height = int(np.round(scaled_ma[period] * (2/3) + scaled_ma[period + 1] * (1/3), 0))
                candle_image[:, pixel_height, period * 3 + 2] = 1

        return candle_image

    def place_volume_bars(self, volumes):
        """
        Take volumes and return image with scaled volume bars
        """
        # Initialize the image for candels to be zeros
        volume_image = np.zeros((1, self.volume_height_pixels, self.n_periods * 3))  # 1 for number of channels

        max_tmp = np.max(volumes)
        normalized_vol = volumes / max_tmp
        scaled_vol = np.round(normalized_vol * (self.volume_height_pixels - 1)).astype(int)
        scaled_vol = (self.volume_height_pixels - 1) - scaled_vol

        for period in range(self.n_periods):
            volume_image[:, scaled_vol[period]:self.volume_height_pixels, period * 3 + 1] = 1

        return volume_image


class MakeImages(ImageBase):
    def __init__(self, data, n_periods: int, height_pixels: int, target_price: str, target_p_ahead: int,
                 volume_bars=False, moving_average=False, volume_height_ratio=1/5):
        super().__init__(n_periods=n_periods, height_pixels=height_pixels, target_price=target_price,
                         target_p_ahead=target_p_ahead, volume_bars=volume_bars,
                         moving_average=moving_average, volume_height_ratio=volume_height_ratio)

        self.data = data
        # need to ensure ascending dates when taking last n_periods and shift target price
        self.data = self.data.sort_values('datetime_open').reset_index(drop=True)

        self.data[self.img_col_name] = -1
        self.data['remove_row'] = False

        if not self.data.columns.isin(['ma' + str(self.n_periods)]).any():  # make ma if not in cols already
            self.data['ma' + str(n_periods)] = self.data[target_price].rolling(n_periods).mean()
            self.data = self.data[~self.data['ma' + str(n_periods)].isna()]
            self.data.reset_index(drop=True, inplace=True)

    def make_images(self):
        """
        Returns: An image for each date n_periods back with height candle_height_pixels
        """

        for t in range(self.n_periods, self.data.shape[0] + 1):
            tmp_data = self.data.iloc[(t - self.n_periods):t]
            candles, skip = self.candle_maker(tmp_data.loc[:, ['open', 'high', 'low', 'close']].values)
            if skip:  # not possible to make candles then remove row
                self.data.loc[t - 1, 'remove_row'] = True
                self.data.loc[t - 1, self.img_col_name] = [np.nan]
                continue
            #  ...
            #  day -1  open, high, low, close
            #  day 0   open, high, low, close
            candle_image = self.place_candles(candles)
            if self.ma and not skip:
                candle_image = self.place_moving_averages(candle_image, tmp_data.loc[:, ['high', 'low', 'ma' + str(self.n_periods)]])

            if self.vb and not skip:
                volume_image = self.place_volume_bars(tmp_data.loc[:, 'volume'].values)
                self.data.loc[t - 1, self.img_col_name] = [np.hstack([candle_image, volume_image])]
            elif not skip:
                self.data.loc[t - 1, self.img_col_name] = [candle_image]

    def make_targets(self, target_type, add_target_to_image=True):
        """
        Returns: Appends target to list with image as first entry and target as second entry.
        """
        # Align image with target
        self.data[self.target_price_pred] = self.data[str(self.target_price)].shift(-self.target_p_ahead)
        # Get if target is up or down and the price diff
        self.data['direction'] = (self.data[self.target_price_pred] > self.data[self.target_price]).astype(float)
        self.data['price_diff'] = self.data[self.target_price_pred] - self.data[self.target_price]
        self.data['return'] = (self.data[self.target_price_pred] - self.data[self.target_price]) / self.data[self.target_price]
        if target_type == "direction":
            self.data['target'] = self.data['direction']
        else:
            self.data['target'] = self.data['return'] > target_type

        if add_target_to_image:
            for i in range(self.n_periods, self.data.shape[0] + 1):
                self.data.loc[i - 1, self.img_col_name].append(self.data.loc[i - 1, 'target'])

    def make_images_and_targets(self, target_type='direction', return_dataset_class=False):
        """
        - target_type takes values "direction" or a float indication the return required for positive (a 1)
        Returns: A Dataset class with prepared images and targets.
        """
        self.make_images()
        self.make_targets(target_type)

        # Only remove rows after making targets to not remove needed prices
        self.data = self.data[~(self.data['remove_row'])]
        self.data = self.data[~(self.data[self.img_col_name] == -1)]  # remove if not enough prior periods
        self.data = self.data[~self.data[self.target_price_pred].isna()]  # remove if not enough future periods
        self.data = self.data[~(np.abs(self.data['return']) > 1)]  # remove if price move more than 100% (data error)

        self.data = self.data.reset_index(drop=True)  # ensure index starts a 0 and runs to len(data)

        if return_dataset_class:
            return ImageDataset(self.data, self.n_periods, self.candle_height_pixels)

    def plot_images(self, n_images_col, n_images_row):

        labels_map = {0: "Down", 1: "Up"}

        figure = plt.figure(figsize=(15, 15))
        cols, rows = n_images_col, n_images_row
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.data), size=(1,)).item()
            img, label = self.data.loc[sample_idx, self.img_col_name]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label] + str(self.data.loc[sample_idx, 'date']))
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()


class MakeImage(ImageBase):
    def __init__(self, n_periods: int, height_pixels: int, target_price: str, target_p_ahead: int,
                 volume_bars=False, moving_average=False, volume_height_ratio=1/5):
        super().__init__(n_periods=n_periods, height_pixels=height_pixels, target_price=target_price,
                         target_p_ahead=target_p_ahead, volume_bars=volume_bars,
                         moving_average=moving_average, volume_height_ratio=volume_height_ratio)

    def make_image(self, raw_candles):
        """
        Input: pandas with n_periods obs with cols ['open', 'high', 'low', 'close', 'ma(p)', 'volume']
        Returns: An image with height candle_height_pixels
        """

        candles, skip = self.candle_maker(raw_candles.loc[:, ['open', 'high', 'low', 'close']].values)

        if skip:  # not possible to make candles then remove row
            return None
            #  ...
            #  day -1  open, high, low, close
            #  day 0   open, high, low, close
        candle_image = self.place_candles(candles)

        if self.ma and not skip:
            candle_image = self.place_moving_averages(candle_image, raw_candles.loc[:, ['high', 'low', 'ma' + str(self.n_periods)]])

        if self.vb and not skip:
            volume_image = self.place_volume_bars(raw_candles.loc[:, 'volume'].values)
            return [np.hstack([candle_image, volume_image])]
        elif not skip:
            return [candle_image]


def fetch_api_candle_images(setup_dict: dict, return_data=True):
    """
    Args:
        setup_dict: Dict with parameters for image to be made
        return_data: Flag to determine if return newly made df

    Returns: Save a new df with candle images and target for the specified parameters
    """

    start_int = int(dt.datetime.timestamp(dt.datetime(year=setup_dict['years'][0] - 1, month=12, day=31, hour=23))) * 1000

    interval = {'1m': Client.KLINE_INTERVAL_1MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE, '1h': Client.KLINE_INTERVAL_1HOUR, '6h': Client.KLINE_INTERVAL_6HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR, '1d': Client.KLINE_INTERVAL_1DAY}

    # Load data
    client = Client(api_key=config.API_KEY, api_secret=config.API_SECRET, requests_params={'timeout': 120})
    klines = client.get_historical_klines(symbol="symbol", interval=interval[setup_dict['freq']], start_str=start_int)

    # Preprocess the data
    cols = ['timestamp_open', 'open', 'high', 'low', 'close', 'volume', 'timestamp_close', 'quote_asset_volume',
                    'n_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df = pd.DataFrame(klines, columns=cols)
    df = df[['timestamp_open', 'open', 'high', 'low', 'close', 'volume', 'timestamp_close']].astype(float)
    df['datetime_open'] = [dt.datetime.fromtimestamp(x / 1000) for x in df.timestamp_open]
    df['datetime_close'] = [dt.datetime.fromtimestamp(x / 1000) for x in df.timestamp_close]
    df['datetime_close'] = [dt.datetime.fromtimestamp(x / 1000) for x in df.timestamp_close]
    df['year'] = df['datetime_open'].dt.year
    imagine = MakeImages(df, setup_dict['n_periods'], setup_dict['height_pixels'], setup_dict['target_price'],
                         setup_dict['target_p_ahead'], setup_dict['vb'], setup_dict['ma'], 1/5)

    imagine.make_images_and_targets(setup_dict['target_type'])

    # Save data
    imagine.data.to_pickle(setup_dict['data_path'])

    if return_data:
        return imagine.data


def make_targets(data, target_price_pred, target_price, target_type, target_p_ahead):
    """
    Returns: Data with price and target columns
    """
    # Align image with target
    data[target_price_pred] = data[str(target_price)].shift(-target_p_ahead)
    # Get if target is up or down and the price diff
    data['direction'] = (data[target_price_pred] > data[target_price]).astype(float)
    data['price_diff'] = data[target_price_pred] - data[target_price]
    data['return'] = data['price_diff'] / data[target_price]
    if target_type == "direction":
        data['target'] = data['direction']
    else:
        data['target'] = data['return'] > target_type

    return data
