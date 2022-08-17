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

        ##

        return scaled_candles, False

    def place_candles(self, candles):
        """
        Take candle pixels and place them in candle image.
        Args:
            candles:

        Returns: array with dim (n_periods x 3) x candle_height_pixels
        """

        ##

        return candle_image

    def place_moving_averages(self, candle_image, prices):
        """
        Take candle image and place moving averages in candle image.
        - prices are high, low and moving averages
        """

        ##

        return candle_image

    def place_volume_bars(self, volumes):
        """
        Take volumes and return image with scaled volume bars
        """

        ##

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

        ##

    def make_targets(self, target_type, add_target_to_image=True):
        """
        Returns: Appends target to list with image as first entry and target as second entry.
        """

        ##

    def make_images_and_targets(self, target_type='direction', return_dataset_class=False):
        self.make_images()
        self.make_targets(target_type)

        """
        Data checks
        """

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

        ##

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

    ##

    return data
