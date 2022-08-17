import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_utils import fetch_api_candle_images, ImageDataset
from lightning import LitUpDown
from model import CNN1, CNN2, CNN3

import itertools

"""
Train model for each combination of parameters. Fetch and create new data if not already downloaded.
"""

B_SIZE = 1000
L_RATE = 0.001

possible_layers = [1, 2, 3]
possible_n_periods = [4, 8, 12]
possible_height_pixels = [10, 25, 50]
possible_target_p_ahead = [2, 4, 8]
possible_freq = ['5m', '15m', '30m', '1h', '6h', '12h', '1d']
target_type = 'direction'  # could be altered
target_price = 'close'

combs = list(itertools.product(possible_layers, possible_n_periods, possible_height_pixels,
                               possible_target_p_ahead, possible_freq))
random.shuffle(combs)

for i, c in enumerate(combs): # yes, it takes many resources
    print(f'{np.round(i / len(combs) * 100, 2)}% done - {len(combs) - i} to go')
    # Optimize model, image dimensions and granularity
    layers = c[0]
    n_periods = c[1]
    height_pixels = c[2]
    target_p_ahead = c[3]
    target_price_pred = target_price + str(target_p_ahead)
    freq = c[4]

    layers = layers + 1 if freq in ['5m', '15m', '30m'] else layers
    height_pixels = height_pixels * 2 if layers == 3 else height_pixels

    epochs = 75 if freq in ['15m', '30m', '1h', '6h', '12h', '1d'] else 20
    setup_dict = dict(batch_size=B_SIZE, epochs=epochs, layers=layers, l_rate=L_RATE,  # training and model params
                      n_periods=n_periods, height_pixels=height_pixels, target_p_ahead=target_p_ahead,  # image params
                      vb=True, ma=True, freq=freq, freq_number=freq[:-1], freq_type=freq[-1:],
                      target_type=target_type, target_price=target_price,  # target and data params
                      target_price_pred=target_price_pred, years=list(range(2020, 2023)))

    # Path variables
    path_to_desktop_folder = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop', 'data')
    data_path = setup_dict['freq'] + '/' + setup_dict['freq'] + '_I' + str(setup_dict['n_periods']) + 'H' + \
                str(setup_dict['height_pixels']) + 'P' + str(setup_dict['target_p_ahead']) + '_vb' + \
                str(int(setup_dict['vb'])) + '_ma' + str(int(setup_dict['ma'])) + '_' + str(setup_dict['years'][0]) \
                + '_' + str(setup_dict['years'][-1]) + '_' + str(target_type) + '_source.pkl'

    print(data_path)
    setup_dict['data_path'] = path_to_desktop_folder + '/' + data_path

    if not os.path.exists(setup_dict['data_path']):  # check whether the specified path exists or not
        data = fetch_api_candle_images(setup_dict)
    else:
        data = pd.read_pickle(setup_dict['data_path'])

    setup_dict['train_years'] = [2020, 2021]
    setup_dict['val_years'] = [2022]

    train_dataset = ImageDataset(data[data.year.isin(setup_dict['train_years'])].reset_index(drop=True),
                                 setup_dict['n_periods'], setup_dict['height_pixels'], setup_dict['target_p_ahead'])
    setup_dict['target_col'] = train_dataset.img_col_name
    train_dataloader = DataLoader(train_dataset, batch_size=setup_dict['batch_size'], shuffle=True)

    val_dataset = ImageDataset(data[data.year.isin(setup_dict['val_years'])].reset_index(drop=True),
                               setup_dict['n_periods'], setup_dict['height_pixels'], setup_dict['target_p_ahead'])
    val_dataloader = DataLoader(val_dataset, batch_size=setup_dict['batch_size'], shuffle=True)

    # Variable for easy save results
    setup_dict['log_path'] = 'exhaustive/' + setup_dict['freq'] + '/I' + str(setup_dict['n_periods']) + '/H' + \
                             str(setup_dict['height_pixels']) + '/P' + str(setup_dict['target_p_ahead']) + '/layers' + \
                             str(setup_dict['layers']) + '/vb' + str(int(setup_dict['vb'])) + '_ma' + str(int(setup_dict['ma']))
    print(setup_dict)

    # Define model
    if setup_dict['layers'] == 1:
        mdl = CNN1(setup_dict['n_periods'], setup_dict['height_pixels'])
    elif setup_dict['layers'] == 2:
        mdl = CNN2(setup_dict['n_periods'], setup_dict['height_pixels'])
    elif setup_dict['layers'] == 3:
        mdl = CNN3(setup_dict['n_periods'], setup_dict['height_pixels'])

    litmdl = LitUpDown(mdl, **setup_dict)

    # Train and validate
    trainer = pl.Trainer(default_root_dir=os.getcwd() + '/logs/' + setup_dict['log_path'],
                         max_epochs=setup_dict['epochs'], log_every_n_steps=5)
    trainer.logger.log_hyperparams(setup_dict)
    trainer.fit(litmdl, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

