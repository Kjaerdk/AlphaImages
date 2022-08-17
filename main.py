import argparse
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data_utils import ImageDataset
from lightning import LitUpDown
from model import CNN1, CNN2, CNN3

"""
Train model with specified parameters.
"""


def main(args):

    dict_args = vars(args)

    # Setup variables
    dict_args['freq'] = str(dict_args['freq_number']) + dict_args['freq_type']

    # Load data and clean
    path = '/path/to/data/'
    data_path = dict_args['freq'] + '/' + dict_args['freq'] + '_I' + str(dict_args['n_periods']) +\
                'H' + str(dict_args['height_pixels']) + 'P' + str(dict_args['target_periods_ahead']) + \
                '_vb' + str(int(dict_args['vb'])) + '_ma' + str(int(dict_args['ma'])) + '_source.pkl'

    print(data_path)

    dict_args['data_path'] = path + data_path

    data = pd.read_pickle(dict_args['data_path'])

    dict_args['train_years'] = [2019, 2020]
    dict_args['val_years'] = [2021]

    train_dataset = ImageDataset(data[data.year.isin(dict_args['train_years'])].reset_index(drop=True),
                                 dict_args['n_periods'], dict_args['height_pixels'], dict_args['target_periods_ahead'])
    dict_args['target_col'] = train_dataset.img_col_name

    train_dataloader = DataLoader(train_dataset, batch_size=dict_args['batch_size'], shuffle=True)

    val_dataset = ImageDataset(data[data.year.isin(dict_args['val_years'])].reset_index(drop=True),
                               dict_args['n_periods'], dict_args['height_pixels'], dict_args['target_periods_ahead'])
    val_dataloader = DataLoader(val_dataset, batch_size=dict_args['batch_size'], shuffle=True)

    # Variables for easy save results
    dict_args['log_path'] = dict_args['freq'] + '/I' + str(dict_args['n_periods']) + '/H' + str(dict_args['height_pixels']) + \
                            '/P' + str(dict_args['target_periods_ahead']) + '/layers' + str(dict_args['layers']) + \
                            '/vb' + str(int(dict_args['vb'])) + '_ma' + str(int(dict_args['ma']))
    print(dict_args)

    # Define model
    # Define model
    if dict_args['layers'] == 1:
        mdl = CNN1(dict_args['n_periods'], dict_args['height_pixels'])
    elif dict_args['layers'] == 2:
        mdl = CNN2(dict_args['n_periods'], dict_args['height_pixels'])
    elif dict_args['layers'] == 3:
        mdl = CNN3(dict_args['n_periods'], dict_args['height_pixels'])

    litmdl = LitUpDown(mdl, **dict_args)

    # Train and validate
    trainer = pl.Trainer(default_root_dir='/path/to/logs/' + dict_args['log_path'],
                         max_epochs=dict_args['epochs'], log_every_n_steps=5)
    trainer.fit(litmdl, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--l_rate', type=float, default=0.001)
    parser.add_argument('--layers', type=int, default=1)

    parser.add_argument('--freq_number', type=int, default=15)
    parser.add_argument('--freq_type', type=str, default='m')

    parser.add_argument('--n_periods', type=int, default=8)
    parser.add_argument('--height_pixels', type=int, default=40)
    parser.add_argument('--vb', type=bool, default=True)
    parser.add_argument('--ma', type=bool, default=True)
    parser.add_argument('--target_periods_ahead', type=int, default=4)

    args = parser.parse_args()

    main(args)
