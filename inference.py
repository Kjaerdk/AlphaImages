import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

from model import CNN1, CNN2, CNN3
from lightning import LitUpDown
from plot_utils import plot_images, plot_roc_curve


# ----------------------------------------------------------------------------------------------------------------------
# Setup variables
# ----------------------------------------------------------------------------------------------------------------------

FREQ_number = '6'
FREQ_type = 'h'
FREQ = FREQ_number + FREQ_type
i = 8
h = 10
target_p_ahead = 2
layers = 1
img_col_name = 'I' + str(i) + 'H' + str(h) + 'P' + str(target_p_ahead)
vb = True
ma = True
target_type = 'direction'
version = 0

case = FREQ + '/I' + str(i) + '/H' + str(h) + '/P' + str(target_p_ahead) + '/layers' + str(layers) + '/vb' + str(int(vb)) + '_ma' + str(int(ma))
path_to_logs = os.path.join(os.path.join(os.path.expanduser('~')), 'path/to/model')
path_to_checkpoint = path_to_logs + case + f'/lightning_logs/version_{version}/checkpoints/'

# ----------------------------------------------------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------------------------------------------------
lit_mdl = LitUpDown(CNN1(i, h), l_rate=0.001).load_from_checkpoint(path_to_checkpoint + os.listdir(path_to_checkpoint)[0])
mdl = lit_mdl.model
mdl.eval()
hyps = lit_mdl.hparams
print(hyps['model'])

# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------
data = pd.read_pickle(hyps['data_path'])
data['date'] = data.datetime_open.dt.date
plot_images(data, hyps['target_col'], 5, 2)

train_data = data.loc[data.year.isin(hyps['train_years']), hyps['target_col']]
train_data = pd.DataFrame.from_dict(dict(zip(train_data.index, train_data.values))).transpose()\
                         .rename(columns={0: 'Image', 1: 'Direction'})
train_images = torch.tensor(np.stack(train_data['Image'].values)).float()  # (n_samples, n_channels, pixels_height, pixels_width)
train_labels = torch.tensor(train_data['Direction'].values.astype(float))  # (n_sample, )

val_data = data.loc[data.year.isin(hyps['val_years']), hyps['target_col']]
val_data = pd.DataFrame.from_dict(dict(zip(val_data.index, val_data.values))).transpose()\
                         .rename(columns={0: 'Image', 1: 'Direction'})
val_images = torch.tensor(np.stack(val_data['Image'].values)).float()  # (n_samples, n_channels, pixels_height, pixels_width)
val_labels = torch.tensor(val_data['Direction'].values.astype(float))  # (n_sample, )

test_data = data.loc[data.year.isin([2022]), hyps['target_col']]
test_data = pd.DataFrame.from_dict(dict(zip(test_data.index, test_data.values))).transpose()\
                         .rename(columns={0: 'Image', 1: 'Direction'})
test_images = torch.tensor(np.stack(test_data['Image'].values)).float()  # (n_samples, n_channels, pixels_height, pixels_width)
test_labels = torch.tensor(test_data['Direction'].values.astype(float))  # (n_sample, )

# ----------------------------------------------------------------------------------------------------------------------
# Get predictions from fitted model
# ----------------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    y_hat_train = mdl.forward(train_images).detach()
    y_hat_val = mdl.forward(val_images).detach()
    y_hat_test = mdl.forward(test_images).detach()


plt.hist(y_hat_train.numpy(), bins=100)
plt.hist(y_hat_val.numpy(), bins=100)
plt.hist(y_hat_test.numpy(), bins=100)

# ----------------------------------------------------------------------------------------------------------------------
# Get metrics
# ----------------------------------------------------------------------------------------------------------------------
