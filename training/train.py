from share import *

import pytorch_lightning as pl
import cv2
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import UnconditionalDataset, train_dataset, test_dataset, val_dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = '../models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
num_epochs = 5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('../models/cldm_v15.yaml').cpu()
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# # Define the checkpoint callback
# checkpoint_callback = ModelCheckpoint(
#     monitor='train_loss',  # Monitor validation loss
#     filename='epoch_{epoch:02d}',  # Filename format
#     dirpath='./models',
#     save_top_k=-1,  # Save all checkpoints
#     every_n_epochs=1,  # Save every 10 epochs
# )

# Misc
dataset = UnconditionalDataset()
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

for epoch in range(num_epochs):
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs = 1, default_root_dir="./models")
    trainer.fit(model, train_dataloader)
    checkpoint_path = f'./models/model_weights_{epoch}.pth'
    torch.save(model.state_dict(), checkpoint_path)