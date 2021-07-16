import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import distributions as D
from argparse import ArgumentParser
from nolatent_fixedvar import *

PATH = "/data/math-multicellular-struct-devel/hert6124/mechanistic_mRNAi/NN_models/lightning_logs/version_1632011/checkpoints/last.ckpt" #direct model, no latent variables

L = np.array([0,0.1])

model = ConditionalAutoEncoder.load_from_checkpoint(PATH,schedule=L,learning_rate=1e-3)
dataset = MechanisticModelDataset(csv_file="mock_training_data.csv", root_dir="/data/math-multicellular-struct-devel/hert6124/pakman-develop/examples/DeepScratch/datadir3")


loaded_data = DataLoader(dataset, 2, False)

predictions = np.zeros((500 ,6))
truepar = np.zeros((500 ,6))

for (idx, batch) in enumerate(loaded_data):
    if(idx <250):
        print(idx)
        means, stds = model.forward(batch)
        means = means.detach().numpy()
        stds = stds.detach().numpy()
        predictions[2 * idx] = means[0]
        predictions[2 * idx + 1] = means[1]

        print(batch["par"])
        truepar[2 * idx] = batch["par"][0]
        truepar[2 * idx + 1] = batch["par"][1]

    if(idx == 250):
        break

np.savetxt("predictions_direct.txt", predictions)
np.savetxt("truepar_direct.txt", truepar)