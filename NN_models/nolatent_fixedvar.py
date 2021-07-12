import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import distributions as D
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint


class MechanisticModelDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the raw data.
        """

        self.ID_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.ID_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        IC_file = os.path.join(self.root_dir,
                               self.ID_frame.iloc[idx, 0])
        FC_file = os.path.join(self.root_dir,
                               self.ID_frame.iloc[idx, 1])
        par_file = os.path.join(self.root_dir,
                                self.ID_frame.iloc[idx, 2])

        IC = torch.from_numpy(np.genfromtxt(IC_file, delimiter=","))
        FC = torch.from_numpy(np.genfromtxt(FC_file, delimiter=","))
        par = torch.from_numpy(np.genfromtxt(par_file, delimiter=","))

        IC = IC.unsqueeze(dim=0)
        FC = FC.unsqueeze(dim=0)
        sample = {'IC': IC, 'FC': FC, 'par': par}

        return sample


class ConditionalAutoEncoder(pl.LightningModule):

    def __init__(self, schedule, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.schedule = torch.from_numpy(schedule)

        self.data_process = nn.Sequential(
            nn.Linear(3840, 3840),
            nn.ReLU(),
            nn.Linear(3840, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self.mu = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Sigmoid()
        )

        self.sigma = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

        self.flat_data = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=(3, 4)),
            nn.Flatten(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, batch):
        data_in = batch['IC']
        data_fin = batch['FC']
        theta = batch['par'].unsqueeze(dim=1)

        data_in_flat = self.flat_data(data_in).squeeze().float()
        data_fin_flat = self.flat_data(data_fin).squeeze().float()
        data_flat = torch.cat((data_in_flat, data_fin_flat), dim=1)
        theta = theta.squeeze().float()
        theta = theta + torch.Tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]).type_as(theta)
        theta = theta * torch.Tensor([0.05, 20, 20, 1.0 / 6.0, 40, 1.0 / 35.0]).type_as(theta)

        means = self.mu(self.data_process(data_flat))
        stds = torch.Tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]).type_as(theta)
        return theta, means, stds

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop, it is independent from forward
        theta, means, stds = self.forward(batch)
        dist = D.MultivariateNormal(means, torch.diag_embed(stds, dim1=-2, dim2=-1))
        log_term = -dist.log_prob(theta)
        loss = log_term.mean()
        print(means[0].detach(), theta[0].detach())
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta, means, stds = self.forward(batch)
        print(means[0].detach(), theta[0].detach())
        dist = D.MultivariateNormal(means, torch.diag_embed(stds, dim1=-2, dim2=-1))
        val_loss = -dist.log_prob(theta).mean()
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add program level args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    pl.seed_everything(42)
    dataset = MechanisticModelDataset(csv_file="mock_training_data.csv",
                                      root_dir="/data/math-multicellular-struct-devel/hert6124/pakman-develop/examples/DeepScratch/datadir3")
    training_size = int(np.floor(0.75 * len(dataset)))
    validation_size = len(dataset) - training_size
    L = frange_cycle_linear(int(np.ceil(0.1 * training_size / args.batch_size)), 0, 1, 1, 0.6)
    train, val = random_split(dataset, [training_size, validation_size])
    autoencoder = ConditionalAutoEncoder(L, args.learning_rate)

    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(gpus=-1, callbacks=[checkpoint_callback], gradient_clip_val=0.75, default_root_dir=os.getcwd(),
                         limit_train_batches=0.1, limit_val_batches=0.1, min_epochs=10, max_epochs=100)
    trainer.fit(autoencoder,
                DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True),
                DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True))