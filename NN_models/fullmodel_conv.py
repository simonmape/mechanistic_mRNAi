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
            nn.Conv2d(2,10,kernel_size =8,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(8,7),stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(4,8),stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(4,9),stride=2),
            nn.Flatten(),
        )

        self.r1_data_process = nn.Sequential(
            nn.Conv2d(2,10,kernel_size =8,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(8,7),stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(4,8),stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10,10,kernel_size=(4,9),stride=2),
            nn.Flatten(),
        )

        self.q_mu = nn.Sequential(
            nn.Linear(306, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

        self.r1_mu = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

        self.r2_mu = nn.Sequential(
            nn.Linear(306, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Sigmoid()
        )

        self.r2_sigma = nn.Sequential(
            nn.Linear(306, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

        self.flat_data = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            #nn.AvgPool2d(kernel_size=(3, 4)),
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
        data_flat = torch.stack((data_in_flat, data_fin_flat),dim=1)
        theta = theta.squeeze().float()
        print(self.r1_data_process(data_flat).size)
        r1_means = self.r1_mu(self.r1_data_process(data_flat))
        r1_stds = torch.ones_like(r1_means)
        r1_dist = D.MultivariateNormal(r1_means, torch.diag_embed(r1_stds, dim1=-2, dim2=-1))
        z_r1 = r1_dist.rsample()

        r2_means = self.r2_mu(torch.cat((self.data_process(data_flat), z_r1), dim=1))
        r2_stds = torch.exp(self.r2_sigma(torch.cat((self.data_process(data_flat), z_r1), dim=1)))
        return r2_means, r2_stds

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop, it is independent from forward
        data_in = batch['IC']
        data_fin = batch['FC']
        theta = batch['par'].unsqueeze(dim=1)

        data_in_flat = self.flat_data(data_in).squeeze().float()
        data_fin_flat = self.flat_data(data_fin).squeeze().float()
        data_flat = torch.stack((data_in_flat, data_fin_flat),dim=1)
        theta = theta.squeeze().float()
        theta = theta + torch.Tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]).type_as(theta)
        theta = theta * torch.Tensor([0.05, 20, 20, 1.0 / 6.0, 40, 1.0 / 35.0]).type_as(theta)

        q_means = self.q_mu(torch.cat((self.data_process(data_flat), theta), dim=1))
        q_stds = torch.ones_like(q_means)
        q_dist = D.MultivariateNormal(q_means, torch.diag_embed(q_stds, dim1=-2, dim2=-1))

        r1_means = self.r1_mu(self.r1_data_process(data_flat))
        r1_stds = torch.ones_like(r1_means)
        r1_dist = D.MultivariateNormal(r1_means, torch.diag_embed(r1_stds, dim1=-2, dim2=-1))

        z_q = q_dist.rsample()
        KL_term = D.kl_divergence(q_dist, r1_dist)

        r2_means = self.r2_mu(torch.cat((self.data_process(data_flat), z_q), dim=1))
        r2_stds = torch.exp(self.r2_sigma(torch.cat((self.data_process(data_flat), z_q), dim=1)))

        print(r1_means[0].detach(), q_means[0].detach())
        print(r2_means[0].detach(), theta[0].detach())

        r2_dist = D.MultivariateNormal(r2_means, torch.diag_embed(r2_stds, dim1=-2, dim2=-1))

        log_term = -r2_dist.log_prob(theta)
        loss = self.schedule[batch_idx] * KL_term.mean() + log_term.mean()

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        theta = batch['par'].unsqueeze(dim=1)
        theta = theta.squeeze().float()
        theta = theta + torch.Tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]).type_as(theta)
        theta = theta * torch.Tensor([0.05, 20, 20, 1.0 / 6.0, 40, 1.0 / 35.0]).type_as(theta)
        r2_means, r2_stds = self.forward(batch)
        r2_dist = D.MultivariateNormal(r2_means, torch.diag_embed(r2_stds, dim1=-2, dim2=-1))
        val_loss = -r2_dist.log_prob(theta).mean()
        # print(val_loss.item(),"True par: ", theta[0], "Simulated par: ", r2_means[0])
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
    L = frange_cycle_linear(int(np.ceil(0.1 * training_size / args.batch_size)), 0, 1, 1, 0.7)
    train, val = random_split(dataset, [training_size, validation_size])
    autoencoder = ConditionalAutoEncoder(L, args.learning_rate)

    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(gpus=-1, callbacks=[checkpoint_callback], gradient_clip_val=0.75, default_root_dir=os.getcwd(),
                         limit_train_batches=0.1, limit_val_batches=0.1, min_epochs=10, max_epochs=100)
    trainer.fit(autoencoder,
                DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True),
                DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True))