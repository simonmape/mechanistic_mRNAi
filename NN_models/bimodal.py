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
                             self.ID_frame.iloc[idx,2])
        
        IC = torch.from_numpy(np.genfromtxt(IC_file,delimiter=","))
        FC = torch.from_numpy(np.genfromtxt(FC_file,delimiter=","))
        par = torch.from_numpy(np.genfromtxt(par_file,delimiter=","))
        
        IC = IC.unsqueeze(dim=0)
        FC = FC.unsqueeze(dim=0)
        sample = {'IC': IC, 'FC': FC,'par':par}

        return sample


class ConditionalAutoEncoder(pl.LightningModule):
    def __init__(self,schedule,learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.schedule = torch.from_numpy(schedule)
        
        self.q = nn.Sequential(
            nn.Linear(11525, 11525), #Fully connected
            nn.ReLU(),
            nn.Linear(11525, 2048),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(2048,128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128,20),
        )
        self.r1 = nn.Sequential(
            nn.Linear(11520, 11520),
            nn.ReLU(),
            nn.Linear(11520, 4096),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(2048,512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128,42),
        )
        self.r2 = nn.Sequential(
            nn.Linear(10, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64,20),
        )
        
        self.flat_data = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def reshape_q(self,q_output):
        q_means,q_stds = torch.split(q_output,10,dim=1)
        t = nn.Tanh()
        q_means = t(q_means)
        r = nn.ReLU()
        safety = torch.ones(10).new_full((10,), 0.0001)
        safety = safety.type_as(q_output)
        q_stds = r(q_stds) + safety
        return q_means,q_stds
        
    def reshape_r1(self,r1_output):
        r1_cats,r1_means,r1_stds = r1_output.split((2,20,20),dim=1)
        r1_cats = r1_cats.reshape((-1,2))
        s = nn.Sigmoid()
        r1_cats = s(r1_cats)
        
        r1_means = r1_means.reshape((-1,2,10))
        t = nn.Tanh()
        r1_means = t(r1_means)
        
        r1_stds = r1_stds.reshape((-1,2,10))
        r = nn.ReLU()
        safety = torch.ones((2,10)).new_full((2,10),0.0001)
        safety = safety.type_as(r1_output)
        r1_stds = r(r1_stds) + safety
        return r1_cats,r1_means,r1_stds
        
    def reshape_r2(self,r2_output):
        r2_means,r2_stds = r2_output.split((5,15),dim=1)
        t = nn.Tanh()
        r2_means = t(r2_means)
        location = torch.Tensor([50,0.025,0,0.01,25])
        location = location.type_as(r2_output)
        scale = torch.Tensor([50,0.025,1,0.01,25])
        scale = scale.type_as(r2_output)
        r2_means = location + r2_means*scale
        
        r = nn.ReLU()
        r2_stds = r(r2_stds)
        safety = torch.ones(15).new_full((15,),0.0001)
        safety = safety.type_as(r2_output)
        r2_stds = r2_stds + safety
        r2_stds.reshape((-1,15))

        m = torch.zeros((r2_means.size()[0],5,5)).type_as(r2_output)
        tril_indices = torch.tril_indices(5,5).squeeze()
        m[:,tril_indices[0], tril_indices[1]] = r2_stds
        r2_stds =m
        return r2_means,r2_stds
        
    
    def forward(self,batch):
        data_in = batch['IC']
        data_fin = batch['FC']
        theta = batch['par'].unsqueeze(dim=1)

        data_in_flat = self.flat_data(data_in).squeeze().float()
        data_fin_flat = self.flat_data(data_fin).squeeze().float()
        data_flat = torch.cat((data_in_flat,data_fin_flat),dim=1)
        theta = theta.squeeze().float()
        
        r1_cats,r1_means,r1_stds = self.reshape_r1(self.r1(data_flat))
        mix = D.Categorical(r1_cats)
        comp = D.Independent(D.Normal(r1_means, r1_stds), 1)
        r1_dist = D.MixtureSameFamily(mix, comp)
        z_r1 = r1_dist.sample()
        
        r2_means,r2_stds = self.reshape_r2(self.r2(z_r1))
        return r2_means,r2_stds
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop, it is independent from forward
        data_in = batch['IC']
        data_fin = batch['FC']
        theta = batch['par'].unsqueeze(dim=1)

        data_in_flat = self.flat_data(data_in).squeeze().float()
        data_fin_flat = self.flat_data(data_fin).squeeze().float()
        data_flat = torch.cat((data_in_flat,data_fin_flat),dim=1)
        theta = theta.squeeze().float()
        
        q_args = torch.cat((data_flat,theta),dim=1)
        q_means,q_stds = self.reshape_q(self.q(q_args))
        q_dist = D.MultivariateNormal(q_means,torch.diag_embed(q_stds,dim1=-2,dim2=-1))
        
        r1_cats,r1_means,r1_stds = self.reshape_r1(self.r1(data_flat))
        mix = D.Categorical(r1_cats)
        comp = D.Independent(D.Normal(r1_means, r1_stds), 1)
        r1_dist = D.MixtureSameFamily(mix, comp)
        
        auxiliary = D.MultivariateNormal(torch.zeros(10), torch.eye(10)).sample()
        auxiliary = auxiliary.type_as(theta)
        z_q = q_means + q_stds*auxiliary
        
        log_prob_q = q_dist.log_prob(z_q)
        log_prob_r1 = r1_dist.log_prob(z_q)
        
        KL_term = log_prob_q - log_prob_r1
        
        r2_means,r2_stds = self.reshape_r2(self.r2(z_q))
        r2_dist = D.MultivariateNormal(r2_means,scale_tril=r2_stds)
        
        log_term = -r2_dist.log_prob(theta)
        loss = self.schedule[batch_idx]*KL_term.mean() + log_term.mean()
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss,sync_dist=True)
        print(self.schedule[batch_idx].item()*KL_term.mean().item(),log_term.mean().item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        theta = batch['par'].unsqueeze(dim=1)
        theta = theta.squeeze().float()
        r2_means, r2_stds = self.forward(batch)
        r2_dist = D.MultivariateNormal(r2_means,scale_tril=r2_stds)
        val_loss = -r2_dist.log_prob(theta).mean()
        self.log('val_loss', val_loss,sync_dist=True)
        print(val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

if __name__ == '__main__':
    parser = ArgumentParser()
    
    #Add program level args
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    args = parser.parse_args()

    pl.seed_everything(42)
    #dataset = MechanisticModelDataset(csv_file="initial_exploration.csv",root_dir="/Users/simonmartinaperez/Desktop/datadir2")
    dataset = MechanisticModelDataset(csv_file="mock_training_data.csv",root_dir="/data/math-multicellular-struct-devel/hert6124/pakman-develop/examples/DeepScratch/datadir2")
    training_size =int(np.floor(0.75*len(dataset)))
    validation_size = len(dataset)-training_size
    train, val = random_split(dataset, [training_size, validation_size])
    L = frange_cycle_linear(int(np.ceil(0.1*training_size/args.batch_size)),0,1,1,0.8)

    autoencoder = ConditionalAutoEncoder(L,args.learning_rate)
    trainer = pl.Trainer(gpus=-1,gradient_clip_val=0.75,default_root_dir=os.getcwd(),limit_train_batches=0.1,limit_val_batches=0.1,min_epochs=10,max_epochs=1000)
    #trainer = pl.Trainer(gradient_clip_val=0.75,default_root_dir=os.getcwd(),limit_train_batches=0.1,limit_val_batches=0.1,min_epochs=10,max_epochs=1000)
    trainer.fit(autoencoder, DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0), DataLoader(val,batch_size=args.batch_size,shuffle=True,num_workers=0))
