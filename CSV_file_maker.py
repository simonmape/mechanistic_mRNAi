import numpy as np
import os
from tqdm import tqdm
import pandas as pd

fileIDs = os.scandir("/data/math-multicellular-struct-devel/hert6124/pakman-develop/examples/DeepScratch/datadir2")
exIDs = []
FCs = []
params = []

'''
Extract the names of the simulated final conditions and 
parameter files while saving corresponding experiment IDs
'''

IDs = []

for ID in tqdm(fileIDs): 
    IDs.append(ID.name[:-9])


'''
Get unique identifiers for each of the simulated final experiments
'''
IDs = list(dict.fromkeys(IDs))

'''
Get reference to the experimental initial condition from 
data file name
'''
experimentICs = [x.strip() for x in open('Mock.txt', 'r').readlines()]

'''
Make list of the experimental initial conditions
'''
ICs = []
for ID in tqdm(range(len(IDs))):
    ICs.append("den_"+experimentICs[int(IDs[ID].split("_")[0])])
    FCs.append(IDs[ID]+"_sim_.txt")
    params.append(IDs[ID]+"_par_.txt")

'''
Convert data into CSV file for export
'''
CSV_dict = {'Initial Condition':ICs,'Final Condition':FCs,'Parameter':params}
data = pd.DataFrame(CSV_dict)
data.to_csv("mock_training_data2.csv",index=False)
