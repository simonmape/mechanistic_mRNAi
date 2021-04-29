import numpy as np
import os
from tqdm import tqdm
import pandas as pd

fileIDs = os.listdir("./Raw_data2")

def make_into_densities(fileID):
    density = np.zeros((180,512))
    
    loc = "./Raw_data2/"+fileID
    cellPositions = np.genfromtxt(loc)
    
    if "_t24" in fileID:
        sep = int((cellPositions.shape[0]-94)/2)
    
    else:
        sep = int(cellPositions.shape[0]/2)
        
    Xs = cellPositions[0:sep]-471
    Ys = cellPositions[sep:2*sep]-1

    for i in range(sep):
        density[int(np.floor(Xs[i]))][int(np.floor(Ys[i]))] +=1
    
    new_name = "/data/math-multicellular-struct-devel/hert6124/pakman-develop/examples/DeepScratch/dendir/"+"den_"+fileID
    np.savetxt(new_name,density,delimiter=",")

for file in tqdm(fileIDs):
    print(file)
    make_into_densities(file)



