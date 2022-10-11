import torch
import torch.nn as nn
from src.utils import *

class PINN(nn.Module):
    def __init__(self, inputDim, arch, outputDim, bc):
        super(PINN,self).__init__()
        self.model = torch.nn.Sequential()
        currDim = inputDim
        layerCount = 1
        activCount = 1
        for i in range(len(arch)):
            if(type(arch[i]) == int):
                self.model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,arch[i],bias=True))
                currDim = arch[i]
                layerCount += 1
            elif(type(arch[i]) == str):
                self.model.add_module('activ '+str(activCount),getActivation(arch[i]))
                activCount += 1
        self.model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,outputDim,bias=True))
        self.bc = bc

    def forward(self, x):
        out = self.model(x)
        u = torch.zeros(len(x), 5, device=device)
        if self.bc == 'ls':
            trial_func = x[:,0]+0.5
        elif self.bc == 'rs':
            trial_func = x[:,0]-0.5
        elif self.bc == 'ls-rs':
            trial_func = (x[:,0] + 0.5) * (x[:,0] - 0.5)
        elif self.bc == 'fc':
            trial_func = (x[:,0]**2-0.25) * (x[:,1]**2-0.25)
        elif self.bc == 'scordelis_lo':
            u[:,0] = out[:,0] * (x[:,0]**2 + x[:,1]**2)
            u[:,1] = out[:,1] * (x[:,0]+0.5) * (x[:,0]-0.5)
            u[:,2] = out[:,2] * (x[:,0]+0.5) * (x[:,0]-0.5)
            u[:,3] = out[:,3]
            u[:,4] = out[:,4]
            return u
        elif self.bc == 'fc_circular':
            trial_func = (1.-x[:,0]*x[:,0]-x[:,1]*x[:,1])/2.
        else:
            raise ValueError('Missing Dirichlet boundary conditions.')

        # broadcast over all five fields
        u = out * trial_func[:, None]
        return u