import torch
from utils import *

class LinearElastic:
    def __init__(self,geometry,E,v):
        self.geom = geometry
        self.get_mu(E, v)
        self.get_lambda(E, v)
        self.get_C()
        self.get_D()

    def get_mu(self,E,v):
        self.mu = 0.5*(1./(1.+v))*E

    def get_lambda(self,E,v):
        # self.Lambda = E * (v/(1.-2.*v))*(1./(1.+v))  # equal to self.Lambda = 2.*self.mu*v/(1.-2.*v)
        self.Lambda = v*E/(1.-v**2) # plane stress

    def get_C(self):
        con_metric = self.geom.con_metric_tensor
        C = torch.zeros(self.geom.batch_size,3,3,device=device)
        C[:,0,0] = self.Lambda * con_metric[:,0,0]**2 + 2. * self.mu * con_metric[:,0,0]**2
        C[:,0,1] = self.Lambda * con_metric[:,0,0] * con_metric[:,1,1] + 2. * self.mu * con_metric[:,0,1]**2
        C[:,0,2] = self.Lambda * con_metric[:,0,0] * con_metric[:,0,1] + 2. * self.mu * con_metric[:,0,0] * con_metric[:,0,1]
        C[:,1,0] = self.Lambda * con_metric[:,0,0] * con_metric[:,1,1] + 2. * self.mu * con_metric[:,0,1]**2
        C[:,1,1] = self.Lambda * con_metric[:,1,1]**2 + 2. * self.mu * con_metric[:,1,1]**2
        C[:,1,2] = self.Lambda * con_metric[:,1,1] * con_metric[:,0,1] + 2. * self.mu * con_metric[:,1,1] * con_metric[:,0,1]
        C[:,2,0] = self.Lambda * con_metric[:,0,0] * con_metric[:,0,1] + 2. * self.mu * con_metric[:,0,0] * con_metric[:,0,1]
        C[:,2,1] = self.Lambda * con_metric[:,1,1] * con_metric[:,0,1] + 2. * self.mu * con_metric[:,1,1] * con_metric[:,0,1]
        C[:,2,2] = self.Lambda * con_metric[:,0,1]**2 + self.mu * (con_metric[:,0,0] * con_metric[:,1,1] + con_metric[:,0,1]**2)
        self.C = C

    def get_D(self):
        con_metric = self.geom.con_metric_tensor
        D = self.mu * con_metric
        self.D = D
