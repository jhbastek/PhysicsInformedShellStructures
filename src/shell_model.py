import torch
from src.utils import *

class LinearNagdhi:
    def __init__(self,geometry):
        self.geom = geometry
        self.get_membrane_strain_matrix()
        self.get_bending_strain_matrix()
        self.get_shear_strain_matrix()

    def get_membrane_strain_matrix(self):
        christoffel_sym = self.geom.christoffel_sym
        cov_curv = self.geom.cov_curv_tensor

        Bm = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bm[:,0,0] = -christoffel_sym[:,0,0,0]
        Bm[:,0,1] = -christoffel_sym[:,1,0,0]
        Bm[:,0,2] = -cov_curv[:,0,0]
        Bm[:,1,0] = -christoffel_sym[:,0,1,1]
        Bm[:,1,1] = -christoffel_sym[:,1,1,1]
        Bm[:,1,2] = -cov_curv[:,1,1]
        Bm[:,2,0] = -christoffel_sym[:,0,0,1] - christoffel_sym[:,0,1,0]
        Bm[:,2,1] = -christoffel_sym[:,1,1,0] - christoffel_sym[:,1,0,1]
        Bm[:,2,2] = -2.*cov_curv[:,0,1]

        Bm1 = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bm1[:,0,0] = 1.
        Bm1[:,2,1] = 1.

        Bm2 = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bm2[:,1,1] = 1.
        Bm2[:,2,0] = 1.

        self.membrane_strain_matrix = Bm, Bm1, Bm2

    def get_bending_strain_matrix(self):
        christoffel_sym = self.geom.christoffel_sym
        mixed_curv = self.geom.mixed_curv_tensor
        third_ff = self.geom.third_fundamental_form

        Bk = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bk[:,0,0] = mixed_curv[:,0,0] * christoffel_sym[:,0,0,0] + mixed_curv[:,1,0] * christoffel_sym[:,0,1,0]
        Bk[:,0,1] = mixed_curv[:,0,0] * christoffel_sym[:,1,0,0] + mixed_curv[:,1,0] * christoffel_sym[:,1,1,0]
        Bk[:,0,2] = third_ff[:,0,0]
        Bk[:,0,3] = -christoffel_sym[:,0,0,0]
        Bk[:,0,4] = -christoffel_sym[:,1,0,0]

        Bk[:,1,0] = mixed_curv[:,0,1] * christoffel_sym[:,0,0,1] + mixed_curv[:,1,1] * christoffel_sym[:,0,1,1]
        Bk[:,1,1] = mixed_curv[:,0,1] * christoffel_sym[:,1,0,1] + mixed_curv[:,1,1] * christoffel_sym[:,1,1,1]
        Bk[:,1,2] = third_ff[:,1,1]
        Bk[:,1,3] = -christoffel_sym[:,0,1,1]
        Bk[:,1,4] = -christoffel_sym[:,1,1,1]

        Bk[:,2,0] = mixed_curv[:,0,1] * christoffel_sym[:,0,0,0] + mixed_curv[:,1,1] *  christoffel_sym[:,0,1,0] + mixed_curv[:,0,0] *  christoffel_sym[:,0,0,1] + mixed_curv[:,1,0] *  christoffel_sym[:,0,1,1]
        Bk[:,2,1] = mixed_curv[:,0,1] * christoffel_sym[:,1,0,0] + mixed_curv[:,1,1] *  christoffel_sym[:,1,1,0] + mixed_curv[:,0,0] *  christoffel_sym[:,1,0,1] + mixed_curv[:,1,0] *  christoffel_sym[:,1,1,1]
        Bk[:,2,2] = 2.*third_ff[:,0,1]
        Bk[:,2,3] = -christoffel_sym[:,0,0,1] - christoffel_sym[:,0,1,0]
        Bk[:,2,4] = -christoffel_sym[:,1,0,1] - christoffel_sym[:,1,1,0]

        Bk1 = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bk1[:,0,0] = -mixed_curv[:,0,0]
        Bk1[:,0,1] = -mixed_curv[:,1,0]
        Bk1[:,0,3] = 1.
        Bk1[:,2,0] = -mixed_curv[:,0,1]
        Bk1[:,2,1] = -mixed_curv[:,1,1]
        Bk1[:,2,4] = 1.

        Bk2 = torch.zeros(self.geom.batch_size,3,5,device=device)
        Bk2[:,1,0] = -mixed_curv[:,0,1]
        Bk2[:,1,1] = -mixed_curv[:,1,1]
        Bk2[:,1,4] = 1.
        Bk2[:,2,0] = -mixed_curv[:,0,0]
        Bk2[:,2,1] = -mixed_curv[:,1,0]
        Bk2[:,2,3] = 1.

        self.bending_strain_matrix = Bk, Bk1, Bk2

    def get_shear_strain_matrix(self):
        mixed_curv = self.geom.mixed_curv_tensor

        By = torch.zeros(self.geom.batch_size,2,5,device=device)
        By[:,0,0] = mixed_curv[:,0,0]
        By[:,0,1] = mixed_curv[:,1,0]
        By[:,0,3] = 1.
        By[:,1,0] = mixed_curv[:,0,1]
        By[:,1,1] = mixed_curv[:,1,1]
        By[:,1,4] = 1.

        By1 = torch.zeros(self.geom.batch_size,2,5,device=device)
        By1[:,0,2] = 1.

        By2 = torch.zeros(self.geom.batch_size,2,5,device=device)
        By2[:,1,2] = 1.

        self.shear_strain_matrix = By, By1, By2