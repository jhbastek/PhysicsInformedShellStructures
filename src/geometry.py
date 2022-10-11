import numpy as np
import torch
import torch.nn.functional as f
from src.utils import *
from torch.autograd.functional import vjp

class Geometry:
    def __init__(self,chart,xi):
        self.chart = chart
        self.batch_size = len(xi)
        self.cov_a1, self.cov_a2, self.cov_a3 = torch.split(self.get_cov_localBasis(xi),3,dim=1)
        self.cov_a1_d, self.cov_a2_d, self.cov_a3_d = self.get_cov_localBasis_d(xi)
        self.get_cov_metric_tensor()
        self.get_sqrt_det_a()
        self.get_parametric_area(xi)
        self.get_surface_area()
        self.get_con_metric_tensor()
        self.get_cov_curv_tensor()
        self.get_mixed_curv_tensor()
        self.get_third_fundamental_form()
        self.get_christoffel_sym()
        self.get_T_u()
        self.get_T_u_d()
        self.get_T_theta()
        self.get_T_theta_d()
        self.get_S()
        # self.get_S_inv()
        # self.get_S_d1()
        # self.get_S_d2()
        # self.get_Sv()

    def get_mapping(self,xi):
        if(self.chart == 'hyperb_parab'):
            x = xi[:,0]
            y = xi[:,1]
            z = 1.*(xi[:,0]**2 - xi[:,1]**2)
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'scordelis_lo'):
            r = 25./50.
            L = 50./50.
            angle = xi[:,1]*4.*np.pi/9.
            x = L * xi[:,0]
            y = r * torch.sin(angle)
            z = r * torch.cos(angle)
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'hyperb_parab_check'):
            x = 0.5*xi[:,0]
            y = 0.5*xi[:,1]
            z = 0.25*xi[:,0]**2 - 0.25*xi[:,1]**2
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'cylinder'):
            r = 1.
            x = xi[:,0]
            y_temp = (xi[:,1] - np.pi/2.) * r
            y = r * torch.sin(y_temp/r)
            z = r * torch.cos(y_temp/r)
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'plate'):
            x = xi[:,0]
            y = xi[:,1]
            z = 0.*(xi[:,0]**2 - xi[:,1]**2)
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'hemisphere'):
            radius = 1.
            x = xi[:,0]
            y = xi[:,1]
            z = torch.sqrt(1.0001 - xi[:,0]**2 - xi[:,1]**2)
            return torch.stack((x,y,z),dim=1)
        elif(self.chart == 'hemisphere_polar'):
            # input: xi[:,0] -> r, xi[:,0]->angle
            radius = 1.
            x = xi[:,0] * torch.sin(xi[:,1])
            y = xi[:,0] * torch.cos(xi[:,1])
            z = torch.sqrt(radius**2 - (xi[:,0] * torch.sin(xi[:,1]))**2 - (xi[:,0] * torch.cos(xi[:,1]))**2)
            return torch.stack((x,y,z),dim=1)
        else:
            assert False, 'Chart unknown.'

    # Local basis
    def get_cov_localBasis(self,xi):
        cov_a1 = torch.zeros(self.batch_size,3,device=device)
        cov_a2 = torch.zeros(self.batch_size,3,device=device)
        m,n = self.batch_size, 3
        first_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1,device=device)*(i==j) for j in range(n)],1)
            jacobian_vjp = vjp(self.get_mapping, xi, v, create_graph=True)[1]
            first_grad_list.append(jacobian_vjp)
        first_grad = torch.cat(first_grad_list,1)
        first_grad_reshape = torch.reshape(first_grad, (self.batch_size,3,2))
        cov_a1 = first_grad_reshape[:,:,0]
        cov_a2 = first_grad_reshape[:,:,1]
        cov_a3_unnorm = torch.cross(cov_a1,cov_a2, dim=1)
        cov_a3 = f.normalize(cov_a3_unnorm, p=2, dim=1)
        # stack in first dimension
        cov_localBasis = torch.cat((cov_a1,cov_a2,cov_a3),dim=1)
        return cov_localBasis

    # Local basis derivatives
    def get_cov_localBasis_d(self,xi):
        m,n = self.batch_size, 9
        second_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1,device=device)*(i==j) for j in range(n)],1)
            hessian_vjp = vjp(self.get_cov_localBasis, xi, v, create_graph=True)[1]
            second_grad_list.append(hessian_vjp)
        second_grad = torch.cat(second_grad_list,1)
        # reshape
        cov_a1_d = torch.stack([second_grad[:,0:5:2],second_grad[:,1:6:2]],dim=1)
        cov_a2_d = torch.stack([second_grad[:,6:11:2],second_grad[:,7:12:2]],dim=1)
        cov_a3_d = torch.stack([second_grad[:,12:17:2],second_grad[:,13:18:2]],dim=1)
        return cov_a1_d, cov_a2_d, cov_a3_d

    # In-plane covariant metric tensor
    def get_cov_metric_tensor(self):
        top = torch.stack((bdot(self.cov_a1,self.cov_a1),bdot(self.cov_a1,self.cov_a2)),dim=1)
        bot = torch.stack((bdot(self.cov_a2,self.cov_a1),bdot(self.cov_a2,self.cov_a2)),dim=1)
        full = torch.stack((top,bot),dim=2)
        self.cov_metric_tensor = full

    # Square root of the determinant of the covariant metric tensor
    def get_sqrt_det_a(self):
        self.sqrt_det_a = torch.sqrt(torch.det(self.cov_metric_tensor))

    # Parametric area for Monte Carlo integration
    def get_parametric_area(self,xi):
        
        if self.chart == 'hemisphere' or self.chart == 'hemisphere_polar':
            self.parametric_area = (torch.max(xi[:,0])-torch.min(xi[:,0]))*(torch.max(xi[:,1])-torch.min(xi[:,1])) * np.pi / 4.
        else:
            self.parametric_area = (torch.max(xi[:,0])-torch.min(xi[:,0]))*(torch.max(xi[:,1])-torch.min(xi[:,1]))

    # Surface area using Monte Carlo integration
    def get_surface_area(self):
        self.surface_area = self.parametric_area * torch.mean(self.sqrt_det_a)
    
    # In-plane contravariant metric tensor (could also be computed by contravariant basis)
    def get_con_metric_tensor(self):
        self.con_metric_tensor = torch.inverse(self.cov_metric_tensor)

    # Covariant second fundamental form
    def get_cov_curv_tensor(self):
        cov_a_d = torch.stack((self.cov_a1_d,self.cov_a2_d),dim=1)
        self.cov_curv_tensor = torch.einsum('...k,...ijk->...ij',self.cov_a3,cov_a_d)

    # Mixed second fundamental form (for higher-order tensors use torch.einsum; for the here considered 2nd order tensors simple torch.mm sufficient)
    def get_mixed_curv_tensor(self):
        con_metric = self.con_metric_tensor
        cov_curv = self.cov_curv_tensor
        mixed_curv_tensor = torch.bmm(con_metric,cov_curv)
        self.mixed_curv_tensor = mixed_curv_tensor

    # Covariant third fundamental form
    def get_third_fundamental_form(self):
        mixed_curv = self.mixed_curv_tensor
        cov_curv = self.cov_curv_tensor
        # matrix product has to be in this order to correspond to index notation, as we treat the mixed tensor as: contravariant index -> row, covariant index -> column
        third_fundamental_form = torch.bmm(cov_curv, mixed_curv)
        self.third_fundamental_form = third_fundamental_form

    # Christoffel symbols (1)(2), accessed in this order
    def get_christoffel_sym(self):
        cov_a = torch.stack((self.cov_a1,self.cov_a2),dim=1)
        con_metric = self.con_metric_tensor
        con_a = torch.einsum('...ij,...jk->...ik',con_metric,cov_a)
        cov_a_d = torch.stack((self.cov_a1_d,self.cov_a2_d),dim=1) # first index a1 or a2, second index derivative w.r.t. to 1 or 2
        christoffel_sym = torch.einsum('...ijk,...lk->...lij',cov_a_d,con_a)
        self.christoffel_sym = christoffel_sym

    # Basis transformation matrix
    def get_T_u(self):
        T_u = torch.stack((self.cov_a1,self.cov_a2,self.cov_a3),dim=1)
        self.T_u = T_u

    # Basis transformation matrix derivative
    def get_T_u_d(self):
        cov_a_d1 = torch.stack((self.cov_a1_d[:,0],self.cov_a2_d[:,0],self.cov_a3_d[:,0]),dim=1)
        cov_a_d2 = torch.stack((self.cov_a1_d[:,1],self.cov_a2_d[:,1],self.cov_a3_d[:,1]),dim=1)
        self.T_u_d = cov_a_d1, cov_a_d2

    # Basis transformation matrix
    def get_T_theta(self):
        T_theta = torch.stack((self.cov_a1,self.cov_a2),dim=1)
        self.T_theta = T_theta

    # Basis transformation matrix derivative
    def get_T_theta_d(self):
        cov_a_d1 = torch.stack((self.cov_a1_d[:,0],self.cov_a2_d[:,0]),dim=1)
        cov_a_d2 = torch.stack((self.cov_a1_d[:,1],self.cov_a2_d[:,1]),dim=1)
        self.T_theta_d = cov_a_d1, cov_a_d2

    def get_S(self):
        T_u_top = self.T_u
        S_top_right = torch.zeros(self.batch_size,3,2,device=device)
        S_bot_left = torch.zeros(self.batch_size,2,3,device=device)
        S_top = torch.cat((T_u_top,S_top_right),dim=2)
        S_bot = torch.cat((S_bot_left,torch.eye(2,device=device).repeat(self.batch_size,1,1)),dim=2)
        S = torch.cat((S_top,S_bot),dim=1)
        self.S = S

    def get_S_inv(self):
        T_u_top = torch.inverse(self.T_u)
        T_theta_bot = torch.zeros(self.batch_size,3,2,device=device)
        S_top_right = torch.zeros(self.batch_size,3,2,device=device)
        S_bot_left = torch.zeros(self.batch_size,3,3,device=device)
        S_top = torch.cat((T_u_top,S_top_right),dim=2)
        S_bot = torch.cat((S_bot_left,T_theta_bot),dim=2)
        S_inv = torch.cat((S_top,S_bot),dim=1)
        self.S_inv = S_inv

    def get_S_d1(self):
        T_u_d1_top = self.T_u_d[0]
        T_theta_d1_bot = torch.zeros(self.batch_size,2,2,device=device) # self.T_theta_d[0]
        S_d1_top_right = torch.zeros(self.batch_size,3,2,device=device)
        S_d1_bot_left = torch.zeros(self.batch_size,2,3,device=device)
        S_d1_top = torch.cat((T_u_d1_top,S_d1_top_right),dim=2)
        S_d1_bot = torch.cat((S_d1_bot_left,T_theta_d1_bot),dim=2)
        S_d1 = torch.cat((S_d1_top,S_d1_bot),dim=1)
        self.S_d1 = S_d1 

    def get_S_d2(self):
        T_u_d2_top = self.T_u_d[1]
        T_theta_d2_bot = torch.zeros(self.batch_size,2,2,device=device) # self.T_theta_d[1]
        S_d2_top_right = torch.zeros(self.batch_size,3,2,device=device)
        S_d2_bot_left = torch.zeros(self.batch_size,2,3,device=device)
        S_d2_top = torch.cat((T_u_d2_top,S_d2_top_right),dim=2)
        S_d2_bot = torch.cat((S_d2_bot_left,T_theta_d2_bot),dim=2)
        S_d2 = torch.cat((S_d2_top,S_d2_bot),dim=1)
        self.S_d2 = S_d2

    # # Transformation from 5-parameter to 6-parameter model
    # def get_Sv(self):
    #     e_y = torch.zeros(self.batch_size,3,device=device)
    #     e_y[:,1] = 1.
    #     cov_v_1_unnorm = torch.cross(e_y,self.cov_a3,dim=1)
    #     cov_v_1 = f.normalize(cov_v_1_unnorm, p=2, dim=1)
    #     cov_v_2 = torch.cross(cov_v_1,self.cov_a3,dim=1)

    #     Sv_top_left = torch.diag_embed(torch.ones(self.batch_size,3,device=device))
    #     Sv_top_right = torch.zeros(self.batch_size,3,2,device=device)
    #     Sv_bot_right = torch.stack((cov_v_2,cov_v_1),dim=2)
    #     Sv_bot_left = torch.zeros(self.batch_size,3,3,device=device)
    #     Sv_top = torch.cat((Sv_top_left,Sv_top_right),dim=2)
    #     Sv_bot = torch.cat((Sv_bot_left,Sv_bot_right),dim=2)
    #     Sv = torch.cat((Sv_top,Sv_bot),dim=1)

    #     self.Sv = Sv