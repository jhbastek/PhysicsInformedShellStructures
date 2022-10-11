import os
import torch
from src.utils import *
from src.eval_utils import *
from src.pinn_model import PINN
from src.geometry import Geometry
from src.shell_model import LinearNagdhi
from src.material_model import LinearElastic
from params import create_param_dict

if __name__ == '__main__':

    # select 'hyperb_parab_strong' as study
    study = 'hyperb_parab_strong'

    # to reproduce results from paper
    fix_seeds()

    # create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('eval', exist_ok=True)
    os.makedirs('loss_history', exist_ok=True)

    # we consider double precision
    torch.set_default_dtype(torch.float64)

    # load study parameters
    param_dict = create_param_dict(study)
    geometry = param_dict['geometry']
    loading = param_dict['loading']
    E = param_dict['E']
    thickness = param_dict['thickness']
    shell_density = param_dict['shell_density']
    nu = param_dict['nu']
    shear_factor = param_dict['shear_factor']
    bcs = param_dict['bcs']
    N_col = param_dict['N_col']
    col_sampling = param_dict['col_sampling']
    epochs = param_dict['epochs']
    opt_switch_epoch = param_dict['opt_switch_epoch']
    FEM_sol_dir = param_dict['FEM_sol_dir']

    # frequency of L2 error evaluation (takes some time)
    l2_eval_freq = 10

    # activate to print out losses
    verbose = True

    # activate to plot predictions and compare to FEM
    plot = True

    # Sample collocation points according to col_sampling
    xi_col_temp = get_col_sampling(col_sampling, N_col)

    # Transform to reference domain (warning: if reference domain changed, BCs must be adjusted accordingly)
    if col_sampling != 'concentric':
        xi_col = xi_col_temp * 1. - 0.5
    # activate gradient tracking for geometric measures
    xi_col.requires_grad = True

    # PINN setup
    arch = [50,'gelu',50,'gelu',50,'gelu']
    pn = PINN(2,arch,5,bcs).to(device)

    # define optimizers
    optimizer_ADAM = torch.optim.Adam(pn.parameters(), lr=1.e-3)
    optimizer_LBFGS = torch.optim.LBFGS(pn.parameters(), tolerance_grad=1e-20, tolerance_change=1e-20, line_search_fn='strong_wolfe')

    # tracking
    loss_strong_form = []
    L2_error_list = []

    # initialize shell
    print('Precompute geometric measures at collocation points.')
    geom = Geometry(geometry,xi_col)
    shell = LinearNagdhi(geom)
    material = LinearElastic(geom,E,nu)
    # geometric quantities
    S = geom.S
    sqrt_det_a = geom.sqrt_det_a
    param_area = geom.parametric_area
    contra_metric = geom.con_metric_tensor
    # additionally required for strong form
    christoffel = geom.christoffel_sym.clone().detach()
    second_ff_cov = geom.cov_curv_tensor.clone().detach()
    second_ff = geom.mixed_curv_tensor
    # we rearrange the strain contributions to 3 matrices to distinguish between terms acting directly
    # on the solution field or the two corresponding derivatives w.r.t. curvilinear coordinates (_1, _2)
    Bm, Bm1, Bm2 = shell.membrane_strain_matrix
    Bk, Bk1, Bk2 = shell.bending_strain_matrix
    By, By1, By2 = shell.shear_strain_matrix
    # material properties, using plane-stress conditions for Lamé constant lambda
    C = material.C
    B = material.C
    D = material.D
    print('Done.')

    def closure():
        def pred_strain(xi):
            def global_to_local(x):
                return bmv(S,pn(x))
            first_grad_list = []
            for i in range(5):
                v = torch.cat([torch.ones(batch_len,1,device=device)*(i==j) for j in range(5)],1)
                jacobian_vjp = vjp_inplace(global_to_local, xi, v, create_graph=True)[1]
                first_grad_list.append(jacobian_vjp)
            first_grad = torch.cat(first_grad_list,1)
            first_grad_reshape = torch.reshape(first_grad, (batch_len,5,2))
            pred_5d = global_to_local(xi)
            pred_5d_1 = first_grad_reshape[:,:,0]
            pred_5d_2 = first_grad_reshape[:,:,1]

            # assemble membrane strains
            membrane_strains = bmv(Bm,pred_5d) + bmv(Bm1,pred_5d_1) + bmv(Bm2,pred_5d_2)
            n = thickness * bmv(C,membrane_strains)
            # assemble bending strains
            bending_strains = bmv(Bk,pred_5d) + bmv(Bk1,pred_5d_1) + bmv(Bk2,pred_5d_2)
            m = (thickness**3/12.) * bmv(B,bending_strains)
            # assemble shear strains
            shear_strains = bmv(By,pred_5d) + bmv(By1,pred_5d_1) + bmv(By2,pred_5d_2)
            q = shear_factor * thickness * bmv(D,shear_strains)

            # be extremely careful with n_ as it is not symmetric!
            n_temp = Voigt_to_full(n) - torch.einsum('...ij,...jk->...ik', second_ff, Voigt_to_full(m))
            n_ = torch.flatten(n_temp,start_dim=1)
            return torch.cat((n_,m,q),dim=1) 

        # obtain n_, m, q (force tensors, follows notation of paper)
        n_, m, q = torch.split(pred_strain(xi_col),[4,3,2],dim=1)

        # obtain required derivatives of force tensors for curvilinear divergence
        second_grad_list = []
        for i in range(9):
            v = torch.cat([torch.ones(batch_len,1,device=device)*(i==j) for j in range(9)],1)
            hessian_vjp = vjp_inplace(pred_strain, xi_col, v, create_graph=True)[1]
            second_grad_list.append(hessian_vjp)
        second_grad = torch.cat(second_grad_list,1)

        # reshape
        n_d = torch.stack([second_grad[:,0:7:2],second_grad[:,1:8:2]],dim=1)
        m_d = torch.stack([second_grad[:,8:13:2],second_grad[:,9:14:2]],dim=1)
        q_d = torch.stack([second_grad[:,14:17:2],second_grad[:,15:18:2]],dim=1)               

        # assemble forces in correct shape (batch x tensor)
        n_ = torch.reshape(n_,(-1,2,2))
        m = Voigt_to_full(m)

        # assemble force derivatives in correct shape (batch x deriv x tensor)
        n_d = torch.reshape(n_d,(-1,2,2,2))
        m_d = Voigt_to_full(m_d)

        # external force (transform from physical to curvilinear)
        if loading == 'gravity':
            f_phys = torch.zeros(N_col,5,device=device)
            f_phys[:,2] = -1. * shell_density * thickness
            f_local = bmv(S,f_phys)
        else:
            raise ValueError('Loading type not recognized.')

        # strong form in consistent notation (finally...)
        first_eq = div_2nd_order_u2(m,m_d,christoffel) - q
        second_eq = div_2nd_order_u2(n_,n_d,christoffel) - torch.einsum('...ij,...j->...i',second_ff,q) + torch.einsum('...i,...ij->...j',f_local[:,0:2],contra_metric)
        third_eq = div_1st_order_u(q,q_d,christoffel) + torch.einsum('...ij,...ij->...',second_ff_cov,n_) + f_local[:,2]

        # full loss
        loss = (torch.mean(first_eq[:,0]**2) + torch.mean(first_eq[:,1]**2) + torch.mean(second_eq[:,0]**2)
            + torch.mean(second_eq[:,1]**2) + torch.mean(third_eq[:]**2))

        if verbose:
            print('Residual loss: {:.2e}'.format(loss.item()))

        # optimizer step
        optimizer.zero_grad()
        if optimizer == optimizer_LBFGS:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        return loss

    print('Start training.')
    for epoch in range(epochs):
        batch_len = len(xi_col)
        if (epoch < opt_switch_epoch):   
            # Adam optimizer step
            optimizer = optimizer_ADAM
        else:
            # LBFGS optimizer step
            optimizer = optimizer_LBFGS
        optimizer.step(closure)
        loss = closure()
        loss_strong_form.append([epoch+1,loss.item()])
        if epoch % l2_eval_freq == 0:
            L2_error = compute_average_L2_error(pn, FEM_sol_dir)
            L2_error_list.append([epoch+1,L2_error.item()])
            print('Epoch: {}, rel. L2 error: {:.2e}'.format(epoch, L2_error))

    print('Training finished.')

    exportList('loss_history/loss_strong_form', loss_strong_form)
    exportList('loss_history/L2_error_strong_form', L2_error_list)
    torch.save(pn,'models/pn_strong_form.pt')

    if plot:
        grid_eval_pinn(geometry)
        plot_shell(geometry, FEM_sol_dir)