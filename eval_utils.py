import numpy as np
import torch
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def grid_eval_pinn(geometry):
    #remember this is without batch, later have to shift everything one dimension higher as first is batch size
    num_points = 100

    if geometry == 'hemisphere':
        radius_grid = torch.linspace(0., 1., num_points).double().to(device) * (1. - 1.e-3)
        theta = torch.linspace(0, 1., num_points).double().to(device)*2.*np.pi
        r_mesh, t_mesh = torch.meshgrid(radius_grid, theta, indexing='ij')
        x_mesh = r_mesh * torch.cos(t_mesh)
        y_mesh = r_mesh * torch.sin(t_mesh)
    else:
        x = torch.linspace(-0.5, 0.5, num_points).double().to(device)
        y = torch.linspace(-0.5, 0.5, num_points).double().to(device)
        x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')

    pn = torch.load('models/pn.pt', map_location=device)
    pn.eval()

    coords = torch.stack((torch.flatten(x_mesh), torch.flatten(y_mesh)),dim=1)

    u0 = pn(coords)[:,0].cpu().detach().numpy()
    u1 = pn(coords)[:,1].cpu().detach().numpy()
    u2 = pn(coords)[:,2].cpu().detach().numpy()
    th1 = pn(coords)[:,3].cpu().detach().numpy()
    th2 = pn(coords)[:,4].cpu().detach().numpy()

    mesh_input = coords.cpu().detach().numpy()

    pinn_pred = np.stack([mesh_input[:,0],mesh_input[:,1],u0,u1,u2,th1,th2],axis=1)
    np.savetxt('eval/pinn_pred.csv', pinn_pred, delimiter=',', header='xi_1,xi_2,u_x,u_y,u_z,th_1,th_2', comments='')

def plot_shell(geometry, FEM_sol_dir):

    if geometry == 'hyperb_parab':
        factor = 0.005
    elif geometry == 'scordelis_lo':
        factor = 0.001
    elif geometry == 'hemisphere':
        factor = 0.05
    else:
        factor = 1.

    data = np.genfromtxt(FEM_sol_dir, delimiter=',', skip_header = 1)
    xi = data[:,0:2]

    data_pinn = np.genfromtxt('eval/pinn_pred.csv', delimiter=',', skip_header = 1)
    xi_p = data_pinn[:,0:2]

    L = 1
    nn = 200
    if geometry == 'hyperb_parab' or geometry == 'scordelis_lo':
        X, Y = np.meshgrid(np.linspace(-L/2, L/2, nn),np.linspace(-L/2, L/2, nn))
    elif geometry == 'hemisphere':
        radius = 1.
        radius_grid = np.linspace(0,radius,nn)*(1.-2.e-3)
        theta = np.linspace(0,2.*np.pi,nn)
        rv, tv = np.meshgrid(radius_grid, theta)
        X = rv * np.cos(tv)
        Y = rv * np.sin(tv)
    else:
        assert False, 'Mapping unknown.'

    FEM_x = griddata(xi, data[:,2].flatten(), (X, Y), method='cubic')
    FEM_y = griddata(xi, data[:,3].flatten(), (X, Y), method='cubic')
    FEM_z = griddata(xi, data[:,4].flatten(), (X, Y), method='cubic')
    FEM_thetas = griddata(xi, np.abs(data[:,5].flatten())+np.abs(data[:,6].flatten()), (X, Y), method='cubic')
    PINN_x = griddata(xi_p, data_pinn[:,2].flatten(), (X, Y), method='cubic')
    PINN_y = griddata(xi_p, data_pinn[:,3].flatten(), (X, Y), method='cubic')
    PINN_z = griddata(xi_p, data_pinn[:,4].flatten(), (X, Y), method='cubic')
    PINN_thetas = griddata(xi_p, np.abs(data_pinn[:,5].flatten())+np.abs(data_pinn[:,6].flatten()), (X, Y), method='cubic')

    def undef(x_mesh, y_mesh, map):
        if map == 'hyperb_parab':
            x = x_mesh
            y = y_mesh
            z = (x_mesh**2 - y_mesh**2)
        elif map == 'scordelis_lo':
            r = 25./50.
            L = 50./50.
            angle = y_mesh*4.*np.pi/9.
            x = L * x_mesh
            y = r * np.sin(angle)
            z = r * np.cos(angle)
        elif map == 'hemisphere':
            x = x_mesh
            y = y_mesh
            z_temp = 1.001-x_mesh**2-y_mesh**2
            z_temp = z_temp.clip(min=0)
            z = np.sqrt(z_temp)
        else:
            False, 'Mapping not implemented for plotting.'
        return x, y, z

    x_undef, y_undef, z_undef = undef(X,Y,geometry)

    color_dimension1 = FEM_thetas*factor
    color_dimension2 = PINN_thetas*factor

    minn, maxx = np.concatenate((color_dimension1,color_dimension2)).min(), np.concatenate((color_dimension1,color_dimension2)).max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='inferno')
    m.set_array([])
    fcolors1 = m.to_rgba(color_dimension1)
    fcolors2 = m.to_rgba(color_dimension2)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,19,(1,8),projection='3d')
    ax2 = fig.add_subplot(1,19,(10,17),projection='3d')
    ax3 = fig.add_subplot(1,19,(18,19),adjustable='box')
    ax4 = fig.add_subplot(1,19,9,adjustable='box')

    ax1.plot_surface(x_undef+FEM_x*factor, y_undef+FEM_y*factor, z_undef+FEM_z*factor, rstride=1, cstride=1, facecolors=fcolors1, vmin=minn, vmax=maxx, shade='auto', rasterized=True)
    if geometry == 'hemisphere':
        ax1.set_title('FEniCS', pad=-20)
        ax1.set_xticks([-1.0,0,1.0])
        ax1.set_yticks([-1.0,0,1.0])
        ax1.set_zticks([0.0,0.35,0.7])
        ax1.view_init(80, 35)
    else:
        ax1.set_title('FEniCS', pad=-50)
        ax1.set_xticks([-0.5,0,0.5])
        ax1.set_yticks([-0.5,0,0.5])
    ax1.tick_params(axis='x', pad=-5)
    ax1.set_xlabel(r'$x_1$',labelpad=-12)
    ax1.tick_params(axis='y',pad=-4)
    ax1.set_ylabel(r'$x_2$',labelpad=-9)
    ax1.tick_params(axis='z',pad=-1)
    ax1.set_zlabel(r'$x_3$',labelpad=-5)

    ax2.plot_surface(x_undef+PINN_x*factor, y_undef+PINN_y*factor, z_undef+PINN_z*factor, rstride=1, cstride=1, facecolors=fcolors2, vmin=minn, vmax=maxx, shade='auto', rasterized=True)
    if geometry == 'hemisphere':
        ax2.set_title('\n\n\n\nPINN', pad=-20)
        ax2.set_xticks([-1.0,0,1.0])
        ax2.set_yticks([-1.0,0,1.0])
        ax2.set_zticks([0.0,0.35,0.7])
        ax2.view_init(80, 35)
    else:
        ax2.set_title('\n\n\n\nPINN', pad=-50)
        ax2.set_xticks([-0.5,0,0.5])
        ax2.set_yticks([-0.5,0,0.5])
    ax2.tick_params(axis='x', pad=-5)
    ax2.set_xlabel(r'$x_1$',labelpad=-12)
    ax2.tick_params(axis='y',pad=-4)
    ax2.set_ylabel(r'$x_2$',labelpad=-9)
    ax2.tick_params(axis='z',pad=-1)
    ax2.set_zlabel(r'$x_3$',labelpad=-5)

    ax4.set_axis_off()
    ax4.set_box_aspect(1.)

    fig.colorbar(m,ax=ax3)
    ax3.set_title(r'$|{{\theta}}|$', x=1.3)
    ax3.set_axis_off()
    ax3.set_box_aspect(4.)
    
    fig.tight_layout()
    plt.show()