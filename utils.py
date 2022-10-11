import numpy as np
import torch
import torch.autograd.functional as ag
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

def get_col_sampling(col_sampling, N_col):
    if(col_sampling == 'mc'):
        xi_col_temp = torch.cat((torch.rand(N_col,1),torch.rand(N_col,1)),dim=1).to(device)
    elif(col_sampling == 'sobol'):
        soboleng = torch.quasirandom.SobolEngine(dimension=2)
        xi_col_temp = soboleng.draw(N_col).to(device)
    elif(col_sampling == 'rectGrid'):
        x = torch.linspace(0, 1, N_col).cpu()
        y = torch.linspace(0, 1, N_col).cpu()
        x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')
        mesh_input_temp = torch.stack((x_mesh,y_mesh),dim=2)
        xi_col_temp = torch.reshape(mesh_input_temp,(N_col**2,2)).to(device)
    elif(col_sampling == 'concentric'):
        xs = torch.linspace(-1, 1, steps=N_col)
        ys = torch.linspace(-1, 1, steps=N_col)
        x, y = torch.meshgrid(xs, ys, indexing='ij')
        xi_col_temp = torch.transpose(torch.stack([torch.flatten(x), torch.flatten(y)]),0,1)
        r = torch.zeros(len(xi_col_temp),device=device)
        phi = torch.zeros(len(xi_col_temp),device=device)
        for i in range(len(xi_col_temp)):
            if xi_col_temp[i,0] > -xi_col_temp[i,1]:
                if xi_col_temp[i,0] > xi_col_temp[i,1]:
                    r[i] = xi_col_temp[i,0]
                    phi[i] = (np.pi/4.) * (xi_col_temp[i,1]/xi_col_temp[i,0])
                else: 
                    r[i] = xi_col_temp[i,1]
                    phi[i] = (np.pi/4.) * (2. - xi_col_temp[i,0]/xi_col_temp[i,1])
            else:
                if xi_col_temp[i,0] < xi_col_temp[i,1]:
                    r[i] = -xi_col_temp[i,0]
                    phi[i] = (np.pi/4.) * (4. + (xi_col_temp[i,1]/xi_col_temp[i,0]))
                else:
                    r[i] = -xi_col_temp[i,1]
                    if (xi_col_temp[i,1] != 0.):
                        phi[i] = (np.pi/4.) * (6. - (xi_col_temp[i,0]/xi_col_temp[i,1]))
                    else:
                        phi[i] = 0.
        polar_coords = torch.stack((r,phi),dim=1)
        xi_col_temp = torch.zeros(N_col**2,2,device=device)
        xi_col_temp[:,0] = polar_coords[:,0] * torch.cos(polar_coords[:,1])
        xi_col_temp[:,1] = polar_coords[:,0] * torch.sin(polar_coords[:,1])
    else:
        raise ValueError('col_sampling not defined/known.')
    return xi_col_temp

def getActivation(activ):
    if(activ == 'relu'):
        sigma = torch.nn.ReLU()
    elif(activ == 'tanh'):
        sigma = torch.nn.Tanh()
    elif(activ == 'sigmoid'):
        sigma = torch.nn.Sigmoid()
    elif(activ == 'leaky'):
        sigma = torch.nn.LeakyReLU()
    elif(activ == 'softplus'):
        sigma = torch.nn.Softplus()
    elif(activ == 'logsigmoid'):
        sigma = torch.nn.LogSigmoid()
    elif(activ == 'elu'):
        sigma = torch.nn.ELU()
    elif(activ == 'gelu'):
        sigma = torch.nn.GELU()
    elif(activ == 'none'):
        sigma = torch.nn.Identity()
    else:
        raise ValueError('Incorrect activation function')
    return sigma

# batched matrix-vector multiplication
def bmv(matrix,vector):
    return torch.einsum('bij,bj->bi',matrix,vector)

# batched vector dot product
def bdot(vector1,vector2):
    return torch.einsum('bi,bi->b',vector1,vector2)

def exportList(name,data):
    np.savetxt(name+".csv", np.array(data), delimiter=',')

def exportLists(name,thickness_list,pred_list,inner_energy_list,work_list,imbalance_list,true_imbalance_list):
    thickness=mean_per_epoch(thickness_list)
    pred=mean_per_epoch(pred_list)
    inner_energy=mean_per_epoch(inner_energy_list)
    work=mean_per_epoch(work_list)
    imbalance=mean_per_epoch(imbalance_list)
    true_imbalance =mean_per_epoch(true_imbalance_list)
    temp = np.stack((np.arange(len(pred)),thickness,pred,inner_energy,work,imbalance,true_imbalance),axis=1)
    np.savetxt(name+".csv", temp, delimiter=',', header='Epoch,t,u_z,Energy,Work,Unscaled Imbalance,Scaled Imbalance',comments='')

def mean_per_epoch(list):
    arr = np.array(list)
    j = 0
    k = 1
    red_arr = []
    for i in range(len(arr)):
        if i == 0:
            red_arr.append(arr[i,1])
        elif abs(arr[i,0] - arr[i-1,0])<0.0001:
            red_arr[j] += arr[i,1]
            k += 1
            if i == len(arr)-1:
                red_arr[j] /= k    
        else:
            red_arr[j] /= k
            red_arr.append(arr[i,1])
            j += 1
            k = 1
    return np.array(red_arr)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

# Glorot initialization of weight matrix
def glorot_init_mat(shape):
    din = shape[0]
    dout = shape[1]
    var = torch.tensor([2.0/(din+dout)])
    std = torch.sqrt(var)
    mean = torch.tensor([0.0])
    dist = torch.distributions.normal.Normal(mean, std)
    return dist.sample(shape)

# for symmetric tensors
def full_to_Voigt(tensor):
    batch_len = len(tensor)
    shape = list(tensor.shape)
    if (len(shape) == 3):
        voigt = torch.zeros(batch_len,3,device=device)
        voigt[:,0] = tensor[:,0,0]
        voigt[:,1] = tensor[:,1,1]
        voigt[:,2] = 2.*tensor[:,0,1]
    elif (len(shape) == 4):
        voigt = torch.zeros(batch_len,2,3,device=device)
        voigt[:,:,0] = tensor[:,:,0,0]
        voigt[:,:,1] = tensor[:,:,1,1]
        voigt[:,:,2] = 2.*tensor[:,:,0,1]
    return voigt

# for symmetric tensors
def Voigt_to_full(voigt):
    batch_len = len(voigt)
    shape = list(voigt.shape)
    if (len(shape) == 2):
        tensor = torch.zeros(batch_len,2,2,device=device)
        tensor[:,0,0] = voigt[:,0]
        tensor[:,1,1] = voigt[:,1]
        tensor[:,0,1] = 0.5*voigt[:,2]
        tensor[:,1,0] = 0.5*voigt[:,2]
    if (len(shape) == 3):
        tensor = torch.zeros(batch_len,2,2,2,device=device)
        tensor[:,:,0,0] = voigt[:,:,0]
        tensor[:,:,1,1] = voigt[:,:,1]
        tensor[:,:,0,1] = 0.5*voigt[:,:,2]
        tensor[:,:,1,0] = 0.5*voigt[:,:,2]
    return tensor

def div_1st_order_u(vector,vector_d, christoffel):
    vector_d_correct = torch.einsum('...ji->...ij',vector_d)
    temp = torch.einsum('...ii->...',vector_d_correct) + torch.einsum('...ili,...l->...',christoffel,vector)
    return temp

# summation over first index
def div_2nd_order_u1(tensor,tensor_d,christoffel):
    tensor_d_correct = torch.einsum('...kij->...ijk',tensor_d)
    a = torch.einsum('...iji->...j',tensor_d_correct)
    b = torch.einsum('...iil,...lj->...j',christoffel,tensor)
    c = torch.einsum('...jil,...il->...j',christoffel,tensor)
    return a+b+c

# summation over second index (as defined in paper)
def div_2nd_order_u2(tensor,tensor_d,christoffel):
    tensor_d_correct = torch.einsum('...kij->...ijk',tensor_d)
    a = torch.einsum('...jii->...j',tensor_d_correct)
    b = torch.einsum('...iil,...jl->...j',christoffel,tensor)
    c = torch.einsum('...jil,...li->...j',christoffel,tensor)
    return a+b+c

# input (2,0) output (1,2)
def grad_2nd_order_udd(tensor,tensor_d, christoffel, con_metric):
    # bring derivative to last index
    tensor_d_correct = torch.einsum('...kij->...ijk',tensor_d)
    # bring dd to ud:
    tensor_ud = torch.einsum('...ij,...ik->...kj',tensor,con_metric)
    tensor_d_correct_ud = torch.einsum('...ijk,...il->...ljk',tensor_d_correct,con_metric)
    temp = tensor_d_correct_ud - torch.einsum('...ijk,...li->...ljk',christoffel,tensor_ud) + torch.einsum('...ijk,...kl->...ilj',christoffel,tensor_ud)
    return temp

# input (2,0) output (1,2)
def grad_2nd_order_uud(tensor,tensor_d, christoffel):
    # bring derivative to last index
    tensor_d_correct = torch.einsum('...kij->...ijk',tensor_d)
    temp = tensor_d_correct + torch.einsum('...ikl,...lj->...ijk',christoffel,tensor) + torch.einsum('...jkl,...il->...ijk',christoffel,tensor)
    return temp

def _grad_preprocess_inplace(inputs, create_graph, need_graph):
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Create at least a new Tensor object in a differentiable way
            if not inp.is_sparse:
                # Since we do not want to distinguish between precomputed and input arguments!
                res.append(inp)
            else:
                # We cannot use view for sparse Tensors so we clone
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)

# small workarond so that Pytorch's vjp to consider precomputed tensors (such as geometrical parameters) in the derivatives 
def vjp_inplace(func, inputs, v=None, create_graph=False, strict=False):
    is_inputs_tuple, inputs = ag._as_tuple(inputs, "inputs", "vjp")
    inputs = _grad_preprocess_inplace(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = ag._as_tuple(outputs, "outputs of the user-provided function", "vjp")
    ag._check_requires_grad(outputs, "outputs", strict=strict)

    if v is not None:
        _, v = ag._as_tuple(v, "v", "vjp")
        v = ag._grad_preprocess(v, create_graph=create_graph, need_graph=False)
        ag._validate_v(v, outputs, is_outputs_tuple)
    else:
        if len(outputs) != 1 or outputs[0].nelement() != 1:
            raise RuntimeError("The vector v can only be None if the "
                               "user-provided function returns "
                               "a single Tensor with a single element.")

    grad_res = ag._autograd_grad(outputs, inputs, v, create_graph=create_graph)

    vjp = ag._fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = ag._grad_postprocess(outputs, create_graph)
    vjp = ag._grad_postprocess(vjp, create_graph)

    return ag._tuple_postprocess(outputs, is_outputs_tuple), ag._tuple_postprocess(vjp, is_inputs_tuple)

def plot_points(xi_col):
    xi_col_np = xi_col.cpu().detach().numpy()
    plt.scatter(xi_col_np[:,0],xi_col_np[:,1],s=1)
    plt.show()

def fix_seeds():
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234) 

def compute_average_L2_error(pn,data):

    data = np.genfromtxt(data, delimiter=',', skip_header = 1)
    xi = data[:,0:2]

    num_points = 100
    x = torch.linspace(-0.5, 0.5, num_points).to(device)
    y = torch.linspace(-0.5, 0.5, num_points).to(device)
    x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')

    coords = torch.stack((torch.flatten(x_mesh), torch.flatten(y_mesh)),dim=1)

    u0 = pn(coords)[:,0].cpu().detach().numpy()
    u1 = pn(coords)[:,1].cpu().detach().numpy()
    u2 = pn(coords)[:,2].cpu().detach().numpy()
    th1 = pn(coords)[:,3].cpu().detach().numpy()
    th2 = pn(coords)[:,4].cpu().detach().numpy()

    data_pinn = np.stack([coords[:,0].cpu().detach().numpy(),coords[:,1].cpu().detach().numpy(),u0,u1,u2,th1,th2],axis=1)

    xi_p = data_pinn[:,0:2]

    L = 1
    nn = 200
    x = np.linspace(-L/2, L/2, nn)
    y = np.linspace(-L/2, L/2, nn)

    X, Y = np.meshgrid(x,y, indexing='ij')

    error = np.zeros(5)
    for i in range(5):
        U_star = griddata(xi, data[:,i+2].flatten(), (X, Y), method='cubic')
        U_star_p = griddata(xi_p, data_pinn[:,i+2].flatten(), (X, Y), method='cubic')
        error[i] = np.linalg.norm(U_star-U_star_p,2)/np.linalg.norm(U_star,2)

    return np.mean(error)

def create_param_dict(study):
    # partly clamped hyperbolic paraboloid (weak form)
    if study == 'hyperb_parab':
        param_dict = {
            'geometry': 'hyperb_parab', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.1,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'ls',
            'N_col': 16384,
            'col_sampling': 'sobol',
            'epochs': 100,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hyperb_parab.csv',
            }
    # fully clamped hyperbolic paraboloid (strong form)
    elif study == 'hyperb_parab_strong':
        param_dict = {
            'geometry': 'hyperb_parab', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.1,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'fc',
            'N_col': 2048,
            'col_sampling': 'sobol',
            # 'epochs': 100,
            'epochs': 10,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hyperb_parab_fully_clamped.csv',
            }
    # Scordelis-Lo roof (weak form)
    elif study == 'scordelis_lo':
        param_dict = {
            'geometry': 'scordelis_lo', 
            'loading': 'gravity',
            'E': 1.,
            'thickness': 0.005,
            'shell_density': 1.,
            'nu': 0.0,
            'shear_factor': 5./6.,
            'bcs': 'scordelis_lo',
            'N_col': 2**16,
            'col_sampling': 'sobol',
            'epochs': 1000,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_scordelis_lo.csv',
            }
    # hemisphere under concentrated load (weak form)
    elif study == 'hemisphere':
        param_dict = {
            'geometry': 'hemisphere', 
            'loading': 'concentrated_load',
            'E': 1.,
            'thickness': 0.05,
            'shell_density': 1.,
            'nu': 0.3,
            'shear_factor': 5./6.,
            'bcs': 'fc_circular',
            'N_col': 280,
            'col_sampling': 'concentric',
            'epochs': 100,
            'opt_switch_epoch': 0,
            'FEM_sol_dir': 'FEM_sol/fenics_pred_hemisphere.csv',
            }
    else:
        raise ValueError('Invalid benchmark')

    return param_dict