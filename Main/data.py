import scipy.io as sio
import numpy as np
import torch

# Define the function for loading the darcy problem dataset
def generate_darcy_dataloader(config):
    '''
    Input:
        config: providing the training configuration
    Output:
        coors: a set of fixed coordinates for PDE predictions    (M, 2)
        data loaders for training, validation and testing
            - each batch consists of (boundary condition values, PDE solution values)
            - boundary condition values shape: (B, M', 3)
            - PDE solution values shape: (B, M)
    '''

    # load the data
    mat_contents = sio.loadmat(r'../data/Darcy_star.mat')
    f_bc = mat_contents['BC_input_var']  # (K, N, 3)
    u = mat_contents['u_field']          # (K, M)
    coor = mat_contents['coor']          # (M, 2)
    BC_flags = mat_contents['IC_flag'].T  # (M, 1)
    num_bc_nodes = f_bc.shape[1]
    print('raw data shape check:', f_bc.shape, u.shape, coor.shape, BC_flags.shape)
    print('number of nodes on boundary:', num_bc_nodes)

    # define dataset
    fbc = torch.tensor(f_bc)    # (K, N, 3)
    sol = torch.tensor(u)       # (K, M)
    coors = torch.tensor(coor)  # (M,2)
    datasize = fbc.shape[0]

    # define data loaders
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(fbc[bar1[0]:bar1[1],:,:], sol[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(fbc[bar2[0]:bar2[1],:,:], sol[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(fbc[bar3[0]:bar3[1],:,:], sol[bar3[0]:bar3[1],:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batchsize'], shuffle=False)

    return coors, BC_flags, num_bc_nodes, train_loader, val_loader, test_loader

# Define the dataset for 
def generate_plate_dataloader(config):

    # load the data
    mat_contents = sio.loadmat(r'../data/plate_dis_high.mat')
    u = mat_contents['final_u'] # (B, M)
    v = mat_contents['final_v'] # (B, M)
    coor = mat_contents['coors'] # (M, 2)
    flag_BCxy = mat_contents['flag_BCxy']  # (M, 1)
    flag_BCy = mat_contents['flag_BCy']  # (M, 1)
    flag_load = mat_contents['flag_BC_load']  # (M, 1)

    # load the scalar factor
    scalar_factor = 1e-4
    youngs = mat_contents['young'][0][0] * scalar_factor
    nu = mat_contents['poisson'][0][0]

    # structure the parameter input
    id_param = np.where(flag_load==1)[0]
    datasize = u.shape[0]
    params = np.concatenate((np.repeat(np.expand_dims(coor[id_param,:],0),datasize,axis=0),
                            np.expand_dims(u[:,id_param],-1)), -1)    # (B, M', 3)
    num_bc_nodes = params.shape[1]

    # define dataset as torch.tensor
    params = torch.tensor(params)    # (B, N, 3)
    u = torch.tensor(u)    # (B, M)
    v = torch.tensor(v)    # (B, N)
    coors = torch.tensor(coor)    # (M,2)

    # define data loader
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(params[bar1[0]:bar1[1],:,:], u[bar1[0]:bar1[1],:], v[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(params[bar2[0]:bar2[1],:,:], u[bar2[0]:bar2[1],:], v[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(params[bar3[0]:bar3[1],:,:], u[bar3[0]:bar3[1],:], v[bar3[0]:bar3[1],:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batchsize'], shuffle=False)

    return coors, train_loader, val_loader, test_loader, youngs, nu, num_bc_nodes, flag_BCxy, flag_BCy, flag_load