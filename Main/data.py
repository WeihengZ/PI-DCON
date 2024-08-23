import scipy.io as sio
import numpy as np
import torch

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
    bar1 = [0,10] # [0,int(0.7*datasize)]
    bar2 = [0,10] # [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [0,10] # [int(0.8*datasize),int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(fbc[bar1[0]:bar1[1],:,:], sol[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(fbc[bar2[0]:bar2[1],:,:], sol[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(fbc[bar3[0]:bar3[1],:,:], sol[bar3[0]:bar3[1],:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batchsize'], shuffle=False)

    return coors, BC_flags, num_bc_nodes, train_loader, val_loader, test_loader